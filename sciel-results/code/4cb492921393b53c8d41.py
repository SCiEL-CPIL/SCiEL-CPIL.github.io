import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor T[i, j, k] = 1 if it's a valid multiplication term
    # T is indexed by (u_idx, v_idx, w_idx)
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        # Compute sum_r U[:,r] V[:,r]^T W[:,r]^T
        # Using einsum: 'ar,br,cr->abc'
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def try_snap_validate(U, V, W):
        # Try multiple quantization levels with increasing precision
        for denom in [1, 2, 4, 8, 16, 32]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        # If all fail, return the best continuous solution as fallback
        return False, U, V, W

    def build_rank46():
        # Standard construction: 4 Strassen blocks (2x2x2) + border terms
        # Strassen for 2x2x2 uses 7 multiplications.
        # We have 2x5x5.
        # Partition j into 0..4.
        # Use Strassen on (j=0,1) and (j=2,3).
        # The remaining j=4 is handled by direct multiplication.
        
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        # Apply Strassen to blocks (j=0,1; k=0,1), (j=0,1; k=2,3), (j=2,3; k=0,1), (j=2,3; k=2,3)
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        # Handle remaining terms involving j=4 or k=4
        # j=4, k=0..4
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        # j=0..3, k=4
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        # j=4, k=4
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    def optimize_rank(U_init, V_init, W_init, target_rank, max_iter=300):
        # Use Adam optimizer with better convergence
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = 0.1
        
        # Initialize moments
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if loss < 1e-10:
                break
                
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Update moments
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * gradU ** 2
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * gradV ** 2
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * gradW ** 2
            
            # Bias correction
            bc1 = 1 - beta1 ** (it + 1)
            bc2 = 1 - beta2 ** (it + 1)
            
            U -= lr * mU / (bc1 * np.sqrt(vU / bc2) + eps)
            V -= lr * mV / (bc1 * np.sqrt(vV / bc2) + eps)
            W -= lr * mW / (bc1 * np.sqrt(vW / bc2) + eps)
            
            # Decay learning rate
            lr *= 0.9995
            
        valid, Us, Vs, Ws = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return Us, Vs, Ws
        return None, None, None

    # Initial construction
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to rank 50
        rank = 50
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    np.random.seed(42)
    
    # Strategy: Try to remove columns with better search
    # Rank 45: Try removing each column
    for c_remove in range(46):
        if time.time() > deadline - 30:
            break
        mask = [i for i in range(46) if i != c_remove]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        # Add small noise to break symmetry
        U_test += np.random.randn(*U_test.shape) * 0.005
        V_test += np.random.randn(*V_test.shape) * 0.005
        W_test += np.random.randn(*W_test.shape) * 0.005
        
        U_opt, V_opt, W_opt = optimize_rank(U_test, V_test, W_test, 45, max_iter=400)
        if U_opt is not None:
            best_result = (U_opt, V_opt, W_opt)
            break
            
    # Rank 44: Try removing pairs of columns
    if best_result is None and time.time() < deadline - 50:
        # Try all pairs, but be smart about which ones to try first
        # Columns that are "similar" are more likely to be removable together
        for c1 in range(46):
            for c2 in range(c1+1, 46):
                if time.time() > deadline - 40:
                    break
                mask = [i for i in range(46) if i != c1 and i != c2]
                U_test = U46[:, mask].copy()
                V_test = V46[:, mask].copy()
                W_test = W46[:, mask].copy()
                
                U_test += np.random.randn(*U_test.shape) * 0.01
                V_test += np.random.randn(*V_test.shape) * 0.01
                W_test += np.random.randn(*W_test.shape) * 0.01
                
                U_opt, V_opt, W_opt = optimize_rank(U_test, V_test, W_test, 44, max_iter=300)
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
            if best_result is not None:
                break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }

    # Fallback: return rank 46 solution
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
