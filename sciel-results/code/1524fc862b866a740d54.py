import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None
    
    def build_rank46():
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
        
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W
    
    def optimize_to_rank(U_init, V_init, W_init, target_rank, max_iter=500, lr=0.03, momentum=0.9):
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        prev_loss = float('inf')
        no_improve = 0
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = prev_loss
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U -= vU
            V -= vV
            W -= vW
            
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    break
            
            if loss < 1e-8:
                break
        
        # Check if best solution snaps to valid
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s
        
        # Also check current
        valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
        if valid:
            return U_s, V_s, W_s
        
        return None, None, None
    
    def polish(U, V, W, max_iter=200, lr=0.01):
        """High-precision polish pass to tighten the solution."""
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW
            if np.sum(diff*diff) < 1e-12:
                break
        return U, V, W
    
    # First, verify rank-46 works
    U46, V46, W46 = build_rank46()
    
    if not validate(U46, V46, W46):
        # Fallback: schoolbook
        rank = n * m * p
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
    
    # Try to find rank-45 solution
    best_rank = 46
    best_result = {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
    
    # Strategy 1: Try zeroing different single columns and re-optimizing
    for restart in range(3):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(100 + restart)
        
        for col_to_zero in range(46):
            if time.time() > deadline - 10:
                break
            
            # Remove column col_to_zero
            mask = np.ones(46, dtype=bool)
            mask[col_to_zero] = False
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            # Add small noise
            U_test += np.random.randn(*U_test.shape) * 0.005
            V_test += np.random.randn(*V_test.shape) * 0.005
            W_test += np.random.randn(*W_test.shape) * 0.005
            
            U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 45, max_iter=300, lr=0.02, momentum=0.9)
            
            if U_opt is not None:
                # Polish
                U_pol, V_pol, W_pol = polish(U_opt, V_opt, W_opt, max_iter=100, lr=0.005)
                U_s = snap_to_half(U_pol)
                V_s = snap_to_half(V_pol)
                W_s = snap_to_half(W_pol)
                if validate(U_s, V_s, W_s):
                    return {
                        "U": [[float(x) for x in row] for row in U_s],
                        "V": [[float(x) for x in row] for row in V_s],
                        "W": [[float(x) for x in row] for row in W_s]
                    }
    
    # Strategy 2: Try removing two columns (rank 44)
    if time.time() < deadline - 20:
        for restart in range(2):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(200 + restart)
            
            for c1 in range(46):
                for c2 in range(c1+1, 46):
                    if time.time() > deadline - 10:
                        break
                    
                    mask = np.ones(46, dtype=bool)
                    mask[c1] = False
                    mask[c2] = False
                    U_test = U46[:, mask].copy()
                    V_test = V46[:, mask].copy()
                    W_test = W46[:, mask].copy()
                    
                    U_test += np.random.randn(*U_test.shape) * 0.003
                    V_test += np.random.randn(*V_test.shape) * 0.003
                    W_test += np.random.randn(*W_test.shape) * 0.003
                    
                    U_opt, V_opt, W_opt = optimize_to_rank(U_test, V_test, W_test, 44, max_iter=200, lr=0.015, momentum=0.9)
                    
                    if U_opt is not None:
                        U_s = snap_to_half(U_opt)
                        V_s = snap_to_half(V_opt)
                        W_s = snap_to_half(W_opt)
                        if validate(U_s, V_s, W_s):
                            return {
                                "U": [[float(x) for x in row] for row in U_s],
                                "V": [[float(x) for x in row] for row in V_s],
                                "W": [[float(x) for x in row] for row in W_s]
                            }
                    if time.time() > deadline - 12:
                        break
    
    # Strategy 3: Random restarts with different initializations
    if time.time() < deadline - 15:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(300 + restart)
            
            # Create random rank-45 initialization
            U_init = np.random.randn(U_dim, 45) * 0.1
            V_init = np.random.randn(V_dim, 45) * 0.1
            W_init = np.random.randn(W_dim, 45) * 0.1
            
            # Mix with the rank-46 solution (drop a random column)
            drop_col = np.random.randint(46)
            mask = np.ones(46, dtype=bool)
            mask[drop_col] = False
            U_init = 0.7 * U46[:, mask] + 0.3 * U_init
            V_init = 0.7 * V46[:, mask] + 0.3 * V_init
            W_init = 0.7 * W46[:, mask] + 0.3 * W_init
            
            U_opt, V_opt, W_opt = optimize_to_rank(U_init, V_init, W_init, 45, max_iter=400, lr=0.02, momentum=0.9)
            
            if U_opt is not None:
                U_s = snap_to_half(U_opt)
                V_s = snap_to_half(V_opt)
                W_s = snap_to_half(W_opt)
                if validate(U_s, V_s, W_s):
                    return {
                        "U": [[float(x) for x in row] for row in U_s],
                        "V": [[float(x) for x in row] for row in V_s],
                        "W": [[float(x) for x in row] for row in W_s]
                    }
    
    # Return rank-46 solution
    return best_result
# EVOLVE-BLOCK-END
