import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
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
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=120, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.999):
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            if loss > prev_loss * 1.001:
                no_improve += 1
                if no_improve > 15:
                    break
            else:
                prev_loss = loss
                no_improve = 0
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= current_lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= current_lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            current_lr *= lr_decay
        
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s
        
        # Final polish: try tighter tolerance
        if best_loss < 1e-4:
            # High-precision polish
            U2 = best_U.copy()
            V2 = best_V.copy()
            W2 = best_W.copy()
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                prod = np.einsum('ar,br,cr->abc', U2, V2, W2, optimize=True)
                diff = prod - T
                loss = np.sum(diff * diff)
                if loss < 1e-12:
                    break
                gradU = np.einsum('abc,br,cr->ar', diff, V2, W2, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U2, W2, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U2, V2, optimize=True)
                U2 -= 0.001 * gradU
                V2 -= 0.001 * gradV
                W2 -= 0.001 * gradW
            valid, U_s, V_s, W_s = try_snap_validate(U2, V2, W2)
            if valid:
                return U_s, V_s, W_s
        
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
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
    best_result = None
    col_nonzeros_U = np.count_nonzero(U46, axis=0)
    col_nonzeros_V = np.count_nonzero(V46, axis=0)
    col_nonzeros_W = np.count_nonzero(W46, axis=0)
    col_complexity = col_nonzeros_U + col_nonzeros_V + col_nonzeros_W
    col_order = np.argsort(col_complexity)
    
    np.random.seed(42)
    
    # Strategy 1: Try removing single columns (rank 45)
    max_attempts = 46
    for attempt in range(max_attempts):
        if time.time() > deadline - 30:
            break
        if attempt >= len(col_order):
            break
            
        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        # Try multiple noise scales
        found = False
        for noise_scale in [0.001, 0.01, 0.05]:
            if time.time() > deadline - 25:
                break
            U_try = U_test.copy()
            V_try = V_test.copy()
            W_try = W_test.copy()
            U_try += np.random.randn(*U_try.shape) * noise_scale
            V_try += np.random.randn(*V_try.shape) * noise_scale
            W_try += np.random.randn(*W_try.shape) * noise_scale
            
            U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=100, lr=0.03)
            
            if U_opt is not None:
                best_result = (U_opt, V_opt, W_opt)
                found = True
                break
        
        if found:
            break
    
    # Strategy 2: If rank 45 failed, try removing pairs of columns (rank 44)
    if best_result is None and time.time() < deadline - 40:
        # Try removing the two simplest columns
        for pair_offset in range(min(10, len(col_order) - 1)):
            if time.time() > deadline - 35:
                break
            c1 = col_order[pair_offset]
            c2 = col_order[pair_offset + 1]
            mask = [i for i in range(46) if i != c1 and i != c2]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for noise_scale in [0.005, 0.02]:
                if time.time() > deadline - 30:
                    break
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * noise_scale
                V_try += np.random.randn(*V_try.shape) * noise_scale
                W_try += np.random.randn(*W_try.shape) * noise_scale
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 44, max_iter=80, lr=0.02)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
            if best_result is not None:
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        # Verify the result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
