import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor T[i, j, k] = 1 if it's a valid multiplication term
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

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def try_snap_validate(U, V, W, denom=8):
        # Try snapping to multiples of 1/denom
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
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
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
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

    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
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

    # Strategy: Try to find rank 45 by warm-starting from rank 46
    # Zero out one column and re-optimize, or use random init with better structure
    np.random.seed(42)
    
    def optimize_tensor(U_init, V_init, W_init, max_iter=1000, lr_init=0.05):
        """Adam-like optimizer for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        best_loss = compute_loss(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = lr_init
        
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        t = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            t += 1
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-14:
                break
            
            # Compute gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * gradU
            mV = beta1 * mV + (1 - beta1) * gradV
            mW = beta1 * mW + (1 - beta1) * gradW
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            
            mU_hat = mU / (1 - beta1 ** t)
            mV_hat = mV / (1 - beta1 ** t)
            mW_hat = mW / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            lr *= 0.995

        return best_U, best_V, best_W, best_loss

    def local_search_discrete(U, V, W, max_steps=2000):
        """Try discrete moves: change one entry at a time to nearby rational values."""
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        cur_loss = compute_loss(U_cur, V_cur, W_cur)
        
        candidates = [-1, -0.5, 0, 0.5, 1]
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try changing a random entry in U
            for _ in range(3):
                r = np.random.randint(0, U_dim)
                c = np.random.randint(0, U_cur.shape[1])
                old_val = U_cur[r, c]
                for new_val in candidates:
                    if abs(new_val - old_val) < 1e-10:
                        continue
                    U_cur[r, c] = new_val
                    new_loss = compute_loss(U_cur, V_cur, W_cur)
                    if new_loss < cur_loss - 1e-10:
                        cur_loss = new_loss
                        improved = True
                        break
                    else:
                        U_cur[r, c] = old_val
            
            # Try changing a random entry in V
            for _ in range(3):
                r = np.random.randint(0, V_dim)
                c = np.random.randint(0, V_cur.shape[1])
                old_val = V_cur[r, c]
                for new_val in candidates:
                    if abs(new_val - old_val) < 1e-10:
                        continue
                    V_cur[r, c] = new_val
                    new_loss = compute_loss(U_cur, V_cur, W_cur)
                    if new_loss < cur_loss - 1e-10:
                        cur_loss = new_loss
                        improved = True
                        break
                    else:
                        V_cur[r, c] = old_val
            
            # Try changing a random entry in W
            for _ in range(3):
                r = np.random.randint(0, W_dim)
                c = np.random.randint(0, W_cur.shape[1])
                old_val = W_cur[r, c]
                for new_val in candidates:
                    if abs(new_val - old_val) < 1e-10:
                        continue
                    W_cur[r, c] = new_val
                    new_loss = compute_loss(U_cur, V_cur, W_cur)
                    if new_loss < cur_loss - 1e-10:
                        cur_loss = new_loss
                        improved = True
                        break
                    else:
                        W_cur[r, c] = old_val
            
            if not improved and step > 100:
                # Try larger perturbations
                for _ in range(3):
                    r = np.random.randint(0, U_dim)
                    c = np.random.randint(0, U_cur.shape[1])
                    U_cur[r, c] = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
                    new_loss = compute_loss(U_cur, V_cur, W_cur)
                    if new_loss < cur_loss - 1e-10:
                        cur_loss = new_loss
                        improved = True
                    else:
                        U_cur[r, c] = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
                
                if not improved:
                    break
        
        return U_cur, V_cur, W_cur, cur_loss

    best_result = None
    
    # Approach 1: Warm start from rank-46 by zeroing one column
    # Try each column of rank-46 and see if we can recover with rank-45
    for col_to_zero in range(46):
        if time.time() > deadline - 30:
            break
        
        U_init = U46[:, :].copy()
        V_init = V46[:, :].copy()
        W_init = W46[:, :].copy()
        
        # Remove one column
        U_init = np.delete(U_init, col_to_zero, axis=1)
        V_init = np.delete(V_init, col_to_zero, axis=1)
        W_init = np.delete(W_init, col_to_zero, axis=1)
        
        # Optimize
        U_opt, V_opt, W_opt, loss = optimize_tensor(U_init, V_init, W_opt if False else W_init, max_iter=300, lr_init=0.03)
        
        if loss < 1e-10:
            valid, Us, Vs, Ws = try_snap_validate(U_opt, V_opt, W_opt, denom=8)
            if valid:
                best_result = (Us, Vs, Ws)
                break
        
        # Also try local search
        U_ls, V_ls, W_ls, ls_loss = local_search_discrete(U_opt, V_opt, W_opt, max_steps=500)
        if ls_loss < 1e-10:
            valid, Us, Vs, Ws = try_snap_validate(U_ls, V_ls, W_ls, denom=8)
            if valid:
                best_result = (Us, Vs, Ws)
                break
    
    # Approach 2: Random sparse initialization with better structure
    if best_result is None:
        for trial in range(30):
            if time.time() > deadline - 25:
                break
            
            U_rand = np.zeros((U_dim, 45))
            V_rand = np.zeros((V_dim, 45))
            W_rand = np.zeros((W_dim, 45))
            
            for r in range(45):
                # Each factor should have a few non-zeros
                n_u = np.random.choice([2, 3, 4])
                n_v = np.random.choice([2, 3, 4])
                n_w = np.random.choice([2, 3, 4])
                
                idx_u = np.random.choice(U_dim, size=n_u, replace=False)
                idx_v = np.random.choice(V_dim, size=n_v, replace=False)
                idx_w = np.random.choice(W_dim, size=n_w, replace=False)
                
                U_rand[idx_u, r] = np.random.choice([-1, 1], size=n_u)
                V_rand[idx_v, r] = np.random.choice([-1, 1], size=n_v)
                W_rand[idx_w, r] = np.random.choice([-1, 1], size=n_w)
            
            U_opt, V_opt, W_opt, loss = optimize_tensor(U_rand, V_rand, W_rand, max_iter=200, lr_init=0.05)
            
            if loss < 1e-10:
                valid, Us, Vs, Ws = try_snap_validate(U_opt, V_opt, W_opt, denom=8)
                if valid:
                    best_result = (Us, Vs, Ws)
                    break
            
            # Local search
            U_ls, V_ls, W_ls, ls_loss = local_search_discrete(U_opt, V_opt, W_opt, max_steps=300)
            if ls_loss < 1e-10:
                valid, Us, Vs, Ws = try_snap_validate(U_ls, V_ls, W_ls, denom=8)
                if valid:
                    best_result = (Us, Vs, Ws)
                    break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
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
