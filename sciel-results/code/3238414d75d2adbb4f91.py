import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
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
    
    def snap_to_grid(arr, denom):
        return np.round(arr * denom) / denom
    
    def try_snap_validate(U, V, W, max_denom=16):
        for denom in [1, 2, 4, 8, 16]:
            if denom > max_denom:
                break
            U_s = snap_to_grid(U, denom)
            V_s = snap_to_grid(V, denom)
            W_s = snap_to_grid(W, denom)
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
            # P7 = (A10 + A11)(B01 + B11) -- wait, let me re-check
            # Actually P7 = (A01 - A11)(B10 + B11)
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

    def try_remove_column(U, V, W, col_idx, denom=2):
        """
        Try to absorb column col_idx into the other columns.
        For each entry in column col_idx, we need to distribute its contribution
        to other columns. This is a linear feasibility problem.
        
        We use a simpler heuristic: for each nonzero in the column, try to find
        another column that shares the same U, V, or W index and can absorb it.
        """
        R = U.shape[1]
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Get the column to remove
        u_col = U[:, col_idx].copy()
        v_col = V[:, col_idx].copy()
        w_col = W[:, col_idx].copy()
        
        # For each position (a,b,c) where the tensor has a 1, check if removing
        # this column still allows reconstruction. The contribution of column r
        # to T[a,b,c] is U[a,r]*V[b,r]*W[c,r].
        
        # We need to adjust other columns to compensate.
        # Simple approach: for each nonzero in u_col, find a column that has
        # the same b and c indices in V and W, and add to it.
        
        # Actually, let's use a linear algebra approach.
        # The residual after removing column r is:
        # R[a,b,c] = -U[a,r]*V[b,r]*W[c,r]
        # We need to find deltas dU, dV, dW such that:
        # sum_r' (dU[a,r']*V[b,r']*W[c,r'] + U[a,r']*dV[b,r']*W[c,r'] + U[a,r']*V[b,r']*dW[c,r']) = R[a,b,c]
        
        # This is complex. Instead, use a greedy approach:
        # Try to absorb each nonzero of the column into existing columns.
        
        u_nz = np.nonzero(u_col)[0]
        v_nz = np.nonzero(v_col)[0]
        w_nz = np.nonzero(w_col)[0]
        
        # The column contributes to positions (a,b,c) where a in u_nz, b in v_nz, c in w_nz
        # with value u_col[a]*v_col[b]*w_col[c]
        
        # For each such position, find another column that can cover it
        absorbed = True
        for a in u_nz:
            for b in v_nz:
                for c in w_nz:
                    val = u_col[a] * v_col[b] * w_col[c]
                    if abs(val) < 1e-10:
                        continue
                    # Find a column r' != col_idx where U[a,r'], V[b,r'], W[c,r'] are all nonzero
                    found = False
                    for r_prime in range(R):
                        if r_prime == col_idx:
                            continue
                        if abs(U[a, r_prime]) > 1e-10 and abs(V[b, r_prime]) > 1e-10 and abs(W[c, r_prime]) > 1e-10:
                            # We can adjust U[a,r_prime] to compensate
                            # Need: dU * V[b,r'] * W[c,r'] = -val
                            # dU = -val / (V[b,r'] * W[c,r'])
                            dU = -val / (V[b, r_prime] * W[c, r_prime])
                            U_new[a, r_prime] += dU
                            found = True
                            break
                    if not found:
                        absorbed = False
                        break
                if not absorbed:
                    break
            if not absorbed:
                break
        
        if not absorbed:
            return False, None, None, None
        
        # Zero out the column
        U_new[:, col_idx] = 0
        V_new[:, col_idx] = 0
        W_new[:, col_idx] = 0
        
        # Snap and validate
        for d in [1, 2, 4, 8, 16]:
            U_s = snap_to_grid(U_new, d)
            V_s = snap_to_grid(V_new, d)
            W_s = snap_to_grid(W_new, d)
            if validate(U_s, V_s, W_s):
                # Remove the zero column
                mask = [i for i in range(R) if i != col_idx]
                return True, U_s[:, mask], V_s[:, mask], W_s[:, mask]
        
        return False, None, None, None

    def newton_polish(U, V, W, max_iter=50, tol=1e-14):
        """Proper Gauss-Newton with line search for tensor decomposition."""
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break
            
            # Gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Line search
            alpha = 1.0
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            for _ in range(30):
                U_new = U - alpha * gradU
                V_new = V - alpha * gradV
                W_new = W - alpha * gradW
                prod_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new, optimize=True)
                loss_new = np.sum((prod_new - T) ** 2)
                if loss_new < best_loss:
                    best_loss = loss_new
                    best_U, best_V, best_W = U_new, V_new, W_new
                alpha *= 0.5
                if alpha < 1e-12:
                    break
            
            U, V, W = best_U, best_V, best_W
        
        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=200, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.999):
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
        restart_count = 0
        
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
            
            if loss > prev_loss * 1.005 and no_improve > 15:
                restart_count += 1
                if restart_count > 5:
                    break
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                continue
            else:
                no_improve += 1
                prev_loss = loss
            
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
        
        if best_loss < 1e-2:
            U2, V2, W2 = newton_polish(best_U, best_V, best_W, max_iter=40, tol=1e-14)
            valid, U_s, V_s, W_s = try_snap_validate(U2, V2, W2)
            if valid:
                return U_s, V_s, W_s
        
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
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

    best_result = None
    np.random.seed(42)
    
    # Strategy 1: Try removing single columns using greedy absorption
    for col_idx in range(46):
        if time.time() > deadline - 60:
            break
        success, U_s, V_s, W_s = try_remove_column(U46, V46, W46, col_idx)
        if success:
            best_result = (U_s, V_s, W_s)
            break
    
    # Strategy 2: If greedy failed, try optimization-based removal
    if best_result is None and time.time() < deadline - 40:
        support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
        support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
        support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
        
        col_scores = []
        for c in range(46):
            support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
            overlap_count = 0
            for c2 in range(46):
                if c2 == c:
                    continue
                overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
                if overlap > 3:
                    overlap_count += 1
            score = support_size + overlap_count * 0.5
            col_scores.append(score)
        
        col_order = np.argsort(col_scores)
        
        for attempt in range(min(15, len(col_order))):
            if time.time() > deadline - 30:
                break
            col_to_zero = col_order[attempt]
            mask = [i for i in range(46) if i != col_to_zero]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            found = False
            for seed in [42, 123]:
                if time.time() > deadline - 25:
                    break
                np.random.seed(seed + attempt * 100)
                U_try = U_test + np.random.randn(*U_test.shape) * 0.05
                V_try = V_test + np.random.randn(*V_test.shape) * 0.05
                W_try = W_test + np.random.randn(*W_test.shape) * 0.05
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=100, lr=0.04)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    found = True
                    break
            if found:
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    # Final: polish and try to snap
    if time.time() < deadline - 10:
        U_pol, V_pol, W_pol = newton_polish(U46, V46, W46, max_iter=30, tol=1e-14)
        valid, U_s, V_s, W_s = try_snap_validate(U_pol, V_pol, W_pol)
        if valid:
            return {
                "U": [[float(x) for x in row] for row in U_s],
                "V": [[float(x) for x in row] for row in V_s],
                "W": [[float(x) for x in row] for row in W_s]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
