import time
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    
    # Build target tensor for M_{2,3,8}
    T_full = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, T_target):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T_target, atol=1e-9, rtol=1e-9)

    def tensor_residual(U, V, W, T_target):
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T_target - computed
        return np.sum(diff * diff)

    # Schoolbook for M_{2,3,8}
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Strategy: Find low rank for M_{2,3,4} and expand
    n2, m2, p2 = 2, 3, 4
    T_234 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_234[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0

    def find_low_rank_234(max_rank=20, min_rank=14, max_iter=3000, lr=0.01):
        rng = np.random.RandomState(SEED)
        for r in range(min_rank, max_rank + 1):
            for restart in range(5):
                if time.time() > deadline:
                    return None
                scale = 1.0 / np.sqrt(r)
                U = rng.normal(size=(n2 * m2, r)) * scale
                V = rng.normal(size=(m2 * p2, r)) * scale
                W = rng.normal(size=(p2 * n2, r)) * scale
                
                prev_loss = tensor_residual(U, V, W, T_234)
                for it in range(max_iter):
                    if time.time() > deadline:
                        return None
                    if prev_loss < 1e-12:
                        break
                    computed = np.einsum('ic,jc,kc->ijk', U, V, W)
                    diff = computed - T_234
                    grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
                    grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
                    grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    new_loss = tensor_residual(U, V, W, T_234)
                    if new_loss < prev_loss * 0.999:
                        lr = min(lr * 1.05, 0.05)
                    else:
                        lr = max(lr * 0.95, 1e-6)
                    prev_loss = new_loss
                
                if prev_loss < 1e-10:
                    # Polish
                    U, V, W = polish(U, V, W, T_234, max_iter=1000)
                    if verify(U, V, W, T_234):
                        return U, V, W
        return None

    def polish(U, V, W, T_target, max_iter=1000, tol=1e-14):
        U = U.copy(); V = V.copy(); W = W.copy()
        prev_loss = tensor_residual(U, V, W, T_target)
        lr = 1e-4
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if prev_loss < tol:
                break
            computed = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = computed - T_target
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            new_loss = tensor_residual(U, V, W, T_target)
            if new_loss < prev_loss:
                lr = min(lr * 1.02, 1e-3)
            else:
                lr = max(lr * 0.9, 1e-7)
            prev_loss = new_loss
        return U, V, W

    result_234 = find_low_rank_234(max_rank=20, min_rank=14)
    
    if result_234 is not None:
        U_loc, V_loc, W_loc = result_234
        r_loc = U_loc.shape[1]
        
        # Expand to M_{2,3,8}
        num_blocks = p // p2 # 8 // 4 = 2
        r_total = r_loc * num_blocks
        
        U_full = np.zeros((n * m, r_total))
        V_full = np.zeros((m * p, r_total))
        W_full = np.zeros((p * n, r_total))
        
        for b in range(num_blocks):
            U_full[:, b*r_loc:(b+1)*r_loc] = U_loc
            
            # V mapping: Global row j * p + k. Block b covers k in [b*p2, (b+1)*p2).
            for j in range(m):
                for k_loc in range(p2):
                    global_row_v = j * p + b * p2 + k_loc
                    local_row_v = j * p2 + k_loc
                    V_full[global_row_v, b*r_loc:(b+1)*r_loc] = V_loc[local_row_v, :]
                    
                    global_row_w = (b * p2 + k_loc) * n + i # Wait, i is not defined here.
            for k_loc in range(p2):
                for i in range(n):
                    global_row_w = (b * p2 + k_loc) * n + i
                    local_row_w = k_loc * n + i
                    W_full[global_row_w, b*r_loc:(b+1)*r_loc] = W_loc[local_row_w, :]
        
        if verify(U_full, V_full, W_full, T_full):
            # Round to half-integers if close
            U_r = np.round(U_full * 2) / 2
            V_r = np.round(V_full * 2) / 2
            W_r = np.round(W_full * 2) / 2
            if verify(U_r, V_r, W_r, T_full):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    return best_result
