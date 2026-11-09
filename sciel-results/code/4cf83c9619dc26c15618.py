"""Program searching for low-rank exact tensor decomposition of 2x3 by 3x8 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    deadline = time.time() + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    base_rank = n * m * p

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-9, rtol=1e-9)

    def tensor_residual(U, V, W, T_target):
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T_target - computed
        return np.sum(diff * diff)

    # ---------- Schoolbook baseline ----------
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
    best = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist(), "rank": base_rank}

    # ---------- Strategy: Find low-rank decomposition for M_{2,3,2} and expand ----------
    # M_{2,3,2} tensor T2[i*3+j, j*2+k, k*2+i] = 1
    n2, m2, p2 = 2, 3, 2
    T2 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0

    def verify2(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T2, atol=1e-9, rtol=1e-9)

    def tensor_residual2(U, V, W):
        return tensor_residual(U, V, W, T2)

    def polish_small(U, V, W, T_target, max_iter=5000, tol=1e-16):
        """High-precision polish using adaptive step size and momentum."""
        U = U.copy().astype(float); V = V.copy().astype(float); W = W.copy().astype(float)
        prev_loss = tensor_residual(U, V, W, T_target)
        if prev_loss < tol:
            return U, V, W
        lr = 1e-3
        # Momentum
        vel_U = np.zeros_like(U); vel_V = np.zeros_like(V); vel_W = np.zeros_like(W)
        beta = 0.9
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
            
            vel_U = beta * vel_U + (1 - beta) * grad_U
            vel_V = beta * vel_V + (1 - beta) * grad_V
            vel_W = beta * vel_W + (1 - beta) * grad_W
            
            U -= lr * vel_U
            V -= lr * vel_V
            W -= lr * vel_W
            
            new_loss = tensor_residual(U, V, W, T_target)
            if new_loss < prev_loss:
                lr = min(lr * 1.01, 1e-2)
            else:
                lr = max(lr * 0.85, 1e-8)
                U += lr * vel_U
                V += lr * vel_V
                W += lr * vel_W
            prev_loss = new_loss
        return U, V, W

    def find_low_rank_m232(max_rank=12, min_rank=8, max_restarts=50, max_iter=3000):
        """Search for a low-rank decomposition of M_{2,3,2} with multiple strategies."""
        rng = np.random.RandomState(SEED)
        best_result = None
        best_loss = np.inf
        
        for r in range(min_rank, max_rank + 1):
            for restart in range(max_restarts):
                if time.time() > deadline:
                    if best_result is not None:
                        return best_result
                    return None
                if time.time() > deadline - 20:
                    break
                    
                # Multiple initialization strategies
                strategy = restart % 4
                if strategy == 0:
                    scale = 1.0 / np.sqrt(r)
                    U = rng.normal(size=(n2 * m2, r)) * scale
                    V = rng.normal(size=(m2 * p2, r)) * scale
                    W = rng.normal(size=(p2 * n2, r)) * scale
                elif strategy == 1:
                    # Sparse initialization
                    U = np.zeros((n2 * m2, r)); V = np.zeros((m2 * p2, r)); W = np.zeros((p2 * n2, r))
                    for c in range(r):
                        U[rng.randint(0, n2*m2), c] = rng.choice([-1, 1])
                        V[rng.randint(0, m2*p2), c] = rng.choice([-1, 1])
                        W[rng.randint(0, p2*n2), c] = rng.choice([-1, 1])
                elif strategy == 2:
                    # Structured initialization based on schoolbook pattern
                    U = np.zeros((n2 * m2, r)); V = np.zeros((m2 * p2, r)); W = np.zeros((p2 * n2, r))
                    for c in range(min(r, n2*m2*p2)):
                        i, j, k = c // (m2*p2), (c // p2) % m2, c % p2
                        U[i * m2 + j, c] = 1
                        V[j * p2 + k, c] = 1
                        W[k * n2 + i, c] = 1
                    if r > n2*m2*p2:
                        U[:, n2*m2*p2:] = rng.normal(size=(n2*m2, r - n2*m2*p2)) * 0.1
                        V[:, n2*m2*p2:] = rng.normal(size=(m2*p2, r - n2*m2*p2)) * 0.1
                        W[:, n2*m2*p2:] = rng.normal(size=(p2*n2, r - n2*m2*p2)) * 0.1
                else:
                    # Mixed
                    U = rng.normal(size=(n2 * m2, r)) * 0.5
                    V = rng.normal(size=(m2 * p2, r)) * 0.5
                    W = rng.normal(size=(p2 * n2, r)) * 0.5
                    # Add some structure
                    for c in range(min(4, r)):
                        U[rng.randint(0, n2*m2), c] = 1.0
                        V[rng.randint(0, m2*p2), c] = 1.0
                        W[rng.randint(0, p2*n2), c] = 1.0
                
                # Gradient Descent with adaptive learning rate
                prev_loss = tensor_residual2(U, V, W)
                lr = 0.01
                for it in range(max_iter):
                    if time.time() > deadline:
                        break
                    if prev_loss < 1e-14:
                        break
                    computed = np.einsum('ic,jc,kc->ijk', U, V, W)
                    diff = computed - T2
                    grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
                    grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
                    grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    new_loss = tensor_residual2(U, V, W)
                    if new_loss < prev_loss * 0.99:
                        lr = min(lr * 1.03, 0.05)
                    elif new_loss > prev_loss * 1.01:
                        lr = max(lr * 0.9, 1e-7)
                        U += lr * grad_U
                        V += lr * grad_V
                        W += lr * grad_W
                    prev_loss = new_loss
                
                if prev_loss < best_loss:
                    best_loss = prev_loss
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if prev_loss < 1e-10:
                    # Polish
                    U_p, V_p, W_p = polish_small(U, V, W, T2, max_iter=5000)
                    if verify2(U_p, V_p, W_p):
                        return U_p, V_p, W_p
        
        # If we found something close but not exact, try harder polishing
        if best_result is not None and best_loss < 1e-6:
            U_p, V_p, W_p = polish_small(best_result[0], best_result[1], best_result[2], T2, max_iter=10000)
            if verify2(U_p, V_p, W_p):
                return U_p, V_p, W_p
        
        return None

    # Try to find a decomposition for M_{2,3,2} with rank <= 12
    result232 = find_low_rank_m232(max_rank=12, min_rank=8, max_restarts=50)
    
    if result232 is not None:
        U_loc, V_loc, W_loc = result232
        r_loc = U_loc.shape[1]
        
        # Expand to M_{2,3,8}
        # p = 8, p2 = 2. Number of blocks = 4.
        num_blocks = p // p2
        r_total = r_loc * num_blocks
        
        U_full = np.zeros((n * m, r_total))
        V_full = np.zeros((m * p, r_total))
        W_full = np.zeros((p * n, r_total))
        
        # U is repeated for each block
        for b in range(num_blocks):
            U_full[:, b*r_loc:(b+1)*r_loc] = U_loc
        
        # V and W are block-diagonal
        # Global V row index: j * p + k, where k is in [0, p)
        # Block b covers k in [b*p2, (b+1)*p2)
        # Local V row index: j * p2 + k_loc, where k_loc = k - b*p2
        # Mapping: global_row_v = j * p + (b * p2 + k_loc), local_row_v = j * p2 + k_loc
        for b in range(num_blocks):
            for j in range(m):
                for k_loc in range(p2):
                    global_row_v = j * p + b * p2 + k_loc
                    local_row_v = j * p2 + k_loc
                    V_full[global_row_v, b*r_loc:(b+1)*r_loc] = V_loc[local_row_v, :]
            
            # W mapping: global_row_w = (b * p2 + k_loc) * n + i, local_row_w = k_loc * n + i
            for k_loc in range(p2):
                for i in range(n):
                    global_row_w = (b * p2 + k_loc) * n + i
                    local_row_w = k_loc * n + i
                    W_full[global_row_w, b*r_loc:(b+1)*r_loc] = W_loc[local_row_w, :]
        
        # Verify the full decomposition
        if verify(U_full, V_full, W_full):
            # Round to half-integers if close
            U_r = np.round(U_full * 2) / 2
            V_r = np.round(V_full * 2) / 2
            W_r = np.round(W_full * 2) / 2
            if verify(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Try direct search on the full M_{2,3,8} tensor if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Try to find a decomposition of the full tensor with rank < 48
        rng2 = np.random.RandomState(SEED + 1000)
        for r_full in range(40, 48):
            if time.time() > deadline - 10:
                break
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                scale = 1.0 / np.sqrt(r_full)
                U_f = rng2.normal(size=(n * m, r_full)) * scale
                V_f = rng2.normal(size=(m * p, r_full)) * scale
                W_f = rng2.normal(size=(p * n, r_full)) * scale
                
                prev_loss = tensor_residual(U_f, V_f, W_f, T)
                lr = 0.005
                for it in range(2000):
                    if time.time() > deadline - 3:
                        break
                    if prev_loss < 1e-14:
                        break
                    computed = np.einsum('ic,jc,kc->ijk', U_f, V_f, W_f)
                    diff = computed - T
                    grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V_f, W_f)
                    grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U_f, W_f)
                    grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U_f, V_f)
                    
                    U_f -= lr * grad_U
                    V_f -= lr * grad_V
                    W_f -= lr * grad_W
                    
                    new_loss = tensor_residual(U_f, V_f, W_f, T)
                    if new_loss < prev_loss * 0.99:
                        lr = min(lr * 1.02, 0.02)
                    elif new_loss > prev_loss * 1.01:
                        lr = max(lr * 0.9, 1e-8)
                        U_f += lr * grad_U
                        V_f += lr * grad_V
                        W_f += lr * grad_W
                    prev_loss = new_loss
                
                if prev_loss < 1e-10:
                    U_p, V_p, W_p = polish_small(U_f, V_f, W_f, T, max_iter=3000)
                    if verify(U_p, V_p, W_p):
                        U_r = np.round(U_p * 2) / 2
                        V_r = np.round(V_p * 2) / 2
                        W_r = np.round(W_p * 2) / 2
                        if verify(U_r, V_r, W_r):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
