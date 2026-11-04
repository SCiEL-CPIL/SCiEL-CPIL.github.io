import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    def verify(rank, U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best = make_schoolbook()
    best_rank = n * m * p
    
    def get_error_count(rank, U, V, W):
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        count += 1
        return count
    
    # Precompute index arrays for vectorized error computation
    idx_U = np.array([(i*m + j) for i in range(n) for j in range(m) for k in range(p)])
    idx_V = np.array([(j*p + k) for i in range(n) for j in range(m) for k in range(p)])
    idx_W = np.array([(k*n + i) for i in range(n) for j in range(m) for k in range(p)])

    def get_smooth_error_np(rank, U, V, W):
        """
        U, V, W are numpy arrays.
        Returns sum of squared errors using vectorized computation.
        """
        U_sel = U[idx_U]
        V_sel = V[idx_V]
        W_sel = W[idx_W]
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        errors = sums - 1.0
        return float(np.sum(errors ** 2))

    def get_gradient_np(rank, U, V, W):
        """
        Compute gradients of the smooth error with respect to U, V, W.
        Returns (grad_U, grad_V, grad_W).
        """
        U_sel = U[idx_U]
        V_sel = V[idx_V]
        W_sel = W[idx_W]
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        errors = sums - 1.0

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)

        contrib_U = 2.0 * errors[:, np.newaxis] * V_sel * W_sel
        contrib_V = 2.0 * errors[:, np.newaxis] * U_sel * W_sel
        contrib_W = 2.0 * errors[:, np.newaxis] * U_sel * V_sel

        for t in range(n*m*p):
            u_idx = idx_U[t]
            v_idx = idx_V[t]
            w_idx = idx_W[t]
            grad_U[u_idx] += contrib_U[t]
            grad_V[v_idx] += contrib_V[t]
            grad_W[w_idx] += contrib_W[t]

        return grad_U, grad_V, grad_W

    def local_search_np(rank, U_init, V_init, W_init, seed, max_steps=300):
        """
        Local search using NumPy for speed.
        U_init, V_init, W_init are lists of lists.
        """
        rng = random.Random(seed)
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        cur_err = get_smooth_error_np(rank, U, V, W)
        if cur_err < 1e-9:
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            if verify(rank, U_s.tolist(), V_s.tolist(), W_s.tolist()):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            return None

        no_improve = 0
        step_size = 0.1

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break

            # Use gradient-based step every 5 steps
            if step % 5 == 0 and step > 0:
                grad_U, grad_V, grad_W = get_gradient_np(rank, U, V, W)
                lr = 0.01
                U_new = U - lr * grad_U
                V_new = V - lr * grad_V
                W_new = W - lr * grad_W
                new_err = get_smooth_error_np(rank, U_new, V_new, W_new)
                if new_err < cur_err:
                    U, V, W = U_new, V_new, W_new
                    cur_err = new_err
                    no_improve = 0
                else:
                    no_improve += 1
                continue

            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U[idx, c]
                best_val = old
                best_err = cur_err
                for delta in [step_size, -step_size, 2*step_size, -2*step_size, 0.5, -0.5]:
                    new_val = old + delta
                    U[idx, c] = new_val
                    new_err = get_smooth_error_np(rank, U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val = new_val
                U[idx, c] = best_val
                if best_val != old:
                    cur_err = best_err
                    no_improve = 0
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V[idx, c]
                best_val = old
                best_err = cur_err
                for delta in [step_size, -step_size, 2*step_size, -2*step_size, 0.5, -0.5]:
                    new_val = old + delta
                    V[idx, c] = new_val
                    new_err = get_smooth_error_np(rank, U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val = new_val
                V[idx, c] = best_val
                if best_val != old:
                    cur_err = best_err
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W[idx, c]
                best_val = old
                best_err = cur_err
                for delta in [step_size, -step_size, 2*step_size, -2*step_size, 0.5, -0.5]:
                    new_val = old + delta
                    W[idx, c] = new_val
                    new_err = get_smooth_error_np(rank, U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val = new_val
                W[idx, c] = best_val
                if best_val != old:
                    cur_err = best_err
                    no_improve = 0
                else:
                    no_improve += 1
            
            if no_improve > 100:
                break

        # Final polish: try rounding to half-integers and verify
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        if verify(rank, U_s.tolist(), V_s.tolist(), W_s.tolist()):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        return None
    
    def structured_init(rank, rng):
        """Initialize with random values from a small set."""
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        vals = [1, -1, 0.5, -0.5, 0]
        
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.8:
                        U[i*m+j][c] = rng.choice(vals)
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.8:
                        V[j*p+k][c] = rng.choice(vals)
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.8:
                        W[k*n+i][c] = rng.choice(vals)
        
        return U, V, W
    
    for rank in range(45, min(n*m*p, 66)):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        num_restarts = 50 if rank <= 50 else 30
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)
            
            U, V, W = structured_init(rank, rng)
            
            result = local_search_np(rank, U, V, W, seed + 7777, max_steps=300)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break
    
    return best
