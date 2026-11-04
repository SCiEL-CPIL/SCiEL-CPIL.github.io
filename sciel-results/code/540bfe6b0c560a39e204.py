import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def compute_reconstruction(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.max(np.abs(Rec - 1.0))

    def simulated_annealing(U, V, W, rank, max_iters=50000, start_temp=1.0, end_temp=1e-5):
        """
        Simulated Annealing to find a half-integer solution with error ~0.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute Rec
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            Rec += np.einsum('ij,jk,ki->ijk', U[:, r].reshape(n, m), V[:, r].reshape(m, p), W[:, r].reshape(p, n))
        
        current_err = np.max(np.abs(Rec - 1.0))
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Valid values: half-integers in a reasonable range
        # We can expand range if needed, but -10..10 should suffice for small matrices
        valid_vals = np.array([x / 2.0 for x in range(-20, 21)])
        
        # Precompute index mappings for speed? No, simple loops are fine for small sizes.
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Cooling schedule
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            # Choose a random variable to perturb
            # 1: U, 2: V, 3: W
            choice = np.random.randint(1, 4)
            
            if choice == 1:
                r = np.random.randint(rank)
                idx = np.random.randint(n * m)
                old_val = U[idx, r]
                new_val = np.random.choice(valid_vals)
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                i_idx = idx // m
                j_idx = idx % m
                
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                # Update Rec slice
                # Rec[i, j, k] += delta * V[j, k] * W[k, i]
                # Vectorized: Rec[i, j, :] += delta * (v_col[j, :] * w_col[:, i])
                delta_vec = delta * (v_col[j_idx, :] * w_col[:, i_idx])
                Rec[i_idx, j_idx, :] += delta_vec
                U[idx, r] = new_val
                
                new_err = np.max(np.abs(Rec - 1.0))
                
                if new_err < current_err or np.random.random() < np.exp(-(new_err - current_err) / temp):
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_err < 1e-9:
                            return best_U, best_V, best_W, True
                else:
                    # Revert
                    Rec[i_idx, j_idx, :] -= delta_vec
                    U[idx, r] = old_val
                    
            elif choice == 2:
                r = np.random.randint(rank)
                idx = np.random.randint(m * p)
                old_val = V[idx, r]
                new_val = np.random.choice(valid_vals)
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                j_idx = idx // p
                k_idx = idx % p
                
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                
                # Rec[i, j, k] += delta * U[i, j] * W[k, i]
                # Vectorized: Rec[:, j, k] += delta * (u_col[:, j] * w_col[k, :])
                delta_vec = delta * (u_col[:, j_idx] * w_col[k_idx, :])
                Rec[:, j_idx, k_idx] += delta_vec
                V[idx, r] = new_val
                
                new_err = np.max(np.abs(Rec - 1.0))
                
                if new_err < current_err or np.random.random() < np.exp(-(new_err - current_err) / temp):
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_err < 1e-9:
                            return best_U, best_V, best_W, True
                else:
                    Rec[:, j_idx, k_idx] -= delta_vec
                    V[idx, r] = old_val
                    
            else: # choice == 3
                r = np.random.randint(rank)
                idx = np.random.randint(p * n)
                old_val = W[idx, r]
                new_val = np.random.choice(valid_vals)
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                k_idx = idx // n
                i_idx = idx % n
                
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                
                # Rec[i, j, k] += delta * U[i, j] * V[j, k]
                # Vectorized: Rec[i, :, k] += delta * (u_col[i, :] * v_col[:, k])
                delta_vec = delta * (u_col[i_idx, :] * v_col[:, k_idx])
                Rec[i_idx, :, k_idx] += delta_vec
                W[idx, r] = new_val
                
                new_err = np.max(np.abs(Rec - 1.0))
                
                if new_err < current_err or np.random.random() < np.exp(-(new_err - current_err) / temp):
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_err < 1e-9:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, :, k_idx] -= delta_vec
                    W[idx, r] = old_val
        
        return best_U, best_V, best_W, best_err < 1e-9

    def structured_init(rank, seed):
        """Structured initialization with algebraic patterns."""
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Random sparse initialization with half-integers
            # Density ~ 3-5 non-zeros per matrix
            n_nonzero_u = np.random.randint(2, 6)
            n_nonzero_v = np.random.randint(2, 6)
            n_nonzero_w = np.random.randint(2, 6)
            
            idx_u = np.random.choice(n * m, n_nonzero_u, replace=False)
            idx_v = np.random.choice(m * p, n_nonzero_v, replace=False)
            idx_w = np.random.choice(p * n, n_nonzero_w, replace=False)
            
            U[idx_u, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], n_nonzero_u)
            V[idx_v, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], n_nonzero_v)
            W[idx_w, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], n_nonzero_w)
            
        return U, V, W

    def search_for_solution(max_rank, max_search_time=100.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Try ranks from best_rank - 1 down to a lower bound
        # Lower bound for 3x3x5 is not 15, but let's try down to 20 to be safe, or just 15.
        for rank in range(min(max_rank, best_rank - 1), 14, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Multiple restarts for this rank
            n_attempts = 50
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 42 + rank * 1000 + attempt * 13
                U, V, W = structured_init(rank, seed)
                
                # Quick check
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Run SA
                U_opt, V_opt, W_opt, success = simulated_annealing(U, V, W, rank, max_iters=10000)
                if success:
                    # Verify strictly
                    if is_valid(U_opt, V_opt, W_opt, rank):
                        best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                        best_rank = rank
                        return True
        
        return False

    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
