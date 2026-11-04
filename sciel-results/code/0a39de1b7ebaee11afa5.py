import numpy as np
import time

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

    def compute_sse(U, V, W, rank):
        """Sum of squared errors - smoother objective for SA."""
        Rec = compute_reconstruction(U, V, W, rank)
        return np.sum((Rec - 1.0) ** 2)

    def simulated_annealing(U, V, W, rank, max_iters=30000, start_temp=1.0, end_temp=1e-6):
        """
        Simulated Annealing with smooth SSE objective to find half-integer solution.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute Rec
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            Rec += np.einsum('ij,jk,ki->ijk', U[:, r].reshape(n, m), V[:, r].reshape(m, p), W[:, r].reshape(p, n))
        
        current_sse = np.sum((Rec - 1.0) ** 2)
        best_sse = current_sse
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Valid values: half-integers in a reasonable range
        valid_vals = np.array([x / 2.0 for x in range(-20, 21)])
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Exponential cooling
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            # Choose a random variable to perturb
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
                delta_vec = delta * (v_col[j_idx, :] * w_col[:, i_idx])
                old_rec_slice = Rec[i_idx, j_idx, :].copy()
                Rec[i_idx, j_idx, :] += delta_vec
                U[idx, r] = new_val
                
                new_sse = np.sum((Rec - 1.0) ** 2)
                
                if new_sse < current_sse or np.random.random() < np.exp(-(new_sse - current_sse) / temp):
                    current_sse = new_sse
                    if new_sse < best_sse:
                        best_sse = new_sse
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_sse < 1e-12:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, j_idx, :] = old_rec_slice
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
                
                delta_vec = delta * (u_col[:, j_idx] * w_col[k_idx, :])
                old_rec_slice = Rec[:, j_idx, k_idx].copy()
                Rec[:, j_idx, k_idx] += delta_vec
                V[idx, r] = new_val
                
                new_sse = np.sum((Rec - 1.0) ** 2)
                
                if new_sse < current_sse or np.random.random() < np.exp(-(new_sse - current_sse) / temp):
                    current_sse = new_sse
                    if new_sse < best_sse:
                        best_sse = new_sse
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_sse < 1e-12:
                            return best_U, best_V, best_W, True
                else:
                    Rec[:, j_idx, k_idx] = old_rec_slice
                    V[idx, r] = old_val
                    
            else:  # choice == 3
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
                
                delta_vec = delta * (u_col[i_idx, :] * v_col[:, k_idx])
                old_rec_slice = Rec[i_idx, :, k_idx].copy()
                Rec[i_idx, :, k_idx] += delta_vec
                W[idx, r] = new_val
                
                new_sse = np.sum((Rec - 1.0) ** 2)
                
                if new_sse < current_sse or np.random.random() < np.exp(-(new_sse - current_sse) / temp):
                    current_sse = new_sse
                    if new_sse < best_sse:
                        best_sse = new_sse
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_sse < 1e-12:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, :, k_idx] = old_rec_slice
                    W[idx, r] = old_val
        
        return best_U, best_V, best_W, best_sse < 1e-12

    def polish_solution(U, V, W, rank, max_iters=5000):
        """
        High-precision polish: try small perturbations around current solution
        to find exact half-integer solution.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            Rec += np.einsum('ij,jk,ki->ijk', U[:, r].reshape(n, m), V[:, r].reshape(m, p), W[:, r].reshape(p, n))
        
        current_sse = np.sum((Rec - 1.0) ** 2)
        best_sse = current_sse
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Smaller step sizes for polishing
        small_steps = np.array([-0.5, 0.5])
        valid_vals = np.array([x / 2.0 for x in range(-10, 11)])
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            choice = np.random.randint(1, 4)
            
            if choice == 1:
                r = np.random.randint(rank)
                idx = np.random.randint(n * m)
                old_val = U[idx, r]
                # Try small step first, then random
                if np.random.random() < 0.5:
                    new_val = old_val + np.random.choice(small_steps)
                    if new_val not in valid_vals:
                        new_val = np.random.choice(valid_vals)
                else:
                    new_val = np.random.choice(valid_vals)
                
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                i_idx = idx // m
                j_idx = idx % m
                
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                delta_vec = delta * (v_col[j_idx, :] * w_col[:, i_idx])
                old_rec_slice = Rec[i_idx, j_idx, :].copy()
                Rec[i_idx, j_idx, :] += delta_vec
                U[idx, r] = new_val
                
                new_sse = np.sum((Rec - 1.0) ** 2)
                
                if new_sse < current_sse:
                    current_sse = new_sse
                    if new_sse < best_sse:
                        best_sse = new_sse
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_sse < 1e-12:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, j_idx, :] = old_rec_slice
                    U[idx, r] = old_val
                    
            elif choice == 2:
                r = np.random.randint(rank)
                idx = np.random.randint(m * p)
                old_val = V[idx, r]
                if np.random.random() < 0.5:
                    new_val = old_val + np.random.choice(small_steps)
                    if new_val not in valid_vals:
                        new_val = np.random.choice(valid_vals)
                else:
                    new_val = np.random.choice(valid_vals)
                
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                j_idx = idx // p
                k_idx = idx % p
                
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                
                delta_vec = delta * (u_col[:, j_idx] * w_col[k_idx, :])
                old_rec_slice = Rec[:, j_idx, k_idx].copy()
                Rec[:, j_idx, k_idx] += delta_vec
                V[idx, r] = new_val
                
                new_sse = np.sum((Rec - 1.0) ** 2)
                
                if new_sse < current_sse:
                    current_sse = new_sse
                    if new_sse < best_sse:
                        best_sse = new_sse
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_sse < 1e-12:
                            return best_U, best_V, best_W, True
                else:
                    Rec[:, j_idx, k_idx] = old_rec_slice
                    V[idx, r] = old_val
                    
            else:
                r = np.random.randint(rank)
                idx = np.random.randint(p * n)
                old_val = W[idx, r]
                if np.random.random() < 0.5:
                    new_val = old_val + np.random.choice(small_steps)
                    if new_val not in valid_vals:
                        new_val = np.random.choice(valid_vals)
                else:
                    new_val = np.random.choice(valid_vals)
                
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                k_idx = idx // n
                i_idx = idx % n
                
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                
                delta_vec = delta * (u_col[i_idx, :] * v_col[:, k_idx])
                old_rec_slice = Rec[i_idx, :, k_idx].copy()
                Rec[i_idx, :, k_idx] += delta_vec
                W[idx, r] = new_val
                
                new_sse = np.sum((Rec - 1.0) ** 2)
                
                if new_sse < current_sse:
                    current_sse = new_sse
                    if new_sse < best_sse:
                        best_sse = new_sse
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_sse < 1e-12:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, :, k_idx] = old_rec_slice
                    W[idx, r] = old_val
        
        return best_U, best_V, best_W, best_sse < 1e-12

    def structured_init(rank, seed):
        """Structured initialization with algebraic patterns."""
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Random sparse initialization with half-integers
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

    def search_for_solution(max_rank, max_search_time=110.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        for rank in range(min(max_rank, best_rank - 1), 14, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # More attempts with tighter time budget
            n_attempts = 80
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
                
                # Run SA with smooth objective
                U_opt, V_opt, W_opt, success = simulated_annealing(U, V, W, rank, max_iters=15000)
                if success:
                    if is_valid(U_opt, V_opt, W_opt, rank):
                        best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                        best_rank = rank
                        return True
                
                # Polish pass if we got close
                sse_after_sa = compute_sse(U_opt, V_opt, W_opt, rank)
                if sse_after_sa < 5.0:  # Close enough to polish
                    U_pol, V_pol, W_pol, polish_success = polish_solution(U_opt, V_opt, W_opt, rank, max_iters=3000)
                    if polish_success:
                        if is_valid(U_pol, V_pol, W_pol, rank):
                            best_solution = {"U": U_pol, "V": V_pol, "W": W_pol}
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
