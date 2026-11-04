import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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
        """Compute full reconstruction tensor efficiently."""
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
        """Compute the error in the tensor reconstruction."""
        Rec = compute_reconstruction(U, V, W, rank)
        return np.max(np.abs(Rec - 1.0))

    def compute_error_fast(U, V, W, rank, prev_rec=None):
        """Fast error computation using previous reconstruction as starting point."""
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.max(np.abs(Rec - 1.0))

    def local_optimize(U, V, W, rank, max_iters=200, atol=1e-9):
        """Try to fix small errors by adjusting entries to nearest valid values.
        Uses batch evaluation and gradient-like hints for efficiency."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Valid values: integers and half-integers
        valid_vals = np.array([x / 2.0 for x in range(-12, 13)])
        
        best_err = compute_error(U, V, W, rank)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_err < atol:
                break
            
            # Compute current reconstruction for gradient hints
            Rec = compute_reconstruction(U, V, W, rank)
            error_tensor = Rec - 1.0
            
            improved = False
            any_change = False
            
            # Try adjusting U entries - use error tensor to prioritize
            for r in range(rank):
                u_col = U[:, r]
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                # Gradient hint: which entries in U would most reduce error
                # d(Rec)/d(U_ij) = v_col[j,:] outer w_col[:,i]
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        old_val = u_col[idx]
                        
                        # Find candidate values near old_val
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 0.75]
                        if len(candidates) == 0:
                            candidates = valid_vals[np.argmin(np.abs(valid_vals - old_val))]
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            U[idx, r] = cand
                            err = compute_error_fast(U, V, W, rank)
                            if err < best_candidate_err - 1e-10:
                                best_candidate_err = err
                                best_candidate = cand
                        U[idx, r] = best_candidate
                        
                        if best_candidate != old_val:
                            any_change = True
                            if best_candidate_err < best_err - 1e-10:
                                best_err = best_candidate_err
                                improved = True
            
            # Try adjusting V entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r]
                w_col = W[:, r].reshape(p, n)
                
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        old_val = v_col[idx]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 0.75]
                        if len(candidates) == 0:
                            candidates = valid_vals[np.argmin(np.abs(valid_vals - old_val))]
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            V[idx, r] = cand
                            err = compute_error_fast(U, V, W, rank)
                            if err < best_candidate_err - 1e-10:
                                best_candidate_err = err
                                best_candidate = cand
                        V[idx, r] = best_candidate
                        
                        if best_candidate != old_val:
                            any_change = True
                            if best_candidate_err < best_err - 1e-10:
                                best_err = best_candidate_err
                                improved = True
            
            # Try adjusting W entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r]
                
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        old_val = w_col[idx]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 0.75]
                        if len(candidates) == 0:
                            candidates = valid_vals[np.argmin(np.abs(valid_vals - old_val))]
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            W[idx, r] = cand
                            err = compute_error_fast(U, V, W, rank)
                            if err < best_candidate_err - 1e-10:
                                best_candidate_err = err
                                best_candidate = cand
                        W[idx, r] = best_candidate
                        
                        if best_candidate != old_val:
                            any_change = True
                            if best_candidate_err < best_err - 1e-10:
                                best_err = best_candidate_err
                                improved = True
            
            if not improved and not any_change:
                break
        
        if best_err < atol:
            return U, V, W, True
        return U, V, W, False

    def structured_init(rank, seed):
        """Create structured initialization based on partial patterns."""
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Use a mix of structured and random initialization
        for r in range(rank):
            pattern = r % 5
            if pattern == 0:
                # Pattern 1: Single 1 in each matrix
                i = (r // 5) % n
                j = (r // 5) % m
                k = (r // 5) % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            elif pattern == 1:
                # Pattern 2: Two 1s in U, one in V, one in W
                i1 = (r // 5) % n
                i2 = (r // 5 + 1) % n
                j = (r // 5) % m
                k1 = (r // 5) % p
                k2 = (r // 5 + 1) % p
                U[i1 * m + j, r] = 1
                U[i2 * m + j, r] = 1
                V[j * p + k1, r] = 1
                W[k1 * n + i1, r] = 1
                W[k2 * n + i2, r] = -1
            elif pattern == 2:
                # Pattern 3: Sparse random with small values
                n_nonzero = np.random.randint(1, 4)
                for _ in range(n_nonzero):
                    idx = np.random.randint(0, n * m)
                    U[idx, r] = np.random.choice([-1, 1])
                n_nonzero = np.random.randint(1, 4)
                for _ in range(n_nonzero):
                    idx = np.random.randint(0, m * p)
                    V[idx, r] = np.random.choice([-1, 1])
                n_nonzero = np.random.randint(1, 4)
                for _ in range(n_nonzero):
                    idx = np.random.randint(0, p * n)
                    W[idx, r] = np.random.choice([-1, 1])
            elif pattern == 3:
                # Pattern 4: Half-integer values
                i = (r // 5) % n
                j = (r // 5) % m
                k = (r // 5) % p
                U[i * m + j, r] = np.random.choice([0.5, 1.0, 1.5])
                V[j * p + k, r] = np.random.choice([0.5, 1.0, 1.5])
                W[k * n + i, r] = np.random.choice([0.5, 1.0, 1.5])
            else:
                # Pattern 5: Random small integers/half-integers
                U[:, r] = np.random.choice([-1, -0.5, 0.5, 1], n * m)
                V[:, r] = np.random.choice([-1, -0.5, 0.5, 1], m * p)
                W[:, r] = np.random.choice([-1, -0.5, 0.5, 1], p * n)
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=80.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        for rank in range(min(max_rank, best_rank - 1), 15, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Try multiple structured initializations with more attempts
            n_attempts = 8
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 42 + rank * 1000 + attempt * 7
                U, V, W = structured_init(rank, seed)
                
                # Check if already valid
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Try local optimization
                U_opt, V_opt, W_opt, success = local_optimize(U, V, W, rank, max_iters=150)
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    return True
        
        return False

    # Try to find a better solution
    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
