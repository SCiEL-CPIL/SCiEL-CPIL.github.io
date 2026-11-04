import numpy as np
import time
from itertools import product

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

    def local_optimize_fast(U, V, W, rank, max_iters=300, atol=1e-9):
        """Fast local optimization using incremental error updates with wider search."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Wider range of valid values
        valid_vals = np.array([x / 2.0 for x in range(-30, 31)])
        
        # Precompute base reconstruction
        Rec_base = compute_reconstruction(U, V, W, rank)
        best_err = np.max(np.abs(Rec_base - 1.0))
        
        # Precompute per-rank contributions for fast delta computation
        contribs = np.zeros((rank, n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            contribs[r] = np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_err < atol:
                break
            
            improved = False
            
            # Try adjusting U entries
            for r in range(rank):
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        old_val = U[idx, r]
                        
                        # Wider candidate range
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[i, j, :] + delta * v_col[j, :] * w_col[:, i]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                U[idx, r] = cand
                                contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                    U[:, r].reshape(n, m), v_col, w_col, optimize=True)
                                Rec_base = np.sum(contribs, axis=0)
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    U[idx, r] = old_val
                                    contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                        U[:, r].reshape(n, m), v_col, w_col, optimize=True)
                                    Rec_base = np.sum(contribs, axis=0)
                        
                        if best_candidate != old_val:
                            U[idx, r] = best_candidate
                            contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                U[:, r].reshape(n, m), v_col, w_col, optimize=True)
                            Rec_base = np.sum(contribs, axis=0)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting V entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        old_val = V[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[:, j, k] + delta * u_col[:, j] * w_col[k, :]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                V[idx, r] = cand
                                contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                    u_col, V[:, r].reshape(m, p), w_col, optimize=True)
                                Rec_base = np.sum(contribs, axis=0)
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    V[idx, r] = old_val
                                    contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                        u_col, V[:, r].reshape(m, p), w_col, optimize=True)
                                    Rec_base = np.sum(contribs, axis=0)
                        
                        if best_candidate != old_val:
                            V[idx, r] = best_candidate
                            contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                u_col, V[:, r].reshape(m, p), w_col, optimize=True)
                            Rec_base = np.sum(contribs, axis=0)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting W entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        old_val = W[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[i, :, k] + delta * u_col[i, :] * v_col[:, k]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                W[idx, r] = cand
                                contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                    u_col, v_col, W[:, r].reshape(p, n), optimize=True)
                                Rec_base = np.sum(contribs, axis=0)
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    W[idx, r] = old_val
                                    contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                        u_col, v_col, W[:, r].reshape(p, n), optimize=True)
                                    Rec_base = np.sum(contribs, axis=0)
                        
                        if best_candidate != old_val:
                            W[idx, r] = best_candidate
                            contribs[r] = np.einsum('ij,jk,ki->ijk', 
                                u_col, v_col, W[:, r].reshape(p, n), optimize=True)
                            Rec_base = np.sum(contribs, axis=0)
                            best_err = best_candidate_err
                            improved = True
            
            if not improved:
                break
        
        if best_err < atol:
            return U, V, W, True
        return U, V, W, False

    def structured_init(rank, seed):
        """Structured initialization with algebraic patterns - more diverse."""
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # More diverse pattern-based initialization
        for r in range(rank):
            pattern = r % 9
            
            if pattern == 0:
                # Single 1s (schoolbook-like)
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            elif pattern == 1:
                # Two 1s in U with cancellation
                i1 = r % n
                i2 = (r + 1) % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i1 * m + j, r] = 1
                U[i2 * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i1, r] = 1
                W[k * n + i2, r] = -1
            elif pattern == 2:
                # Sparse random with half-integers
                for _ in range(np.random.randint(1, 5)):
                    idx = np.random.randint(0, n * m)
                    U[idx, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                for _ in range(np.random.randint(1, 5)):
                    idx = np.random.randint(0, m * p)
                    V[idx, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                for _ in range(np.random.randint(1, 5)):
                    idx = np.random.randint(0, p * n)
                    W[idx, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            elif pattern == 3:
                # Half-integers at structured positions
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = np.random.choice([0.5, 1.0, 1.5, -0.5, -1.0])
                V[j * p + k, r] = np.random.choice([0.5, 1.0, 1.5, -0.5, -1.0])
                W[k * n + i, r] = np.random.choice([0.5, 1.0, 1.5, -0.5, -1.0])
            elif pattern == 4:
                # Random small values (denser)
                U[:, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], n * m)
                V[:, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], m * p)
                W[:, r] = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], p * n)
            elif pattern == 5:
                # Diagonal-like with perturbation
                i = r % n
                j = r % m
                k = r % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                if np.random.random() < 0.4:
                    U[(i + 1) % n * m + j, r] = np.random.choice([0.5, -0.5])
                if np.random.random() < 0.4:
                    V[j * p + (k + 1) % p, r] = np.random.choice([0.5, -0.5])
            elif pattern == 6:
                # Mixed with negatives
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = np.random.choice([-1, 1])
                V[j * p + k, r] = np.random.choice([-1, 1])
                W[k * n + i, r] = np.random.choice([-1, 1])
                if np.random.random() < 0.6:
                    U[(i + 1) % n * m + (j + 1) % m, r] = np.random.choice([0.5, -0.5, 1.0])
            elif pattern == 7:
                # Block-like: multiple nonzeros per matrix
                for _ in range(np.random.randint(2, 4)):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    U[i * m + j, r] = np.random.choice([-1, 1, 0.5])
                for _ in range(np.random.randint(2, 4)):
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    V[j * p + k, r] = np.random.choice([-1, 1, 0.5])
                for _ in range(np.random.randint(2, 4)):
                    k = np.random.randint(0, p)
                    i = np.random.randint(0, n)
                    W[k * n + i, r] = np.random.choice([-1, 1, 0.5])
            else:
                # Very sparse with large values
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = np.random.choice([1, 2, 1.5])
                V[j * p + k, r] = np.random.choice([1, 2, 1.5])
                W[k * n + i, r] = np.random.choice([1, 2, 1.5])
                # Add one more nonzero
                if np.random.random() < 0.5:
                    U[(i + 1) % n * m + j, r] = np.random.choice([-0.5, 0.5])
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=90.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        for rank in range(min(max_rank, best_rank - 1), 15, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            n_attempts = 20
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 42 + rank * 1000 + attempt * 13
                U, V, W = structured_init(rank, seed)
                
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                U_opt, V_opt, W_opt, success = local_optimize_fast(U, V, W, rank, max_iters=200)
                if success:
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
