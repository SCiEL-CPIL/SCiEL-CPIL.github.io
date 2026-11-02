import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def verify_strict(U, V, W, tol=1e-4):
        """Stricter verification for final validation."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        err = abs(s - expected)
                        if err > max_err:
                            max_err = err
        return max_err < tol, max_err

    # Schoolbook rank 27
    rank_sb = 27
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Rank-26 Strassen-based decomposition
    # This is a known decomposition: use Strassen's 7-multiplication for the 2x2 core
    # and handle the border terms with standard multiplications
    U26 = np.zeros((nm, 26))
    V26 = np.zeros((mp, 26))
    W26 = np.zeros((pn, 26))
    
    # Indices: row i (0,1,2), col j (0,1,2)
    # We'll use a specific rank-26 construction
    # Based on Strassen's algorithm adapted for 3x3
    
    # Let's define the 26 terms explicitly
    # This is a known rank-26 decomposition for 3x3 matrix multiplication
    
    # Term definitions (using 0-indexed):
    # The decomposition uses 26 scalar multiplications
    
    # Define U, V, W entries
    # Row/col indices in U: i*m+j where i=row of A, j=col of A
    # Row/col indices in V: l*p+k where l=row of B, k=col of B  
    # Row/col indices in W: k*n+i where k=col of result, i=row of result
    
    terms = []
    
    # We need to find a valid rank-26 decomposition
    # Let me use a systematic approach based on Strassen
    
    # For the 2x2 sub-blocks, use Strassen's 7 multiplications
    # For the border, use standard multiplications
    
    # Let's construct it carefully
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A * B
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We'll define 26 products of the form (linear combo of A entries) * (linear combo of B entries)
    
    # This is complex, so let me use a different approach:
    # Start with rank 27 and try to merge terms
    
    # Actually, let me just use the optimization approach but with better initialization
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = rank_sb
    
    def objective(vars, target_rank):
        r = target_rank
        U_f = vars[:nm*r].reshape(nm, r)
        V_f = vars[nm*r:2*nm*r].reshape(mp, r)
        W_f = vars[2*nm*r:3*nm*r].reshape(pn, r)
        
        errors = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W_f[k*n+i, :] * U_f[i*m+j, :] * V_f[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        errors.append(s - expected)
        return np.array(errors)

    def polish_solution(U_init, V_init, W_init, target_rank, max_iter=5000):
        """Polish a solution using least_squares with tight tolerance."""
        x0 = np.concatenate([np.array(U_init).flatten(), 
                            np.array(V_init).flatten(), 
                            np.array(W_init).flatten()])
        try:
            res = least_squares(objective, x0, args=(target_rank,), method='lm', 
                               max_nfev=max_iter, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if res.cost < 1e-10:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    def try_reduce_rank(target_rank, num_starts=15):
        """Try to find a decomposition of the given rank."""
        for start_idx in range(num_starts):
            if time.time() > deadline - 5:
                return None
            
            if start_idx == 0:
                # Random initialization with small values
                np.random.seed(42 + start_idx)
                U_init = np.random.randn(nm, target_rank) * 0.3
                V_init = np.random.randn(mp, target_rank) * 0.3
                W_init = np.random.randn(pn, target_rank) * 0.3
            elif start_idx == 1:
                # Another random seed
                np.random.seed(100 + start_idx)
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            else:
                # Reduction from schoolbook: pick random subset of columns
                np.random.seed(200 + start_idx)
                keep_indices = np.random.choice(rank_sb, target_rank, replace=False)
                sort_keep = np.sort(keep_indices)
                
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                for new_col, old_col in enumerate(sort_keep):
                    U_init[:, new_col] = U[:, old_col]
                    V_init[:, new_col] = V[:, old_col]
                    W_init[:, new_col] = W[:, old_col]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(objective, x0, args=(target_rank,), method='lm', max_nfev=2000)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                sol = res.x
                U_sol = sol[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
                W_sol = sol[2*nm*target_rank:3*nm*target_rank].reshape(pn, target_rank)
                
                # Try rounding to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-3):
                    # Polish the solution
                    polished = polish_solution(U_r, V_r, W_r, target_rank)
                    if polished:
                        U_p, V_p, W_p = polished
                        # Round again after polishing
                        U_final = np.round(U_p * 2) / 2
                        V_final = np.round(V_p * 2) / 2
                        W_final = np.round(W_p * 2) / 2
                        if verify(U_final, V_final, W_final, tol=1e-3):
                            return U_final.tolist(), V_final.tolist(), W_final.tolist()
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                # Try rounding to integers
                U_i = np.round(U_sol)
                V_i = np.round(V_sol)
                W_i = np.round(W_sol)
                
                if verify(U_i, V_i, W_i, tol=1e-3):
                    polished = polish_solution(U_i, V_i, W_i, target_rank)
                    if polished:
                        U_p, V_p, W_p = polished
                        U_final = np.round(U_p * 2) / 2
                        V_final = np.round(V_p * 2) / 2
                        W_final = np.round(W_p * 2) / 2
                        if verify(U_final, V_final, W_final, tol=1e-3):
                            return U_final.tolist(), V_final.tolist(), W_final.tolist()
                    return U_i.tolist(), V_i.tolist(), W_i.tolist()
                
                # Try quarter-integer rounding
                U_q = np.round(U_sol * 4) / 4
                V_q = np.round(V_sol * 4) / 4
                W_q = np.round(W_sol * 4) / 4
                
                if verify(U_q, V_q, W_q, tol=1e-3):
                    return U_q.tolist(), V_q.tolist(), W_q.tolist()
                
        return None

    # Try rank 23 first (best case)
    if time.time() - start_time < deadline - 15:
        result = try_reduce_rank(23, num_starts=15)
        if result:
            best_U, best_V, best_W = result
            best_rank = 23
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 24
    if time.time() - start_time < deadline - 15:
        result = try_reduce_rank(24, num_starts=15)
        if result:
            best_U, best_V, best_W = result
            best_rank = 24
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 25
    if time.time() - start_time < deadline - 15:
        result = try_reduce_rank(25, num_starts=15)
        if result:
            best_U, best_V, best_W = result
            best_rank = 25
            return {"U": best_U, "V": best_V, "W": best_W}

    # Try rank 26
    if time.time() - start_time < deadline - 10:
        result = try_reduce_rank(26, num_starts=15)
        if result:
            best_U, best_V, best_W = result
            best_rank = 26
            return {"U": best_U, "V": best_V, "W": best_W}

    # Fallback to schoolbook rank 27
    return {"U": best_U, "V": best_V, "W": best_W}
