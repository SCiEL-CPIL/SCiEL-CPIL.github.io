import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_np(U, V, W, tol=1e-9):
        """Vectorized verification."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def solve_for_U(V, W, rank):
        """Given V and W, solve for U via least squares."""
        U = np.zeros((n * m, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = np.zeros((p, rank), dtype=np.float64)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p, dtype=np.float64)
                try:
                    sol, res, r, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = sol
                except:
                    return None
        return U
    
    def solve_for_V(U, W, rank):
        """Given U and W, solve for V via least squares."""
        V = np.zeros((m * p, rank), dtype=np.float64)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank), dtype=np.float64)
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n, dtype=np.float64)
                try:
                    sol, res, r, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = sol
                except:
                    return None
        return V
    
    def solve_for_W(U, V, rank):
        """Given U and V, solve for W via least squares."""
        W = np.zeros((p * n, rank), dtype=np.float64)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank), dtype=np.float64)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m, dtype=np.float64)
                try:
                    sol, res, r, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = sol
                except:
                    return None
        return W
    
    def als_search(rank, n_iter=50, n_restarts=5, seed=42):
        """Alternating least squares search for rank-r decomposition."""
        rng = np.random.default_rng(seed)
        best_err = float('inf')
        best_U = best_V = best_W = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            U = solve_for_U(V, W, rank)
            if U is None:
                continue
            
            for it in range(n_iter):
                if time.time() > deadline - 10:
                    break
                
                # Alternate solving
                V_new = solve_for_V(U, W, rank)
                if V_new is None:
                    break
                U_new = solve_for_U(V_new, W, rank)
                if U_new is None:
                    break
                W_new = solve_for_W(U_new, V_new, rank)
                if W_new is None:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (s - 1.0)**2
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
        
        if best_U is not None and verify_np(best_U, best_V, best_W, tol=1e-8):
            return best_U, best_V, best_W
        return None
    
    def try_rank(rank, n_restarts=3):
        """Try to find a valid decomposition of given rank."""
        if time.time() > deadline - 15:
            return None
        result = als_search(rank, n_iter=80, n_restarts=n_restarts, seed=rank*1000+42)
        if result is not None:
            U, V, W = result
            return U.tolist(), V.tolist(), W.tolist()
        return None
    
    # Try ranks from low to high
    # Lower bound is at least max(n*m, m*p, p*n) = 24
    # But realistically, tensor rank is higher
    # Let's try a range of ranks
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try structured ranks first (divisors, common values)
    candidate_ranks = []
    for r in range(24, rank_schoolbook + 1):
        candidate_ranks.append(r)
    
    # Prioritize: try lower ranks with more effort, higher ranks with less
    # Group by difficulty
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        if target_rank > best_rank:
            break
        
        # More restarts for lower ranks
        n_restarts = 5 if target_rank < 40 else 3 if target_rank < 60 else 2
        
        result = try_rank(target_rank, n_restarts=n_restarts)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
