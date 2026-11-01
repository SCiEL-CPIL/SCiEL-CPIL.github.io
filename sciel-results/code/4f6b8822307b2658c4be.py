import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin before 150s kill

    size_u = n * m   # 4
    size_v = m * p   # 18
    size_w = p * n   # 18
    n_tensors = n * m * p  # 36

    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-7:
                        return False
        return True

    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 36."""
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def solve_for_VW(U_fixed, R):
        """
        Given U_fixed (size_u x R), solve for V (size_v x R) and W (size_w x R)
        such that for all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
        
        This is a bilinear system in V and W. We solve it by:
        1. For each column k, the equations involving V[:, j*p+k] and W[k*n+i]
           are bilinear. We alternate: fix W, solve for V; fix V, solve for W.
        
        Actually, let's use a different approach: vectorize and use least squares
        with alternating minimization.
        """
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Initialize V and W randomly
        np.random.seed(12345)
        V = np.random.randn(size_v, R) * 0.5
        W = np.random.randn(size_w, R) * 0.5
        
        # Alternating least squares
        for iteration in range(200):
            if time.time() > deadline - 5:
                return None, None
            
            # Fix W, solve for V
            # For each (j, k): V[j*p+k, :] should satisfy
            #   sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # This is: for fixed (j,k), we need V_vec such that
            #   M_{jk} @ V_vec = b_{jk}
            # where M_{jk}[i, r] = sum_r' U[i*m+j, r'] * W[k*n+i, r'] ... 
            # Wait, that's not right. Let me re-derive.
            
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed (j,k), this gives n equations (one per i) in R unknowns V[j*p+k, :]:
            #   sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1  for i=0..n-1
            # Let A_{jk}[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Then A_{jk} @ V[j*p+k, :] = ones(n)
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U_fixed[i*m + j, :] * W[k*n + i, :]
                    # Solve A @ v = ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p + k, :] = v_sol
                    except:
                        pass
            
            # Fix V, solve for W
            # For fixed (j,k), W[k*n+i, :] should satisfy
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # For fixed (k), we have m*n equations (one per (i,j)) in R unknowns per i:
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed k and i: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let B_k[i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then B_k @ W[k*n+i, :] = 1 for all i
            
            for k in range(p):
                B = np.zeros((n, R))
                for i in range(n):
                    for j in range(m):
                        B[i, :] += U_fixed[i*m + j, :] * V[j*p + k, :]
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    for i in range(n):
                        W[k*n + i, :] = w_sol[i, :]
                except:
                    pass
        
        # Verify
        if verify(U_fixed, V, W):
            return V, W
        return None, None

    def try_rank(R, U_pattern_type='sparse', seed=0):
        """Try to find a decomposition of rank R with a structured U."""
        np.random.seed(seed)
        
        if U_pattern_type == 'sparse':
            # U has at most 2 nonzeros per column
            U = np.zeros((size_u, R))
            for r in range(R):
                # Pick 2 random entries for this column
                if R <= 36:
                    # Try to make it structured: for each column, pick 2 rows
                    rows = np.random.choice(size_u, size=2, replace=False)
                    U[rows[0], r] = 1
                    U[rows[1], r] = 1
                else:
                    rows = np.random.choice(size_u, size=min(2, size_u), replace=False)
                    for row in rows:
                        U[row, r] = 1
        elif U_pattern_type == 'dense':
            U = np.random.randn(size_u, R) * 0.5
        elif U_pattern_type == 'structured':
            # U[i*m+j, r] is nonzero only for specific (i,j) pairs based on r
            U = np.zeros((size_u, R))
            for r in range(R):
                # Each column corresponds to a (i,j) pair or a combination
                i = r % n
                j = (r // n) % m
                U[i*m + j, r] = 1
                # Maybe add another entry
                if r < R and (r + 1) % 3 == 0:
                    i2 = (r + 1) % n
                    j2 = ((r + 1) // n) % m
                    U[i2*m + j2, r] = 1
        
        V, W = solve_for_VW(U, R)
        if V is not None:
            return U, V, W
        return None

    def try_strassen_hybrid(R_strassen, R_schoolbook_per_col):
        """
        Try a hybrid: use R_strassen terms for the first 2 columns (2x2x2 subproblem)
        and R_schoolbook_per_col terms for each of the remaining 7 columns.
        """
        R_total = R_strassen + 7 * R_schoolbook_per_col
        U = np.zeros((size_u, R_total))
        V = np.zeros((size_v, R_total))
        W = np.zeros((size_w, R_total))
        
        # For the 2x2x2 part (k=0,1), use a rank-R_strassen decomposition
        # For the 2x2x1 part (k=2..8), use rank-R_schoolbook_per_col each
        
        # This is complex to set up directly. Let's use the solver instead.
        return None

    # Strategy: Try various ranks from low to high, using structured U patterns
    # and the alternating least-squares solver.
    
    best_U, best_V, best_W = build_schoolbook()
    best_rank = 36
    
    # Try ranks from 28 to 35 (we know 36 works)
    for R in range(28, 36):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple U patterns and seeds
        for pattern in ['sparse', 'structured', 'dense']:
            if time.time() > deadline - 10:
                break
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                result = try_rank(R, pattern, seed)
                if result is not None:
                    U, V, W = result
                    if verify(U, V, W):
                        best_U, best_V, best_W = U, V, W
                        best_rank = R
                        found = True
                        break
            if found:
                break
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = build_schoolbook()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
