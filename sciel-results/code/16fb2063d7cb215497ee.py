"""Program for matrix multiplication tensor decomposition 2x4x7.
Uses structured search with alternating minimization and exact verification.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 115s < 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56
    
    def verify_exact(U, V, W):
        """Check if decomposition is valid using exact arithmetic."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Check if decomposition is valid using floating point."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def to_clean(arr):
        """Convert floating point values to clean integers or half-integers."""
        clean = []
        for row in arr:
            clean_row = []
            for val in row:
                # Try to snap to nearest half-integer
                snapped = round(val * 2) / 2
                if abs(val - snapped) < 1e-6:
                    clean_row.append(snapped)
                else:
                    clean_row.append(val)
            clean.append(clean_row)
        return clean
    
    def alternating_minimize(U, V, W, max_iter=200, tol=1e-10):
        """Alternating minimization to refine a rank-r decomposition.
        U: (nm, r), V: (mp, r), W: (pn, r)
        """
        nm, mp, pn = n*m, m*p, p*n
        r = U.shape[1] if hasattr(U, 'shape') else len(U[0])
        
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prev_err = _compute_error(U, V, W)
            
            # Update U: for each row i*m+j, solve for U[i*m+j, :] 
            # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # We need: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                    # Let A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # Then: A @ U[i*m+j,:] = ones(p)
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each row j*p+k, solve for V[j*p+k, :]
            # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            for j in range(m):
                for k in range(p):
                    # We need: sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                    # Let B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    # Then: B @ V[j*p+k,:] = ones(n)
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each row k*n+i, solve for W[k*n+i, :]
            # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            for k in range(p):
                for i in range(n):
                    # We need: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                    # Let C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    # Then: C @ W[k*n+i,:] = ones(m)
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            curr_err = _compute_error(U, V, W)
            if abs(curr_err - prev_err) < tol:
                break
        
        return U, V, W, curr_err
    
    def _compute_error(U, V, W):
        """Compute the maximum error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = abs(s - 1)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def structured_init(r):
        """Create a structured initial guess exploiting the tensor structure."""
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize with a pattern based on the tensor structure
        # Use small random values with some structure
        for c in range(r):
            # Assign values based on a hash-like pattern
            for i in range(n):
                for j in range(m):
                    U[i*m+j, c] = 1.0 + 0.1 * np.sin(i + j + c)
            for j in range(m):
                for k in range(p):
                    V[j*p+k, c] = 1.0 + 0.1 * np.cos(j + k + c)
            for k in range(p):
                for i in range(n):
                    W[k*n+i, c] = 1.0 + 0.1 * np.sin(k + i + c)
        
        return U, V, W
    
    def random_init(r):
        """Create a random initial guess."""
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        return U, V, W
    
    # Try ranks from low to high
    # Based on known results, rank for 2x4x7 matrix mult is likely around 20-30
    # Start with a range of ranks and multiple trials
    ranks_to_try = list(range(8, min(35, best_rank)))
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            break
        
        n_trials = 15  # Number of random restarts per rank
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            # Alternate between structured and random init
            if trial % 2 == 0:
                U, V, W = structured_init(r)
            else:
                U, V, W = random_init(r)
            
            U_refined, V_refined, W_refined, err = alternating_minimize(U, V, W, max_iter=150)
            
            if err < 1e-8:
                # Convert to clean representation
                U_clean = to_clean(U_refined.tolist())
                V_clean = to_clean(V_refined.tolist())
                W_clean = to_clean(W_refined.tolist())
                
                if verify_exact(U_clean, V_clean, W_clean):
                    best_rank = r
                    best_result = {"U": U_clean, "V": V_clean, "W": W_clean}
                    break
                elif verify_fast(U_clean, V_clean, W_clean, tol=1e-6):
                    # Close but not exact - try to polish with more iterations
                    U_polished, V_polished, W_polished, err2 = alternating_minimize(
                        U_clean, V_clean, W_clean, max_iter=100, tol=1e-12)
                    U_clean2 = to_clean(U_polished.tolist())
                    V_clean2 = to_clean(V_polished.tolist())
                    W_clean2 = to_clean(W_polished.tolist())
                    if verify_exact(U_clean2, V_clean2, W_clean2):
                        best_rank = r
                        best_result = {"U": U_clean2, "V": V_clean2, "W": W_clean2}
                        break
    
    return best_result
# EVOLVE-BLOCK-END
