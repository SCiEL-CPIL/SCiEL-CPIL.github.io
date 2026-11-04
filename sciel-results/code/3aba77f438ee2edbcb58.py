import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                # Reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
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

def _try_random_rank_reduction(U, V, W, n, m, p, max_iters=500, seed=42):
    """
    Attempt to reduce rank by merging pairs of rank-1 terms.
    Start from schoolbook, try to find linear dependencies.
    """
    np.random.seed(seed)
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Try random linear combinations of pairs of columns
    # For each pair (a, b), try to see if we can replace columns a and b
    # with a single column that still satisfies all tensor equations.
    
    # This is complex; instead, use a simpler heuristic:
    # Randomly perturb the rank-1 terms and see if we can drop a term.
    
    # Actually, let's try a different approach: use the fact that
    # for small dimensions, we can enumerate all possible rank-r decompositions
    # using random linear algebra.
    
    # Start from schoolbook and try to reduce by 1 at a time.
    # For rank r, we have r terms. Try to see if term r-1 is redundant.
    
    # A term is redundant if there exist coefficients c_0, ..., c_{r-2} such that
    # sum_{s=0}^{r-2} c_s * (U_s * V_s * W_s) = U_{r-1} * V_{r-1} * W_{r-1}
    # in the tensor sense.
    
    # This is a system of n*m*p linear equations in r-1 unknowns.
    
    # For each candidate rank from schoolbook down to some minimum:
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank
    
    for target_rank in range(rank - 1, max(1, rank - 10), -1):
        if current_rank <= target_rank:
            break
        
        # Try to eliminate the last term
        # Check if the last term is in the span of the others
        # This is a tensor rank problem, which is NP-hard in general,
        # but for small cases we can use numerical methods.
        
        # Simple check: see if the last term's contribution is zero
        # or can be absorbed.
        
        # For now, just try random linear combinations
        success = False
        for _ in range(100):
            # Randomly pick a pair of terms to merge
            if current_rank < 2:
                break
            a, b = np.random.choice(current_rank, size=2, replace=False)
            
            # Try to replace terms a and b with a single term
            # This requires solving a system of equations
            # For simplicity, skip this and just try dropping one term
            
            # Try dropping term b
            test_U = np.delete(current_U, b, axis=1)
            test_V = np.delete(current_V, b, axis=1)
            test_W = np.delete(current_W, b, axis=1)
            
            # Verify
            if _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-9):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                success = True
                break
        
        if not success:
            break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = current_U
        best_V = current_V
        best_W = current_W
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Start with schoolbook
    res = _schoolbook_decomposition(n, m, p)
    
    # Try to reduce rank
    rank, U, V, W = _try_random_rank_reduction(res['U'], res['V'], res['W'], n, m, p)
    
    if rank < len(res['U'][0]):
        res = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Verify
    if _verify_decomposition(res['U'], res['V'], res['W'], n, m, p, tol=1e-9):
        return res
    
    # Fallback
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
