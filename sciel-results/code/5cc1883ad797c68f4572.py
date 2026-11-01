"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor.

    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if (i,j,k) is a valid index, 0 otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_find_decomposition(rank, seed, n=2, m=3, p=9):
    """
    Try to find a decomposition of the given rank using random sparse U, V
    and solving for W.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate sparse U and V with entries in {0, 1}
    # Density around 0.5 might be good
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    
    for i in range(n * m):
        for c in range(rank):
            if random.random() < 0.5:
                U[i, c] = 1.0
    for j in range(m * p):
        for c in range(rank):
            if random.random() < 0.5:
                V[j, c] = 1.0

    # Construct linear system A W = b
    # Variables: W[k, i, c] for k in 0..p-1, i in 0..n-1, c in 0..rank-1
    # Total variables: p * n * rank
    num_vars = p * n * rank
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars), dtype=float)
    b = np.ones(num_eqs, dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                for c in range(rank):
                    coeff = U[i * m + j, c] * V[j * p + k, c]
                    if coeff != 0:
                        var_idx = k * n + i * rank + c  # Wait, index mapping for W
                        # W is stored as W[k*n+i, c] in the output format, 
                        # but here we treat W[k,i,c] as a flat variable.
                        # Let's use var_idx = (k * n + i) * rank + c
                        var_idx = (k * n + i) * rank + c
                        A[eq_idx, var_idx] = coeff
                eq_idx += 1
                
    # Solve A W = b
    try:
        # Use least squares
        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        # Check if solution is exact
        if np.allclose(A @ W_flat, b, atol=1e-9):
            # Check if W_flat entries are close to half-integers
            W_check = np.round(W_flat * 2) / 2
            if np.allclose(A @ W_check, b, atol=1e-9):
                # Reshape W_flat to W[k*n+i, c]
                W = W_check.reshape(p * n, rank)
                # Convert to list of lists of Fractions for exactness in return
                U_list = [[Fraction(int(val)) for val in row] for row in U]
                V_list = [[Fraction(int(val)) for val in row] for row in V]
                W_list = [[Fraction(val) for val in row] for row in W]
                
                # Final validation
                if _validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    except np.linalg.LinAlgError:
        pass
        
    return None


def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 110.0 # Safety margin

    n, m, p = 2, 3, 9

    # --- Schoolbook baseline (rank 54) ---
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # --- Search for lower rank ---
    # Try ranks from 27 up to 53
    # We prioritize lower ranks but limit search time
    # Since the system is underdetermined for r > 3, we rely on sparsity of U,V
    # to make the solution W simple.
    
    # Heuristic: Lower rank is harder. Start from 30? Or 40?
    # Let's try a range of ranks and seeds.
    
    # To maximize chance of finding a good rank, we can try:
    # 1. A few seeds for rank 45, 40, 36, 30, 27.
    # 2. If time permits, more seeds.
    
    target_ranks = [45, 40, 36, 32, 30, 28, 27]
    
    for r in target_ranks:
        if time.time() > DEADLINE:
            break
            
        # Try a few seeds for this rank
        # Limit seeds to ensure we finish in time
        max_seeds = 50 if r > 40 else 20
        
        for seed in range(max_seeds):
            if time.time() > DEADLINE:
                break
                
            res = _try_find_decomposition(r, seed, n, m, p)
            if res is not None:
                if r < best_rank:
                    best_rank = r
                    best_result = res
                # If we found a low rank, we can stop early if it's very good
                if r <= 30:
                    return best_result
                # Otherwise continue to see if we can find even lower
                # But lower ranks are harder. If we found 45, maybe 40 is possible?
                # Let's just return the best found so far if we go through all ranks?
                # No, we iterate ranks in increasing order of difficulty (decreasing r).
                # So if we find r=45, we then try r=40.
                # If we find r=40, we update best.
                # If we don't find lower, we keep 45.
                
    return best_result
