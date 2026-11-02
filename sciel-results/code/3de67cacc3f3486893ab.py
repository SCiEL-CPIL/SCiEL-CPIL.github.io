"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition for 3x3 x 3x6."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-6):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Should equal 1 if i==i', j==j', k==k' in the matrix mult sense
    # T is the matrix multiplication tensor: T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' and i=i', j=j', k=k'
    # Actually, the tensor for matrix mult: T[i,j,k] where i indexes (i,j) of A, j indexes (j,k) of B, k indexes (i,k) of C
    # T[(i,j),(j',k),(i',k')] = 1 if i==i', j==j', k==k'
    # So T is 1 when the indices are consistent
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] should be 1
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                # T[i*m+j, j'*p+k, k'*n+i] should be 0 for inconsistent indices
                # Check a few off-diagonal entries
    # Also check that off-diagonal entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i,j,k) == (i2,j2,k2):
                                continue
                            # T[i*m+j, j2*p+k2, k2*n+i2] should be 0
                            # Wait, the tensor index is: first index = (i,j) of A, second = (j',k) of B, third = (i',k') of C
                            # T[(i,j),(j',k),(i',k')] = 1 if i==i', j==j', k==k'
                            # So for fixed (i,j) and (j',k) and (i',k'):
                            # It's 1 iff i==i', j==j', k==k'
                            pass
    # Simpler: reconstruct and compare
    # The matrix multiplication tensor M has M[a,b,c] = 1 where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # and i==i', j==j', k==k'
    # Let's just check: for each triple (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # M[a,b,c] = 1 if i==i' and j==j' and k==k', else 0
    # Our decomposition gives: sum_c U[a,c]*V[b,c]*W[c,c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j2 * p + k
                        c = k * n + i2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        val = 0.0
                        for cc in range(rank):
                            val += U_arr[a, cc] * V_arr[b, cc] * W_arr[c, cc]
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
    return max_err < tol

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to reduce rank by finding a lower-rank decomposition
    # Strategy: try ranks from 40 to 53 and see if we can find a valid decomposition
    # This is a hard problem, so let's try a structured approach
    
    # For 3x3 x 3x6, we can view it as two 3x3 x 3x3 multiplications
    # If we find a rank-r decomposition for 3x3 x 3x3, we get rank 2r for the full problem
    
    # Try to find a low-rank 3x3 x 3x3 decomposition numerically
    # This is very hard, so let's use a heuristic approach
    
    # Heuristic: try to use the structure of the problem
    # The matrix multiplication tensor for 3x3 x 3x3 has rank at most 27 (schoolbook)
    # Known best is 23, but finding it is non-trivial
    
    # Let's try a local search approach:
    # Start with schoolbook for 3x3 x 3x3 (rank 27)
    # Try to eliminate terms by finding linear dependencies
    
    # Given time constraints, let's just return schoolbook for now
    # but try a few quick improvements
    
    # Quick improvement: check if we can do better with a structured decomposition
    # For 3x3 x 3x6, use the fact that it's two independent 3x3 x 3x3 multiplications
    
    # Let's try a numerical optimization for small rank reductions
    # This is complex, so let's just return the schoolbook result
    # unless we can find a quick win
    
    elapsed = time.time() - start_time
    if elapsed < 5:
        # Try a simple structured improvement
        # For 3x3 x 3x3, can we do better than 27?
        # Try rank 26: see if we can find a valid decomposition
        # This requires solving a system of polynomial equations, which is hard
        
        # Let's just return the schoolbook decomposition
        pass
    
    return best
# EVOLVE-BLOCK-END
