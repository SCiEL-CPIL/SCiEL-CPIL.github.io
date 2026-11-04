import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 54)
    best_U, best_V, best_W, best_rank = None, None, None, n * m * p
    
    def schoolbook():
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
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find a lower rank decomposition using random search + local refinement
    # The multiplication tensor M_{ijk} = delta_{jk} (for A_{ij} * B_{jk} = C_{ik})
    # We want: sum_r U_{ir} V_{jr} W_{kr} = delta_{jk} for all i,j,k
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # For a given rank r, we need to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j==k else 0
    
    # This is a system of equations. Let's try rank values below 54.
    # Known lower bounds: rank >= n*m = 9, rank >= m*p = 18, rank >= p*n = 18
    # So minimum possible rank is 18.
    
    # Let's try to use a structure: decompose the 3x6 matrix into 6 column vectors
    # Each column is a 3x3 matrix times a 3-vector.
    # If we use the same decomposition for all 6 columns, rank would be 6 * rank(3x3x1)
    # But rank(3x3x1) = 9 (schoolbook), so 6*9 = 54.
    
    # Alternative: use a shared decomposition.
    # The tensor is T[i,j,k] = delta_{jk}
    # We can write T[i,j,k] = sum_{a=1}^3 sum_{b=1}^3 delta_{jb} delta_{ka} delta_{ia} * 1
    # That's the schoolbook with rank 9*... wait.
    
    # Actually T[i,j,k] = delta_{jk}
    # We can decompose as: T[i,j,k] = sum_{r} U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let's try a structured ansatz with rank = 3*3 + 3*3 + 3*3 = 27?
    # Or rank = 3*6 = 18?
    
    # For rank 18: U is 9x18, V is 18x18, W is 18x9
    # This is underdetermined, so there might be solutions.
    
    # Let's try a specific structure:
    # V[j*p+k, r] where r goes from 0 to 17
    # Let r = 3*j + k for j in 0..2, k in 0..5
    # Then V is identity-like: V[j*p+k, 3*j+k] = 1
    # Then we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{jk}
    # With V[j*p+k, 3*j+k] = 1 and 0 otherwise:
    # U[i*m+j, 3*j+k] * W[k*n+i, 3*j+k] = delta_{jk}
    # For j=k: U[i*m+j, 3*j+j] * W[j*n+i, 3*j+j] = 1
    # For j!=k: U[i*m+j, 3*j+k] * W[k*n+i, 3*j+k] = 0
    
    # This means for each (j,k) pair, either U[i*m+j, 3*j+k] = 0 or W[k*n+i, 3*j+k] = 0
    # For j=k: both must be nonzero and their product must be 1.
    
    # Let's set U[i*m+j, 3*j+k] = 1 if j==k else 0, and W[k*n+i, 3*j+k] = 1 if j==k else 0
    # Then for j=k: U[i*m+j, 3*j+j] * W[j*n+i, 3*j+j] = 1*1 = 1 ✓
    # For j!=k: at least one is 0, so product is 0 ✓
    
    # Wait, but U[i*m+j, 3*j+k] is indexed by (i, j) and r = 3*j+k
    # For fixed i, j: U[i*m+j, r] for r = 3*j+k where k ranges 0..5
    # So U[i*m+j, 3*j+0], U[i*m+j, 3*j+1], ..., U[i*m+j, 3*j+5]
    # We set these to 1 only when k=j, i.e., U[i*m+j, 3*j+j] = 1, others 0
    
    # Similarly W[k*n+i, 3*j+k] for fixed k, i: W[k*n+i, r] for r = 3*j+k where j ranges 0..2
    # So for fixed k, i: W[k*n+i, 3*0+k], W[k*n+i, 3*1+k], W[k*n+i, 3*2+k]
    # We set these to 1 only when j=k, i.e., W[k*n+i, 3*k+k] = 1, others 0
    
    # Let's verify: rank = 18
    # U is 9x18: U[i*m+j, 3*j+k] = 1 if k==j else 0
    # V is 18x18: V[j*p+k, 3*j+k] = 1, else 0 (identity permutation)
    # W is 18x9: W[k*n+i, 3*j+k] = 1 if j==k else 0
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # V[j*p+k, r] is 1 only when r = 3*j+k
    # So = U[i*m+j, 3*j+k] * W[k*n+i, 3*j+k]
    # U[i*m+j, 3*j+k] = 1 if k==j else 0
    # W[k*n+i, 3*j+k] = 1 if j==k else 0
    # So product = 1 if j==k else 0 = delta_{jk} ✓
    
    # This gives rank 18! Let's construct it.
    rank18 = 18
    U18 = [[0] * rank18 for _ in range(n * m)]
    V18 = [[0] * rank18 for _ in range(m * p)]
    W18 = [[0] * rank18 for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = 3 * j + k
            V18[j * p + k][r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = 3 * j + k
                if k == j:
                    U18[i * m + j][r] = 1
                    W18[k * n + i][r] = 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    expected = 1 if j == k else 0
                    if s != expected:
                        return False
        return True
    
    if verify(U18, V18, W18, n, m, p):
        best_U, best_V, best_W, best_rank = U18, V18, W18, 18
    
    return {"U": best_U, "V": best_V, "W": best_W}
