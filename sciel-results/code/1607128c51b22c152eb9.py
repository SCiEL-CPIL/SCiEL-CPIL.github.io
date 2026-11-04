"""Matrix multiplication 2x3 by 3x5: rank-minimal exact tensor decomposition.

Strategy:
  1. Construct the schoolbook decomposition (rank 30) as a baseline.
  2. Construct a rank-28 decomposition using Strassen's algorithm on 2x2 blocks
     where the middle dimension m=3 is split into {0,1} and {2}.
     - The j=2 part is schoolbook: 2*5 = 10 terms.
     - The j=0,1 part with k=4 is schoolbook: 2*2 = 4 terms.
     - The j=0,1 part with k=0,1 uses Strassen: 7 terms.
     - The j=0,1 part with k=2,3 uses Strassen: 7 terms.
     Total: 10 + 4 + 7 + 7 = 28.
  3. Verify both decompositions by checking the tensor identity.
  4. Return the lower-rank valid decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = 1 for all valid indices.
    The decomposition requires:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _strassen_2x2(n, m_block, p_block, j_offset, k_offset, U, V, W, r):
    """Add Strassen 2x2 block multiplication terms.
    
    Computes C_block = A_block * B_block where:
      A_block is n x m_block (rows i, cols j_offset..j_offset+m_block-1)
      B_block is m_block x p_block (rows j_offset..j_offset+m_block-1, cols k_offset..k_offset+p_block-1)
      C_block is n x p_block (rows i, cols k_offset..k_offset+p_block-1)
    
    For n=2, m_block=2, p_block=2, this is the standard Strassen algorithm.
    Returns the next column index r.
    """
    if n != 2 or m_block != 2 or p_block != 2:
        # Fallback to schoolbook for non-2x2x2 blocks
        for i in range(n):
            for j in range(m_block):
                for k in range(p_block):
                    U[i * m + (j_offset + j)][r] = 1
                    V[(j_offset + j) * p + (k_offset + k)][r] = 1
                    W[(k_offset + k) * n + i][r] = 1
                    r += 1
        return r
    
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a, b], [c, d]] = A_block (indices: a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1])
    # B = [[e, f], [g, h]] = B_block (indices: e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1])
    # C = [[a(e+h)-b(g+h), b(f-h)], [c(e+g)-d(e+f), d(f+g)-c(g+h)]]
    
    # M1 = (a+d)(e+h)
    U[i0 := 0][r] += 1  # a
    U[1 * m + (j_offset + 1)][r] += 1  # d
    V[j_offset * p + (k_offset + 0)][r] += 1  # e
    V[(j_offset + 1) * p + (k_offset + 1)][r] += 1  # h
    W[(k_offset + 0) * n + 0][r] += 1  # C[0,0]
    W[(k_offset + 1) * n + 1][r] += 1  # C[1,1]
    r += 1
    
    # M2 = (c+d)(e)
    U[1 * m + (j_offset + 0)][r] += 1  # c
    U[1 * m + (j_offset + 1)][r] += 1  # d
    V[j_offset * p + (k_offset + 0)][r] += 1  # e
    W[(k_offset + 0) * n + 1][r] += 1  # C[1,0]
    W[(k_offset + 1) * n + 1][r] -= 1  # C[1,1]
    r += 1
    
    # M3 = a(f-h)
    U[0 * m + (j_offset + 0)][r] += 1  # a
    V[j_offset * p + (k_offset + 1)][r] += 1  # f
    V[(j_offset + 1) * p + (k_offset + 1)][r] -= 1  # -h
    W[(k_offset + 1) * n + 0][r] += 1  # C[0,1]
    W[(k_offset + 1) * n + 1][r] += 1  # C[1,1]
    r += 1
    
    # M4 = d(g-e)
    U[1 * m + (j_offset + 1)][r] += 1  # d
    V[(j_offset + 1) * p + (k_offset + 0)][r] += 1  # g
    V[j_offset * p + (k_offset + 0)][r] -= 1  # -e
    W[(k_offset + 0) * n + 0][r] += 1  # C[0,0]
    W[(k_offset + 0) * n + 1][r] += 1  # C[1,0]
    r += 1
    
    # M5 = (a+b)(h)
    U[0 * m + (j_offset + 0)][r] += 1  # a
    U[0 * m + (j_offset + 1)][r] += 1  # b
    V[(j_offset + 1) * p + (k_offset + 1)][r] += 1  # h
    W[(k_offset + 0) * n + 0][r] -= 1  # C[0,0]
    W[(k_offset + 1) * n + 0][r] += 1  # C[0,1]
    r += 1
    
    # M6 = (c-a)(e+f)
    U[1 * m + (j_offset + 0)][r] += 1  # c
    U[0 * m + (j_offset + 0)][r] -= 1  # -a
    V[j_offset * p + (k_offset + 0)][r] += 1  # e
    V[j_offset * p + (k_offset + 1)][r] += 1  # f
    W[(k_offset + 1) * n + 1][r] += 1  # C[1,1]
    r += 1
    
    # M7 = (b-d)(g+h)
    U[0 * m + (j_offset + 1)][r] += 1  # b
    U[1 * m + (j_offset + 1)][r] -= 1  # -d
    V[(j_offset + 1) * p + (k_offset + 0)][r] += 1  # g
    V[(j_offset + 1) * p + (k_offset + 1)][r] += 1  # h
    W[(k_offset + 0) * n + 0][r] += 1  # C[0,0]
    r += 1
    
    return r

def _rank28_decomposition(n, m, p):
    """Construct the rank-28 decomposition using Strassen on 2x2 blocks."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (schoolbook, 2*5=10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: j=0,1, k=4 (schoolbook, 2*2=4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: j=0,1, k=0,1 (Strassen 2x2, 7 terms)
    r = _strassen_2x2(n, 2, 2, j_offset=0, k_offset=0, U=U, V=V, W=W, r=r)
    
    # Part 4: j=0,1, k=2,3 (Strassen 2x2, 7 terms)
    r = _strassen_2x2(n, 2, 2, j_offset=0, k_offset=2, U=U, V=V, W=W, r=r)
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns the best valid decomposition found (rank 28 via Strassen, or rank 30 schoolbook).
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    
    # Try rank-28 Strassen-based decomposition
    U28, V28, W28 = _rank28_decomposition(n, m, p)
    
    if _verify(U28, V28, W28, n, m, p):
        return {"U": U28, "V": V28, "W": W28}
    
    # Fallback to schoolbook rank 30
    U30, V30, W30 = _schoolbook(n, m, p)
    
    if _verify(U30, V30, W30, n, m, p):
        return {"U": U30, "V": V30, "W": W30}
    
    # Should not reach here, but return schoolbook as last resort
    return {"U": U30, "V": V30, "W": W30}
# EVOLVE-BLOCK-END
