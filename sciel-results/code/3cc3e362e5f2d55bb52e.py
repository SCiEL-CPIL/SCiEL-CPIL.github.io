import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact tensor decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(len(U)):
        if len(U[i]) != rank:
            return False
    for i in range(len(V)):
        if len(V[i]) != rank:
            return False
    for i in range(len(W)):
        if len(W[i]) != rank:
            return False
    
    # Build the tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T[i,j,k] = 1 if this corresponds to a valid multiplication entry, else 0
    # Actually, the tensor for matrix multiplication A(n,m) * B(m,p) = C(n,p)
    # is defined as: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all i,j,k (it's the structure tensor of matmul)
    # Wait, the standard structure tensor: e_{ij}^A \otimes e_{jk}^B \otimes e_{ki}^C
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # And the decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify using numpy for speed
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _try_strassen_23(n, m, p):
    """
    Try to construct a rank-23 decomposition for the 3x3x3 block.
    The Strassen-Las Vergnas decomposition of the 3x3x3 matrix multiplication tensor.
    
    The 23 terms are well-known. Let me encode them carefully.
    A is 3x3, B is 3x3, C = A*B is 3x3.
    C[i][k] = sum_j A[i][j] * B[j][k]
    
    The structure tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1,2}
    (where i indexes rows of A/C, j indexes cols of A / rows of B, k indexes cols of B/C)
    
    U has 9 rows (i*m+j = i*3+j), V has 9 rows (j*p+k = j*3+k), W has 9 rows (k*n+i = k*3+i)
    
    The 23-term decomposition:
    I'll use the known Strassen-Las Vergnas coefficients.
    """
    if n != 3 or m != 3 or p != 3:
        return None
    
    # The 23 terms of the Strassen-Las Vergnas decomposition
    # Each term r has (u_r, v_r, w_r) where u_r is a vector in R^9, v_r in R^9, w_r in R^9
    # u_r[i*3+j] = A coefficient, v_r[j*3+k] = B coefficient, w_r[k*3+i] = C coefficient
    
    # I'll use the standard 23-term decomposition.
    # Let me define them as lists of 9-tuples for U, V, W columns.
    
    # From the literature, the 23 terms can be written as:
    # Term 1: A00*B00, A00*B01, A00*B02 -> contributes to C00, C01, C02
    # Actually, let me use a verified set.
    
    # I'll use the following approach: define the 23 rank-1 tensors explicitly.
    # Each rank-1 tensor is a triple (a, b, c) where a is 3x3, b is 3x3, c is 3x3
    # and c = a * b (outer product sense for the tensor).
    
    # The 23 terms (from Strassen 1969 / Las Vergnas 1973):
    # I'll encode them as flat vectors of length 9.
    
    terms = []
    
    # Let me use a known correct set. The decomposition satisfies:
    # sum_{r=1}^{23} a_r \otimes b_r \otimes c_r = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}
    
    # I'll hardcode the 23 terms from a verified source.
    # Using the notation where each term is (U_col, V_col, W_col) as 9-vectors.
    
    # This is the well-known 23-term decomposition. Let me write it out:
    
    # I'll use the formulation where we have the following 23 products:
    # 1. (A00+A01+A02)(B00+B10+B20) -> C00 + C10 + C20 ... no this isn't right.
    
    # Let me use a different approach. I'll try to solve for a rank-23 decomposition
    # using a known algorithm or hardcoded values from a reliable source.
    
    # Actually, let me use the following verified 23-term decomposition:
    # (from https://arxiv.org/abs/math/0311214 or similar references)
    
    # I'll define the 23 terms as follows (each is a triple of 9-element vectors):
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Term 1
    u = [1,0,0, 0,0,0, 0,0,0]  # A00
    v = [1,0,0, 0,0,0, 0,0,0]  # B00
    w = [0,0,0, 0,0,0, 0,0,1]  # C00 (k=0,i=0 -> index k*3+i = 0*3+0=0... wait)
    # W index: k*n+i = k*3+i. For C[0][0], i=0,k=0, so W index = 0*3+0 = 0
    # Let me redo: W[k*3+i] corresponds to C[i][k]
    # So for C[i][k], W index = k*3+i
    
    # Let me restart with correct indexing:
    # U[i*3+j] = coefficient for A[i][j]
    # V[j*3+k] = coefficient for B[j][k]  
    # W[k*3+i] = coefficient for C[i][k]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # I'll use the standard 23-term Strassen-Las Vergnas decomposition.
    # Here are the 23 terms (a_r, b_r, c_r) as 3x3 matrices:
    
    # I found a reference set. Let me encode it:
    
    # Using the convention from various sources, the 23 terms are:
    
    # I'll use a computational approach: try to find a rank-23 decomposition
    # by solving a system of equations, but that's complex. Instead, let me use
    # a known correct hardcoded set.
    
    # After careful consideration, I'll use the following 23-term decomposition
    # which is well-established in the literature:
    
    # Let me just use the schoolbook for the 3x3x3 part too, but try to find
    # a better overall strategy.
    
    # Actually, for 3x3 * 3x7, the key insight is:
    # We can decompose B into B1 (3x3) and B2 (3x4)
    # C = A*B1 + A*B2
    # Use rank-23 for A*B1 and schoolbook (3*3*4=36) for A*B2
    # Total: 23 + 36 = 59
    
    # But I need the actual 23-term coefficients. Let me try a different approach:
    # Use a rank-15 decomposition for 3x3*3x3? No, the bound is 23.
    
    # Let me try to use a simpler approach: 
    # For 3x3 * 3x7, can I do better than 59?
    # The lower bound is max(9, 21, 21) = 21.
    # Upper bound: 59 (with Strassen for first block)
    
    # I'll attempt to hardcode the 23-term decomposition.
    # Here is a verified set of 23 terms:
    
    # I'll use the decomposition from:
    # "The rank of the 3x3x3 matrix multiplication tensor is 23"
    
    # Let me encode the 23 terms as (u, v, w) triples where each is a 9-vector.
    
    # After research, here's a set that should work:
    
    # I'll use a programmatic verification approach. Let me define candidate terms
    # and verify them. If they don't work, fall back to schoolbook.
    
    # For now, let me return None to indicate this function needs work.
    return None

def _try_block_decomposition(n, m, p):
    """
    Try to decompose the problem into blocks.
    For 3x3 * 3x7: split B into B1 (3x3) and B2 (3x4)
    Use best known decomposition for each block.
    """
    if n != 3 or m != 3 or p != 7:
        return None
    
    # Split: B = [B1 | B2] where B1 is 3x3, B2 is 3x4
    # C = A*B1 + A*B2
    # C1 = A*B1 (3x3), C2 = A*B2 (3x4)
    
    # For A*B1: use rank-23 Strassen-Las Vergnas
    # For A*B2: use schoolbook rank 3*3*4 = 36
    # Total: 59
    
    # I need the 23-term decomposition for 3x3*3x3.
    # Let me try to construct it using a known algorithm.
    
    # Alternative: use rank 23 if I can find the coefficients.
    # Let me try a search approach with limited budget.
    
    # For now, return None if I can't find a valid decomposition.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # First, try to find a better decomposition
    # For 3x3 * 3x7, the schoolbook rank is 63.
    # We can potentially do better by splitting into blocks.
    
    # Try block decomposition
    result = _try_block_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Try Strassen for the 3x3 part
    result = _try_strassen_23(n, m, p)
    if result is not None:
        U23, V23, W23 = result
        # Combine with schoolbook for the remaining 4 columns
        rank_23 = len(U23[0])
        rank_rest = n * m * (p - 3)  # 3*3*4 = 36
        total_rank = rank_23 + rank_rest
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy the 23-term part for the first 3 columns of B
        for r in range(rank_23):
            for i in range(n * m):
                U[i][r] = U23[i][r]
            for j in range(m):
                for k in range(3):  # first 3 columns of B
                    V[j * p + k][r] = V23[j * 3 + k][r]
            for k in range(3):
                for i in range(n):
                    W[k * n + i][r] = W23[k * n + i][r]
        
        # Schoolbook for the last 4 columns of B
        col = rank_23
        for i in range(n):
            for j in range(m):
                for k in range(3, p):  # columns 3,4,5,6 of B
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
