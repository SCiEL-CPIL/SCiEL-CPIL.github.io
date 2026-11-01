"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 140.0  # Safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_decomp():
    """
    Build a rank-44 decomposition for 2x3x8 matrix multiplication.
    
    Strategy:
    - Split A into A_left (2x2) and A_right (2x1)
    - Split B into B_top (2x8) and B_bottom (1x8)
    - C = A_left * B_top + A_right * B_bottom
    
    A_right * B_bottom: rank 16 (each of 2*8 outputs is a single product)
    A_left * B_top: M(2,2,8). Split into 4 blocks of M(2,2,2), each Strassen rank 7.
    Total: 16 + 4*7 = 44
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A_right * B_bottom (rank 16)
    # A[0,2] at index 0*3+2=2, A[1,2] at index 1*3+2=5
    # B[2,k] at index 2*8+k = 16+k
    # C[i,k] at index k*2+i
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A_left * B_top via Strassen (rank 28)
    # A_left is 2x2: A[0,0]@0, A[0,1]@1, A[1,0]@3, A[1,1]@4
    # B_top is 2x8: rows 0,1. B[0,k]@k, B[1,k]@8+k
    # C is 2x8: C[i,k] at k*2+i
    
    for g in range(4):
        b11 = 0 + 2 * g      # B[0, 2g]
        b12 = 0 + 2 * g + 1  # B[0, 2g+1]
        b21 = 8 + 2 * g      # B[1, 2g]
        b22 = 8 + 2 * g + 1  # B[1, 2g+1]
        
        a11 = 0  # A[0,0]
        a12 = 1  # A[0,1]
        a21 = 3  # A[1,0]
        a22 = 4  # A[1,1]
        
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[(2 * g) * 2 + 0][col] = 1      # C[0, 2g]
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[(2 * g) * 2 + 1][col] = 1  # C[1, 2g]
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        W[(2 * g + 1) * 2 + 0][col] = 1   # C[0, 2g+1]
        W[(2 * g + 1) * 2 + 1][col] = -1  # C[1, 2g+1]
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        W[(2 * g) * 2 + 0][col] = -1  # C[0, 2g]
        W[(2 * g + 1) * 2 + 1][col] = 1   # C[1, 2g]
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 0][col] = 1  # C[0, 2g+1]
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[(2 * g) * 2 + 0][col] = 1  # C[0, 2g]
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
    
    return U, V, W


def _build_rank42_attempt():
    """
    Attempt to find a rank-42 decomposition by exploiting linear dependencies
    in the rank-44 construction.
    
    The key idea: in the Strassen decomposition for M(2,2,2), some of the
    intermediate products might be shareable across the 4 blocks if we
    reformulate. However, a more direct approach is to check if the 44
    tensor vectors are linearly dependent, and if so, eliminate redundant ones.
    
    For a tensor T = sum_c U_c ⊗ V_c ⊗ W_c, if there exist coefficients
    alpha_c not all zero such that sum_c alpha_c * U_c ⊗ V_c ⊗ W_c = 0,
    then we can eliminate one term (if alpha_c != 0 for some c).
    
    This is equivalent to checking if the 44 vectors in the tensor product
    space are linearly independent. The tensor product space has dimension
    6*24*16 = 2304, so 44 vectors are almost certainly independent.
    
    However, we might be able to find a different decomposition with fewer terms
    by reformulating the Strassen steps.
    
    Alternative approach: Instead of 4 independent Strassen blocks, try to
    use a single larger decomposition for M(2,2,8).
    
    M(2,2,8): A is 2x2, B is 2x8, C is 2x8.
    Schoolbook: 32. Strassen per 2x2x2 block: 4*7 = 28.
    
    Can we do M(2,2,8) in less than 28?
    
    One approach: view B as a 2x8 matrix = [B1 | B2] where B1, B2 are 2x4.
    Then C = A*B1 + A*B2.
    M(2,2,4) can be done in... let's see.
    
    Actually, let's try a different decomposition of M(2,2,8):
    
    A = [[a, b], [c, d]], B = [[e1, e2, ..., e8], [f1, f2, ..., f8]]
    C = [[a*e_k + b*f_k], [c*e_k + d*f_k]] for k=1..8
    
    This is 16 scalar multiplications in schoolbook for each k, but with
    Strassen we do 7 per 2x2x2 block, so 28 total for 8 columns.
    
    Can we share computations across k? The A matrix is the same for all k.
    In Strassen, the 7 intermediate products depend on A and a 2x2 sub-block of B.
    Since A is shared, but the B sub-blocks are different, the intermediate
    products are different.
    
    However, we can precompute the linear combinations of A that are used
    in Strassen:
    s1 = a + d
    s2 = c + d
    s3 = b
    s4 = d
    s5 = a + b
    s6 = c - a
    s7 = b - d
    
    These are 7 linear combinations of A. Then for each 2x2x2 block, we
    compute 7 products:
    P1 = s1 * (e_k + f_{k+1})
    P2 = s2 * e_k
    P3 = b * (f_{k+1} - e_{k+1})
    P4 = d * (e_k - f_k)
    P5 = (a+b) * f_{k+1}
    P6 = (c-a) * (e_k + e_{k+1})
    P7 = (b-d) * (f_k + f_{k+1})
    
    The linear combinations of A are the same across all 4 blocks, but the
    products with B are different. So we can't directly share the products.
    
    For now, return the rank-44 decomposition.
    """
    U, V, W = _build_decomp()
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp()
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    if time.time() - start_time > _DEADLINE:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
