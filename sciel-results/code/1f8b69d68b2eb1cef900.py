"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the tensor T[i,j,k] for matrix multiplication (n x m) * (m x p).
    We store it as a 3D array T[i,j,k] = 1 if the (i,j) input and (j,k) input
    contribute to output (i,k).
    Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has entries T[i,j,k] = 1 for all valid i,j,k.
    """
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _tensor_to_vec(T, n, m, p):
    """Flatten tensor to a vector. Index: (i,j,k) -> i*m*p + j*p + k"""
    return T.reshape(-1).astype(int)

def _vec_to_tensor(v, n, m, p):
    """Unflatten vector to tensor."""
    return v.reshape((n, m, p))

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        # u is indexed by (i,j), v by (j,k), w by (k,i)
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        # For each (i,j,k): u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    # The true tensor is all ones
    expected = np.ones((n, m, p), dtype=float)
    if not np.allclose(T, expected, atol=1e-9):
        return False
    return True

def _strassen_2x2(A, B):
    """Multiply two 2x2 matrices using Strassen's 7-multiplication algorithm.
    Returns the 7 products and their combinations.
    """
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    e, f, g, h = B[0,0], B[0,1], B[1,0], B[1,1]
    
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (f - h)
    M4 = d * (g - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    return C11, C12, C21, C22

def _construct_strassen_28():
    """Construct the Strassen-based rank-28 decomposition for 2x4 * 4x4.
    
    We split the 4x4 matrix into four 2x2 blocks.
    A is 2x4: [A1 | A2] where A1, A2 are 2x2
    B is 4x4: [[B11, B12], [B21, B22]] where each is 2x2
    
    C = A * B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each 2x2*2x2 product uses Strassen's 7 multiplications.
    Total: 4 * 7 = 28 scalar multiplications.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # We'll use symbolic variables to track the Strassen products.
    # For each of the 4 block products, we have 7 Strassen terms.
    
    # Block product 1: A1 * B11
    # A1 = rows 0,1 of A (cols 0,1), B11 = rows 0,1 of B (cols 0,1)
    # Strassen terms for A1*B11:
    # M1 = (A1[0,0]+A1[1,1])*(B11[0,0]+B11[1,1])
    # M2 = (A1[1,0]+A1[1,1])*B11[0,0]
    # M3 = A1[0,0]*(B11[0,1]-B11[1,1])
    # M4 = A1[1,1]*(B11[1,0]-B11[0,0])
    # M5 = (A1[0,0]+A1[0,1])*B11[1,1]
    # M6 = (A1[1,0]-A1[0,0])*(B11[0,0]+B11[0,1])
    # M7 = (A1[0,1]-A1[1,1])*(B11[1,0]+B11[1,1])
    
    # Let's define a helper to add a Strassen block product
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, base_col):
        """Add 7 Strassen terms for a 2x2 * 2x2 block product.
        a_row_offset, a_col_offset: offset in A (2x4) for the 2x2 block
        b_row_offset, b_col_offset: offset in B (4x4) for the 2x2 block
        base_col: starting column index in U, V, W
        """
        # A block: rows a_row_offset to a_row_offset+1, cols a_col_offset to a_col_offset+1
        # B block: rows b_row_offset to b_row_offset+1, cols b_col_offset to b_col_offset+1
        
        # Strassen M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[a_row_offset*2 + 0][base_col + 0] = 1  # A[0,0]
        U[a_row_offset*2 + 1][base_col + 0] = 1  # A[1,1]
        V[b_row_offset*4 + 0][base_col + 0] = 1  # B[0,0]
        V[b_row_offset*4 + 3][base_col + 0] = 1  # B[1,1]
        # Output: C[0,0] += M1, C[0,1] -= M5, C[1,0] += M4, C[1,1] += M1
        # Wait, I need to be more careful about the W indices.
        # W is indexed by (k, i) where k is output col (0..3), i is output row (0..1)
        # W[k*n + i] = W[k*2 + i]
        
        # Let me re-derive. For the block product A1*B11 contributing to C[0:2, 0:2]:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 contributes to C11 (+1) and C22 (+1)
        W[0*2 + 0][base_col + 0] = 1  # C11: k=0, i=0
        W[1*2 + 1][base_col + 0] = 1  # C22: k=1, i=1
        
        # M2: (A[1,0]+A[1,1])*B[0,0]
        U[a_row_offset*2 + 2][base_col + 1] = 1  # A[1,0]
        U[a_row_offset*2 + 3][base_col + 1] = 1  # A[1,1]
        V[b_row_offset*4 + 0][base_col + 1] = 1  # B[0,0]
        # M2 contributes to C21 (+1) and C22 (-1)
        W[0*2 + 1][base_col + 1] = 1  # C21: k=0, i=1
        W[1*2 + 1][base_col + 1] = -1  # C22: k=1, i=1
        
        # M3: A[0,0]*(B[0,1]-B[1,1])
        U[a_row_offset*2 + 0][base_col + 2] = 1  # A[0,0]
        V[b_row_offset*4 + 1][base_col + 2] = 1  # B[0,1]
        V[b_row_offset*4 + 3][base_col + 2] = -1  # B[1,1]
        # M3 contributes to C12 (+1) and C22 (+1)
        W[1*2 + 0][base_col + 2] = 1  # C12: k=1, i=0
        W[1*2 + 1][base_col + 2] = 1  # C22: k=1, i=1
        
        # M4: A[1,1]*(B[1,0]-B[0,0])
        U[a_row_offset*2 + 3][base_col + 3] = 1  # A[1,1]
        V[b_row_offset*4 + 2][base_col + 3] = 1  # B[1,0]
        V[b_row_offset*4 + 0][base_col + 3] = -1  # B[0,0]
        # M4 contributes to C11 (+1) and C21 (+1)
        W[0*2 + 0][base_col + 3] = 1  # C11: k=0, i=0
        W[0*2 + 1][base_col + 3] = 1  # C21: k=0, i=1
        
        # M5: (A[0,0]+A[0,1])*B[1,1]
        U[a_row_offset*2 + 0][base_col + 4] = 1  # A[0,0]
        U[a_row_offset*2 + 1][base_col + 4] = 1  # A[0,1]
        V[b_row_offset*4 + 3][base_col + 4] = 1  # B[1,1]
        # M5 contributes to C11 (-1) and C12 (+1)
        W[0*2 + 0][base_col + 4] = -1  # C11: k=0, i=0
        W[1*2 + 0][base_col + 4] = 1  # C12: k=1, i=0
        
        # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[a_row_offset*2 + 2][base_col + 5] = 1  # A[1,0]
        U[a_row_offset*2 + 0][base_col + 5] = -1  # A[0,0]
        V[b_row_offset*4 + 0][base_col + 5] = 1  # B[0,0]
        V[b_row_offset*4 + 1][base_col + 5] = 1  # B[0,1]
        # M6 contributes to C22 (+1)
        W[1*2 + 1][base_col + 5] = 1  # C22: k=1, i=1
        
        # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[a_row_offset*2 + 1][base_col + 6] = 1  # A[0,1]
        U[a_row_offset*2 + 3][base_col + 6] = -1  # A[1,1]
        V[b_row_offset*4 + 2][base_col + 6] = 1  # B[1,0]
        V[b_row_offset*4 + 3][base_col + 6] = 1  # B[1,1]
        # M7 contributes to C11 (+1)
        W[0*2 + 0][base_col + 6] = 1  # C11: k=0, i=0
    
    # Block 1: A1 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A1 (rows 0-1, cols 0-1) * B12 (rows 0-1, cols 2-3)
    add_strassen_block(0, 0, 0, 2, 7)
    
    # Block 3: A2 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 0-1)
    add_strassen_block(0, 2, 2, 0, 14)
    
    # Block 4: A2 (rows 0-1, cols 2-3) * B22 (rows 2-3, cols 2-3)
    add_strassen_block(0, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with the Strassen-based rank-28 decomposition
    result = _construct_strassen_28()
    
    # Verify the decomposition
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if Strassen verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
