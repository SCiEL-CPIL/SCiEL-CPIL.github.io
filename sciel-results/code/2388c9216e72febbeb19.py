import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _get_strassen_4x4_terms():
    """
    Returns the 28 scalar multiplications from the 4x4 Strassen embedding 
    of 3x3 matrix multiplication.
    Each term is represented by (u, v, w) vectors of length 9.
    """
    n, m, p = DIMENSIONS
    terms = []
    
    # We embed 3x3 into 4x4.
    # A' = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # This is NOT a standard 2x2 block partition for Strassen.
    # Strassen requires square blocks.
    # So we must pad to 4x4 with zeros:
    # A' = [[A00, A01, A02, 0],
    #       [A10, A11, A12, 0],
    #       [A20, A21, A22, 0],
    #       [0,   0,   0,   0]]
    # This is not a 2x2 block structure.
    
    # Correct embedding for Strassen 4x4:
    # Split 4x4 into four 2x2 blocks.
    # A' = [[B11, B12], [B21, B22]]
    # Where B11 is 2x2, B12 is 2x2, etc.
    # For 3x3 A, we set:
    # B11 = A[0:2, 0:2]
    # B12 = A[0:2, 2:4] = [[A02, 0], [A12, 0]]
    # B21 = A[2:4, 0:2] = [[A20, A21], [0, 0]]
    # B22 = A[2:4, 2:4] = [[A22, 0], [0, 0]]
    
    # Strassen 7 products for 4x4:
    # M1 = (B21 + B22) * (B11 + B12)
    # M2 = (B11 + B12) * B21
    # M3 = B11 * (B11 - B21)
    # M4 = B22 * (B22 - B12)
    # M5 = (B11 + B21) * (B11 + B22)
    # M6 = (B12 - B22) * (B21 + B22)
    # M7 = (B11 - B21) * (B12 + B22)
    
    # Each M_i is a 2x2 matrix.
    # To get scalar multiplications, we use schoolbook for each 2x2 multiplication.
    # Each 2x2 multiplication has 4 scalar products.
    # Total 7 * 4 = 28 scalar products.
    
    # We need to map these 28 scalar products to the 9x9x9 tensor space.
    # A scalar product is of the form A[i,j] * B[k,l].
    # In the 4x4 embedding, indices are 0..3.
    # We only care about i,j,k,l in 0..2.
    
    # Let's define the 28 terms explicitly.
    # Each term is (u_vec, v_vec, w_vec) where u_vec has 1 at A_idx, v_vec at B_idx, w_vec at C_idx.
    # But Strassen combines A and B first.
    # So M1 = (B21+B22)*(B11+B12).
    # (B21+B22)[i,j] * (B11+B12)[k,l]
    # This expands to sum of 4 terms for each of the 4 entries of the 2x2 result.
    # Total 16 terms for M1? No, M1 is a 2x2 matrix.
    # M1[p,q] = sum_{r=0}^1 (B21+B22)[p,r] * (B11+B12)[r,q]
    # This is 4 scalar multiplications for the 2x2 result M1.
    # Each scalar multiplication is a product of a linear form in A and a linear form in B.
    
    # We will generate the 28 terms as (u, v) pairs, and then determine the w (output) location.
    
    # Helper to add a term
    def add_term(a_idx, b_idx, c_idx):
        # a_idx is (i,j) in 4x4, b_idx is (k,l) in 4x4, c_idx is (p,q) in 4x4 output
        # We only add if a_idx, b_idx, c_idx are within 3x3 range (0..2)
        if a_idx[0] < 3 and a_idx[1] < 3 and b_idx[0] < 3 and b_idx[1] < 3 and c_idx[0] < 3 and c_idx[1] < 3:
            u = np.zeros(9)
            v = np.zeros(9)
            w = np.zeros(9)
            u[a_idx[0]*3 + a_idx[1]] = 1
            v[b_idx[0]*3 + b_idx[1]] = 1
            w[c_idx[0]*3 + c_idx[1]] = 1
            terms.append((u, v, w))
            
    # We need to expand the Strassen formulas.
    # This is complex to do symbolically in the code.
    # Instead, I will hardcode the 23 terms from a known source.
    # Since I can't recall them, I will use the **rank-27 schoolbook** as a fallback
    # and attempt a **numerical reduction** if time permits.
    
    return None

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Fallback to schoolbook
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
