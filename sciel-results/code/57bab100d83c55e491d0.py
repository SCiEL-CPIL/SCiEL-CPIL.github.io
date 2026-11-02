"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 @ 2x2.
    Returns U, V, W with rank 7.
    U has shape (4, 7) where U[i*2+j, c]
    V has shape (4, 7) where V[j*2+k, c]
    W has shape (4, 7) where W[k*2+i, c]
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each term (c, i, j, k), we need U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Term (i,j,k,coeff): contributes coeff * U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    
    # Let's define the terms for each M:
    # Each M is a product of two 2x2 matrices (sums/diffs of blocks)
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #    terms: (0,0,0,1), (0,0,1,1), (1,1,0,1), (1,1,1,1)
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    #    terms: (1,0,0,1), (1,1,0,1)
    # M3 = A00(B01-B11) = A00*B01 - A00*B11
    #    terms: (0,0,1,1), (0,0,1,-1)
    # M4 = A11(B10-B00) = A11*B10 - A11*B00
    #    terms: (1,1,1,1), (1,1,0,-1)
    # M5 = (A00+A10)B11 = A00*B11 + A10*B11
    #    terms: (0,0,1,1), (1,0,1,1)
    # M6 = (A10-A00)(B00+B10) = A10*B00 + A10*B10 - A00*B00 - A00*B10
    #    terms: (1,0,0,1), (1,0,1,1), (0,0,0,-1), (0,0,1,-1)
    # M7 = (A01-A11)(B01+B11) = A01*B01 + A01*B11 - A11*B01 - A11*B11
    #    terms: (0,1,1,1), (0,1,1,1), (1,1,1,-1), (1,1,1,-1)
    #    Wait, A01 is row 0 col 1, so i=0,j=1. B01 is row 0 col 1, so j=0,k=1.
    #    A01*B01: i=0, j=1 (for A), k=1 (for B), but B index is (j_B, k_B) where j_B is row of B.
    #    In our notation, A has indices (i_A, j_A), B has (j_B, k_B), C has (k_C, i_C).
    #    For the term A[i_A, j_A] * B[j_B, k_B] to contribute to C[k_C, i_C], we need j_A = j_B and k_B = k_C.
    #    So each term is (i_A, j_A, k_B, coeff) and it contributes to C[k_B, i_A].
    #    U[i_A*2+j_A, c], V[j_A*2+k_B, c], W[k_B*2+i_A, c]
    
    # Let me redo:
    # M1 = (A00+A11)(B00+B11)
    # A00*B00: i=0,j=0,k=0, coeff=1 -> C[0,0] += 1
    # A00*B11: i=0,j=0,k=1, coeff=1 -> C[1,0] += 1
    # A11*B00: i=1,j=1,k=0, coeff=1 -> C[0,1] += 1
    # A11*B11: i=1,j=1,k=1, coeff=1 -> C[1,1] += 1
    # M2 = (A10+A11)B00
    # A10*B00: i=1,j=0,k=0, coeff=1 -> C[0,1] += 1
    # A11*B00: i=1,j=1,k=0, coeff=1 -> C[0,1] += 1
    # M3 = A00(B01-B11)
    # A00*B01: i=0,j=0,k=1, coeff=1 -> C[1,0] += 1
    # A00*B11: i=0,j=0,k=1, coeff=-1 -> C[1,0] -= 1
    # M4 = A11(B10-B00)
    # A11*B10: i=1,j=1,k=0, coeff=1 -> C[0,1] += 1
    # A11*B00: i=1,j=1,k=0, coeff=-1 -> C[0,1] -= 1
    # M5 = (A00+A10)B11
    # A00*B11: i=0,j=0,k=1, coeff=1 -> C[1,0] += 1
    # A10*B11: i=1,j=0,k=1, coeff=1 -> C[1,1] += 1
    # M6 = (A10-A00)(B00+B10)
    # A10*B00: i=1,j=0,k=0, coeff=1 -> C[0,1] += 1
    # A10*B10: i=1,j=0,k=0, coeff=1 -> C[0,1] += 1  (B10 is row 1 col 0, so j_B=1, k_B=0)
    # Wait, B10 means B[1,0], so j_B=1, k_B=0. But in A10*B10, A is A[1,0] so j_A=0. 
    # For A[i_A, j_A] * B[j_B, k_B], we need j_A = j_B. Here j_A=0, j_B=1. They don't match!
    # So A10 * B10 is not a valid term for matrix multiplication.
    # Let me re-examine Strassen.
    
    # Actually, in Strassen, the blocks are 2x2 matrices. Let me denote:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # Now, each M is a product of two 2x2 "matrices" (which are actually sums/diffs of scalar blocks).
    # But wait, A00, A01, etc. are themselves 1x1 blocks (scalars) in the 2x2 case? No, in the 2x2 case,
    # A is a 2x2 matrix, so A00, A01, A10, A11 are scalars.
    # So M1 = (A00+A11)(B00+B11) is a scalar product.
    # But in the tensor decomposition, we need to express C as a sum of rank-1 tensors.
    # For 2x2 @ 2x2, the tensor has 8 entries (2*2*2).
    # Each rank-1 term is U[i,j] * V[j,k] * W[k,i] for fixed (i,j,k).
    #
    # So for M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This contributes to C as:
    # C00 += A00*B00 + A11*B11 (from M1's contribution to C00, but wait, M1 is added to C00 with coeff 1)
    # Actually, let's think differently.
    
    # The standard way to build Strassen decomposition:
    # We have 7 multiplications M1..M7.
    # Each Mq is a product of two linear combinations of A entries and B entries.
    # Mq = (sum_a alpha_qa * A_a) * (sum_b beta_qb * B_b)
    # where A_a are entries of A, B_b are entries of B.
    # Then C = sum_q gamma_qc * Mq where gamma_qc are coefficients for each C entry.
    
    # For the tensor decomposition:
    # C[i,j] = sum_{a,b} A[a,b] * B[b,j] (for square matrices, but here we have 4x4 @ 4x5)
    # Wait, for 2x2 @ 2x2: C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # Each rank-1 term in the decomposition corresponds to a triple (i,k,j) with coefficient 1.
    # But Strassen groups these terms.
    
    # Let me just use the known Strassen decomposition for 2x2.
    # The 7 terms are:
    # Term 1: (A00+A11)(B00+B11) -> contributes to C00, C11
    # Term 2: (A10+A11)B00 -> contributes to C10, C11
    # Term 3: A00(B01-B11) -> contributes to C01, C11
    # Term 4: A11(B10-B00) -> contributes to C10, C00
    # Term 5: (A00+A10)B11 -> contributes to C11, C00
    # Term 6: (A10-A00)(B00+B10) -> contributes to C10
    # Term 7: (A01-A11)(B01+B11) -> contributes to C01, C00
    
    # For each term, we need to expand it into rank-1 tensors.
    # But each term is already a product of two linear forms, so it's a sum of rank-1 tensors.
    # However, in the tensor decomposition, each "column" corresponds to one rank-1 term.
    # So if a term is (A00+A11)(B00+B11), it expands to 4 rank-1 terms.
    # But that would give rank 7*4 = 28, which is not 7.
    
    # I think I'm confusing myself. Let me look up the correct formulation.
    # In the tensor decomposition for matrix multiplication, each column c corresponds to a triple (i,j,k)
    # and the contribution is A[i,j] * B[j,k] * C[k,i] (with appropriate signs/coefficients).
    #
    # For Strassen, the decomposition has 7 columns. Each column c has:
    # U[i,j,c], V[j,k,c], W[k,i,c] for all i,j,k.
    # The product U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c gives C[i,k] (or C[k,i] depending on convention).
    
    # Let me just hardcode the Strassen 2x2 decomposition.
    # After research, the standard Strassen decomposition for 2x2 has rank 7.
    # The entries are as follows (using 0-based indexing):
    
    # I'll use a different approach: just return the schoolbook for now and focus on the 4x4x5 structure.
    # For 4x4 @ 4x5, we can use a block decomposition.
    
    # Actually, let me just implement a basic Strassen-like decomposition for 4x4 @ 4x4 and then extend to 4x5.
    # But that's complex. Let me try a simpler approach: use the known lower bound.
    # The schoolbook rank for 4x4 @ 4x5 is 80.
    # Strassen for 4x4 @ 4x4 has rank 7^2 = 49 (if we use Strassen recursively).
    # For 4x4 @ 4x5, we can pad the 4x5 to 4x8 and use Strassen, but that gives higher rank.
    # Or we can use a direct decomposition.
    
    # Let me just return the schoolbook for now, but with a time guard.
    return None  # Placeholder, will be replaced

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # For now, return schoolbook
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
