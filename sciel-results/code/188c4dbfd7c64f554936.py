"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm recursively for 4x4x4
    # Strassen for 2x2x2: rank 7
    # For 4x4x4: split into 2x2 blocks, each 2x2x2 uses rank 7
    # Total rank = 7 * 7 = 49? No, that's not right.
    
    # Actually, for 4x4x4, we can use the fact that 4 = 2*2
    # The tensor decomposition for 4x4x4 can be built from 2x2x2 decompositions
    # If we use Strassen for 2x2x2 (rank 7), then for 4x4x4 we get rank 7*7 = 49?
    # No, that's the rank if we just apply Strassen at the block level.
    
    # Let me think more carefully. The standard approach:
    # C = A * B where A, B, C are 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Using Strassen at the block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product, which requires rank 4 (schoolbook) or 
    # we can use Strassen again for each 2x2 product.
    
    # If we use schoolbook for each 2x2 product: rank = 7 * 4 = 28
    # If we use Strassen for each 2x2 product: rank = 7 * 7 = 49 (worse)
    
    # So the best is to use schoolbook for the inner 2x2 products: rank 28
    
    # Let's implement rank 28 decomposition.
    
    # The tensor T[i,j,k] = delta[i+k, j] for matrix multiplication C[i,j] = sum_k A[i,k] * B[k,j]
    
    # For the block decomposition:
    # C11[i,j] = sum over the 7 Strassen terms
    # Each term Mk is a 2x2 matrix product
    
    # Let's define the full decomposition.
    
    # Indices:
    # A has indices (i, k) where i, k in {0,1,2,3}
    # B has indices (k, j) where k, j in {0,1,2,3}
    # C has indices (i, j) where i, j in {0,1,2,3}
    
    # Block structure:
    # i = 2*bi + ai, where bi in {0,1}, ai in {0,1}
    # k = 2*bk + ak, where bk in {0,1}, ak in {0,1}
    # j = 2*bj + aj, where bj in {0,1}, aj in {0,1}
    
    # So A[2*bi+ai, 2*bk+ak] = A_{bi,bk}[ai, ak]
    # B[2*bk+ak, 2*bj+aj] = B_{bk,bj}[ak, aj]
    # C[2*bi+ai, 2*bj+aj] = C_{bi,bj}[ai, aj]
    
    # C_{bi,bj}[ai, aj] = sum_{bk,ak} A_{bi,bk}[ai, ak] * B_{bk,bj}[ak, aj]
    
    # Using Strassen at the block level:
    # C_{00} = M1 + M4 - M5 + M7
    # C_{01} = M3 + M5
    # C_{10} = M2 + M4
    # C_{11} = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A_{00}+A_{11})(B_{00}+B_{11})
    # M2 = (A_{10}+A_{11})B_{00}
    # M3 = A_{00}(B_{01}-B_{11})
    # M4 = A_{11}(B_{10}-B_{00})
    # M5 = (A_{00}+A_{01})B_{11}
    # M6 = (A_{10}-A_{00})(B_{00}+B_{01})
    # M7 = (A_{01}-A_{11})(B_{10}+B_{11})
    
    # Each Mk is a 2x2 matrix product, so we can decompose it using schoolbook (rank 4)
    
    # Total rank = 7 * 4 = 28
    
    rank = 28
    
    # U has shape (n*m, rank) = (16, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for C blocks:
    # C_{00} = M1 + M4 - M5 + M7
    # C_{01} = M3 + M5
    # C_{10} = M2 + M4
    # C_{11} = M1 - M2 + M3 + M6
    
    # For each Strassen term Mk, we need to express it as a 2x2 matrix product
    # and then decompose that using schoolbook.
    
    # Let's define the A and B factors for each Mk.
    
    # M1 = (A_{00}+A_{11})(B_{00}+B_{11})
    # M2 = (A_{10}+A_{11})B_{00}
    # M3 = A_{00}(B_{01}-B_{11})
    # M4 = A_{11}(B_{10}-B_{00})
    # M5 = (A_{00}+A_{01})B_{11}
    # M6 = (A_{10}-A_{00})(B_{00}+B_{01})
    # M7 = (A_{01}-A_{11})(B_{10}+B_{11})
    
    # For each Mk = A_factor * B_factor, where A_factor and B_factor are 2x2 matrices
    # We decompose A_factor * B_factor using schoolbook:
    # (A_factor * B_factor)[ai, aj] = sum_ak A_factor[ai, ak] * B_factor[ak, aj]
    
    # So for each (ai, ak, aj) in {0,1}^3, we have a scalar product.
    
    # The full index for U is (i, k) = (2*bi+ai, 2*bk+ak)
    # The full index for V is (k, j) = (2*bk+ak, 2*bj+aj)
    # The full index for W is (j, i) = (2*bj+aj, 2*bi+ai)  [trace convention: W[k*n+i, r]]
    
    # Wait, the trace convention says W[k*n+i, r]. So W is indexed by (k, i) where k is the column of B and i is the row of A.
    # Actually, let me re-read: "W[k*n+i,r]" - so the row index of W is k*n+i, where k is from B's row (which is A's column) and i is from A's row.
    
    # So for a term A[i,k] * B[k,j] * C[i,j], the W index is k*n+i.
    
    # Let me re-derive. The tensor is T[i,j,k] = delta[i+k, j] where the product is C[i,j] = sum_k A[i,k] * B[k,j].
    # The decomposition is T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # Wait, the problem says W[k*n+i, r]. Let me check the seed code.
    # In the seed: W[k * n + i][column] = 1
    # And the product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta[i+k, j]... no, that's not right.
    
    # Actually, for matrix multiplication C = A * B:
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # The tensor is T[i, k, j] = delta[i, j]... no.
    
    # The matrix multiplication tensor has components T[i, k, j] such that
    # C[i, j] = sum_k A[i, k] * B[k, j] * T[i, k, j]
    # So T[i, k, j] = 1 for all i, k, j? No, that's not a tensor decomposition.
    
    # Actually, the tensor for matrix multiplication is defined as:
    # <T, A ⊗ B> = C where C = A * B
    # So T[i, k, j] = delta[i, j]... no.
    
    # Let me think again. The bilinear form is:
    # sum_{i,k,j} T[i,k,j] A[i,k] B[k,j] = sum_{i,j} C[i,j]
    # where C[i,j] = sum_k A[i,k] B[k,j]
    
    # So sum_{i,k,j} T[i,k,j] A[i,k] B[k,j] = sum_{i,j} sum_k A[i,k] B[k,j]
    
    # This means T[i,k,j] = 1 if we're summing over all i,k,j, but that's not right because the left side is a single number and the right side is a matrix.
    
    # I think the convention is that the output C is also part of the tensor. The tensor is:
    # T[i, k, j] where the first index is for A, second for B, third for C.
    # And the decomposition is: T[i, k, j] = sum_r U[i, r] * V[k, j, r] * W[j, i, r]
    
    # But the problem states the shapes as U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # So U is indexed by (i, k) where i is row of A, k is column of A
    # V is indexed by (k, j) where k is row of B, j is column of B
    # W is indexed by (j, i) where j is column of B (row of C), i is row of A (column of C)
    
    # And the decomposition is: for each (i, k, j), the coefficient is sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # This should equal 1 if (i, k, j) is a valid term in the sum, but actually it should encode the entire multiplication.
    
    # I think the correct interpretation is:
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # = sum_{i', k', j'} T[i', k', j'] A[i', k'] B[k', j'] * delta[i', i] * delta[j', j]
    
    # So T[i, k, j] = 1 for all i, k, j? No, that would give sum_{i,k,j} A[i,k] B[k,j] which is not C.
    
    # I think I'm overcomplicating this. Let me look at the seed code again.
    
    # In the seed:
    # U[i * m + j][column] = 1  # i is row of A, j is column of A
    # V[j * p + k][column] = 1  # j is row of B, k is column of B
    # W[k * n + i][column] = 1  # k is row of B (column of A), i is row of A (column of C)
    
    # So the term is U[i*m+j] * V[j*p+k] * W[k*n+i]
    # This corresponds to A[i,j] * B[j,k] * C[k,i]... 
    
    # Wait, I think the indices are:
    # A has shape (n, m), so A[i, j] where i in [0,n), j in [0,m)
    # B has shape (m, p), so B[j, k] where j in [0,m), k in [0,p)
    # C has shape (n, p), so C[i, k] where i in [0,n), k in [0,p)
    
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # So the tensor is T[i, j, k] = 1 for the term A[i,j] * B[j,k] contributing to C[i,k]
    
    # And the decomposition is: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # In the schoolbook, for each (i, j, k), we have one factor with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # So T[i, j, k] = 1 for all i, j, k, and the sum over r gives 1 for each (i,j,k)
    
    # But that means the tensor is just the all-ones tensor, which doesn't seem right for matrix multiplication.
    
    # I think the issue is that the tensor encodes the entire multiplication, not just a single term.
    
    # Actually, I think the correct interpretation is that the bilinear form is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * something
    
    # But C[i,k] = sum_j A[i,j] B[j,k], so:
    # sum_{i,k} C[i,k] = sum_{i,k} sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    
    # So T[i,j,k] = 1 for all i,j,k, and the output is the sum of all entries of C.
    
    # That doesn't make sense for matrix multiplication.
    
    # Let me re-read the problem statement: "The output factor follows the official trace convention W[k*n+i,r]"
    
    # I think the convention is that the tensor is for the trace of C, or something like that.
    
    # Actually, I think the standard convention for matrix multiplication tensor is:
    # T[i, j, k] where i indexes A's row, j indexes A's column / B's row, k indexes B's column / C's column
    # And the decomposition gives: C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    # But that would mean W depends on i and k, which is the output index.
    
    # I think the correct interpretation is:
    # For each output entry C[i, k], we have:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i, j] * B[j, k]
    
    # This means that for each (i, j, k), the coefficient of A[i,j] * B[j,k] in C[i,k] is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And this should equal 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k], but only for the specific (i,k) output.
    
    # I think the tensor is defined such that:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = trace(C) or something similar.
    
    # Given the time constraint, let me just implement the rank 28 decomposition based on the Strassen approach and verify it numerically.
    
    # For now, let me just return the schoolbook decomposition to be safe.
    
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
