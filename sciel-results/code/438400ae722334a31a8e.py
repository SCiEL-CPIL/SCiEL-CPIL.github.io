import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    
    # Target rank 68
    rank = 68
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # 1. A[0,0] * B[0, :] -> C[0, :]
    # A[0,0] is at U[0,0]
    # B[0,k] is at V[0*8+k, col]
    # C[0,k] is at W[k*3+0, col]
    for k in range(8):
        U[0, col] = 1
        V[k, col] = 1
        W[k * n + 0, col] = 1
        col += 1
    # col is now 8
    
    # 2. A[0,1:3] * B[1:3, :] -> C[0, :]
    # A[0,1] * B[1,k] -> C[0,k]
    # A[0,2] * B[2,k] -> C[0,k]
    for k in range(8):
        # Term A[0,1] * B[1,k]
        U[0 * m + 1, col] = 1
        V[1 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1
        
        # Term A[0,2] * B[2,k]
        U[0 * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1
    # col is now 8 + 16 = 24
    
    # 3. A[1:3, 0] * B[0, :] -> C[1:3, :]
    # A[1,0] * B[0,k] -> C[1,k]
    # A[2,0] * B[0,k] -> C[2,k]
    for k in range(8):
        # Term A[1,0] * B[0,k]
        U[1 * m + 0, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1
        
        # Term A[2,0] * B[0,k]
        U[2 * m + 0, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
    # col is now 24 + 16 = 40
    
    # 4. A[1:3, 1:3] * B[1:3, :] -> C[1:3, :]
    # A[1:3, 1:3] is 2x2. B[1:3, :] is 2x8.
    # Split B[1:3, :] into 4 blocks of 2x2: B_block_j for j=0..3
    # Each block uses Strassen rank 7.
    # Total 4 * 7 = 28.
    
    # Strassen coefficients for 2x2 multiplication
    # Local indices 0,1 for rows and cols of the 2x2 submatrices
    # A_sub[i, j] corresponds to global A[i+1, j+1]
    # B_sub[i, k] corresponds to global B[i+1, 2*j_block + k]
    # C_sub[i, k] corresponds to global C[i+1, 2*j_block + k]
    
    strassen_terms = [
        # (A_idx, B_idx, C_idx, sign)
        # P1 = (A00+A11)(B00+B11)
        ((0, 0), (0, 0), (0, 0), 1),
        ((1, 1), (0, 0), (0, 0), 1),
        ((0, 0), (1, 1), (0, 0), 1),
        ((1, 1), (1, 1), (0, 0), 1),
        # P4 = A11(B10-B00)
        ((1, 1), (1, 0), (0, 0), 1),
        ((1, 1), (0, 0), (0, 0), -1),
        # P7 = (A01-A11)(B10+B11)
        ((0, 1), (1, 0), (0, 0), 1),
        ((0, 1), (1, 1), (0, 0), 1),
        ((1, 1), (1, 0), (0, 0), -1),
        ((1, 1), (1, 1), (0, 0), -1),
        
        # P3 = A00(B01-B11)
        ((0, 0), (0, 1), (0, 1), 1),
        ((0, 0), (1, 1), (0, 1), -1),
        # P5 = (A00+A01)B11
        ((0, 0), (1, 1), (0, 1), 1),
        ((0, 1), (1, 1), (0, 1), 1),
        
        # P2 = (A10+A11)B00
        ((1, 0), (0, 0), (1, 0), 1),
        ((1, 1), (0, 0), (1, 0), 1),
        # P4 part for C10
        ((1, 1), (1, 0), (1, 0), 1),
        ((1, 1), (0, 0), (1, 0), -1),
        
        # P1 part for C11
        ((0, 0), (0, 0), (1, 1), 1),
        ((1, 1), (0, 0), (1, 1), 1),
        ((0, 0), (1, 1), (1, 1), 1),
        ((1, 1), (1, 1), (1, 1), 1),
        # P6 = (A10-A00)(B00+B01)
        ((1, 0), (0, 0), (1, 1), 1),
        ((1, 0), (0, 1), (1, 1), 1),
        ((0, 0), (0, 0), (1, 1), -1),
        ((0, 0), (0, 1), (1, 1), -1),
        # P3 part for C11
        ((0, 0), (0, 1), (1, 1), 1),
        ((0, 0), (1, 1), (1, 1), -1),
        # P2 part for C11
        ((1, 0), (0, 0), (1, 1), -1),
        ((1, 1), (0, 0), (1, 1), -1)
    ]
    
    # Wait, the standard Strassen decomposition has 7 multiplications, but the additions are free in the tensor model?
    # No, the tensor rank counts the multiplications. The additions are linear combinations of the terms.
    # So we just need to define the 7 multiplicative terms for each 2x2 block.
    # The 7 terms are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to expand these into unit terms (single A entry * single B entry).
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = A10B00 + A11B00
    # M3 = A00B01 - A00B11
    # M4 = A11B10 - A11B00
    # M5 = A00B11 + A01B11
    # M6 = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # Total unit terms: 4 + 2 + 2 + 2 + 2 + 4 + 4 = 20?
    # No, Strassen reduces multiplications from 8 to 7.
    # But the tensor rank is the number of scalar multiplications.
    # If we use Strassen, we introduce intermediate sums.
    # The rank of the tensor is the minimum number of scalar multiplications.
    # For 2x2, rank is 7.
    # The 7 multiplications are on the *sums* of entries.
    # But the tensor decomposition must be in terms of the original entries A[i,j] and B[k,l].
    # So we cannot directly use "A00+A11" as a single entry in U.
    # We must expand the 7 Strassen multiplications into the original basis.
    # This expansion results in 20 terms?
    # No, the rank is 7. This means there exists a decomposition with 7 terms.
    # The terms are of the form (sum of A) * (sum of B).
    # But U, V, W matrices have entries for single A[i,j] and B[k,l].
    # So we cannot put "A00+A11" in U.
    # We must use the fact that the tensor is bilinear.
    # The standard Strassen algorithm works because we can compute the sums first.
    # But in the tensor decomposition model, we are limited to the original coordinates.
    # The rank of M_2,2,2 is 7.
    # This 7-rank decomposition uses terms like (A00+A11)(B00+B11).
    # This term corresponds to 4 unit tensors: A00B00, A00B11, A11B00, A11B11.
    # If we sum these 4 unit tensors, we get the bilinear form (A00+A11)(B00+B11).
    # So one "Strassen term" replaces 4 unit terms?
    # No. The rank is the number of terms in the decomposition.
    # If we use the 7 Strassen terms, each term is a bilinear form of 2x2 submatrices.
    # But we need to express the result in terms of the original 2x2 matrices.
    # The 7 Strassen terms are not unit tensors.
    # However, the rank 7 decomposition is valid.
    # We can implement it by creating 7 columns in U, V, W.
    # For each Strassen term, the U column will have 1s or -1s in the corresponding positions.
    # Example: M1 = (A00+A11)(B00+B11).
    # U col: [1, 0, 0, 1] (for A00, A01, A10, A11)
    # V col: [1, 0, 0, 1] (for B00, B01, B10, B11)
    # W col: [1, 0, 0, 1] (for C00, C01, C10, C11) ?
    # No, the W column must reconstruct the contribution to C.
    # The term M1 contributes to C00, C11.
    # C00 += M1
    # C11 += M1
    # So W col for M1: [1, 0, 0, 1] (C00, C01, C10, C11).
    
    # Let's verify.
    # Sum over r of U[i,r] V[j,r] W[k,r].
    # If U has 1 at A00 and A11, and V has 1 at B00 and B11,
    # then U[i]V[j] is 1 if (i in {A00,A11} and j in {B00,B11}).
    # This is exactly (A00+A11)(B00+B11).
    # And W[k] adds this to C00 and C11.
    # This works!
    
    # So for each 2x2 block, we add 7 columns.
    
    for block_idx in range(4):
        # Local indices for this block
        # A rows: 1, 2 (global)
        # B cols: 2*block_idx, 2*block_idx+1 (global)
        
        # M1: (A11+A22)(B11+B22) -> C11, C22
        # A11 is global (1,1), A22 is global (2,2)
        # B11 is global (1, 2*block_idx), B22 is global (2, 2*block_idx+1)
        # C11 is global (1, 2*block_idx), C22 is global (2, 2*block_idx+1)
        
        U[1 * m + 1, col] = 1
        U[2 * m + 2, col] = 1
        V[1 * p + 2 * block_idx, col] = 1
        V[2 * p + (2 * block_idx + 1), col] = 1
        W[(2 * block_idx) * n + 1, col] = 1
        W[(2 * block_idx + 1) * n + 2, col] = 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21
        # A21 is (2,1), A22 is (2,2)
        # B11 is (1, 2*block_idx)
        # C21 is (2, 2*block_idx)
        
        U[2 * m + 1, col] = 1
        U[2 * m + 2, col] = 1
        V[1 * p + 2 * block_idx, col] = 1
        W[(2 * block_idx) * n + 2, col] = 1
        col += 1
        
        # M3: A11(B12-B22) -> C12
        # A11 is (1,1)
        # B12 is (1, 2*block_idx+1), B22 is (2, 2*block_idx+1)
        # C12 is (1, 2*block_idx+1)
        
        U[1 * m + 1, col] = 1
        V[1 * p + (2 * block_idx + 1), col] = 1
        V[2 * p + (2 * block_idx + 1), col] = -1
        W[(2 * block_idx + 1) * n + 1, col] = 1
        col += 1
        
        # M4: A22(B21-B11) -> C11, C21
        # A22 is (2,2)
        # B21 is (2, 2*block_idx), B11 is (1, 2*block_idx)
        # C11 is (1, 2*block_idx), C21 is (2, 2*block_idx)
        
        U[2 * m + 2, col] = 1
        V[2 * p + 2 * block_idx, col] = 1
        V[1 * p + 2 * block_idx, col] = -1
        W[(2 * block_idx) * n + 1, col] = 1
        W[(2 * block_idx) * n + 2, col] = 1
        col += 1
        
        # M5: (A11+A12)B22 -> C12, C22
        # A11 is (1,1), A12 is (1,2)
        # B22 is (2, 2*block_idx+1)
        # C12 is (1, 2*block_idx+1), C22 is (2, 2*block_idx+1)
        
        U[1 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[2 * p + (2 * block_idx + 1), col] = 1
        W[(2 * block_idx + 1) * n + 1, col] = 1
        W[(2 * block_idx + 1) * n + 2, col] = 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22
        # A21 is (2,1), A11 is (1,1)
        # B11 is (1, 2*block_idx), B12 is (1, 2*block_idx+1)
        # C22 is (2, 2*block_idx+1)
        
        U[2 * m + 1, col] = 1
        U[1 * m + 1, col] = -1
        V[1 * p + 2 * block_idx, col] = 1
        V[1 * p + (2 * block_idx + 1), col] = 1
        W[(2 * block_idx + 1) * n + 2, col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11, C22
        # A12 is (1,2), A22 is (2,2)
        # B21 is (2, 2*block_idx), B22 is (2, 2*block_idx+1)
        # C11 is (1, 2*block_idx), C22 is (2, 2*block_idx+1)
        
        U[1 * m + 2, col] = 1
        U[2 * m + 2, col] = -1
        V[2 * p + 2 * block_idx, col] = 1
        V[2 * p + (2 * block_idx + 1), col] = 1
        W[(2 * block_idx) * n + 1, col] = 1
        W[(2 * block_idx + 1) * n + 2, col] = 1
        col += 1
    
    # col should be 40 + 28 = 68
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
