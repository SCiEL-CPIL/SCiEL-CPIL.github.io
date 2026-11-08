import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 4+2 block split
    # A = [[A11(4x4), A12(4x2)], [A21(2x4), A22(2x2)]]
    # B = [[B11(4x4), B12(4x2)], [B21(2x4), B22(2x2)]]
    
    # Total Rank:
    # C11 (4x4) = A11 B11 (Strassen 28) + A12 B21 (Trivial 32) = 60
    # C12 (4x2) = A11 B12 (Trivial 32) + A12 B22 (Trivial 16) = 48
    # C21 (2x4) = A21 B11 (Trivial 32) + A22 B21 (Trivial 16) = 48
    # C22 (2x2) = A21 B12 (Trivial 16) + A22 B22 (Strassen 7) = 23
    # Total: 179
    
    total_rank = 179
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add trivial multiplication terms
    def add_trivial(matA_rows, matA_cols, matB_rows, matB_cols, matC_rows, matC_cols):
        nonlocal col
        for i in range(len(matA_rows)):
            for k in range(len(matA_cols)):
                for j in range(len(matB_cols)):
                    # A[i,k] * B[k,j] -> C[i,j]
                    a_idx = matA_rows[i] * m + matA_cols[k]
                    b_idx = matB_rows[k] * p + matB_cols[j]
                    c_idx = matC_cols[j] * n + matC_rows[i]
                    
                    U[a_idx, col] = 1.0
                    V[b_idx, col] = 1.0
                    W[c_idx, col] = 1.0
                    col += 1
                    
    # Helper to add Strassen 2x2 terms
    def add_strassen_2x2(matA_rows, matA_cols, matB_rows, matB_cols, matC_rows, matC_cols):
        nonlocal col
        # matA_rows, matA_cols are lists of 2 indices
        # e.g. [0,1]
        
        # Strassen patterns
        # M0 = (a+d)(e+h)
        # M1 = (c+d)e
        # M2 = a(f-h)
        # M3 = d(g-e)
        # M4 = (a+b)h
        # M5 = (c-a)(e+f)
        # M6 = (b-d)(g+h)
        
        a00, a01 = matA_rows[0], matA_cols[0]
        a10, a11 = matA_rows[1], matA_cols[1]
        # Note: matA_rows is [r0, r1], matA_cols is [c0, c1]
        # A00 is (r0, c0), A01 is (r0, c1), A10 is (r1, c0), A11 is (r1, c1)
        
        b00, b01 = matB_rows[0], matB_cols[0]
        b10, b11 = matB_rows[1], matB_cols[1]
        
        c00, c01 = matC_rows[0], matC_cols[0]
        c10, c11 = matC_rows[1], matC_cols[1]
        
        # M0
        U[matA_rows[0]*m + matA_cols[0], col] += 1
        U[matA_rows[1]*m + matA_cols[1], col] += 1
        V[matB_rows[0]*p + matB_cols[0], col] += 1
        V[matB_rows[1]*p + matB_cols[1], col] += 1
        W[matC_cols[0]*n + matC_rows[0], col] += 1
        W[matC_cols[1]*n + matC_rows[1], col] += 1
        col += 1
        
        # M1
        U[matA_rows[1]*m + matA_cols[0], col] += 1
        U[matA_rows[1]*m + matA_cols[1], col] += 1
        V[matB_rows[0]*p + matB_cols[0], col] += 1
        W[matC_cols[0]*n + matC_rows[1], col] += 1
        W[matC_cols[1]*n + matC_rows[1], col] -= 1
        col += 1
        
        # M2
        U[matA_rows[0]*m + matA_cols[0], col] += 1
        V[matB_rows[0]*p + matB_cols[1], col] += 1
        V[matB_rows[1]*p + matB_cols[1], col] -= 1
        W[matC_cols[1]*n + matC_rows[0], col] += 1
        W[matC_cols[1]*n + matC_rows[1], col] += 1
        col += 1
        
        # M3
        U[matA_rows[1]*m + matA_cols[1], col] += 1
        V[matB_rows[1]*p + matB_cols[0], col] += 1
        V[matB_rows[0]*p + matB_cols[0], col] -= 1
        W[matC_cols[0]*n + matC_rows[0], col] += 1
        W[matC_cols[0]*n + matC_rows[1], col] += 1
        col += 1
        
        # M4
        U[matA_rows[0]*m + matA_cols[0], col] += 1
        U[matA_rows[0]*m + matA_cols[1], col] += 1
        V[matB_rows[1]*p + matB_cols[1], col] += 1
        W[matC_cols[0]*n + matC_rows[0], col] -= 1
        W[matC_cols[1]*n + matC_rows[0], col] += 1
        col += 1
        
        # M5
        U[matA_rows[1]*m + matA_cols[0], col] += 1
        U[matA_rows[0]*m + matA_cols[0], col] -= 1
        V[matB_rows[0]*p + matB_cols[0], col] += 1
        V[matB_rows[0]*p + matB_cols[1], col] += 1
        W[matC_cols[1]*n + matC_rows[1], col] += 1
        col += 1
        
        # M6
        U[matA_rows[0]*m + matA_cols[1], col] += 1
        U[matA_rows[1]*m + matA_cols[1], col] -= 1
        V[matB_rows[1]*p + matB_cols[0], col] += 1
        V[matB_rows[1]*p + matB_cols[1], col] += 1
        W[matC_cols[0]*n + matC_rows[0], col] += 1
        col += 1

    # Define blocks
    # A11: rows 0-3, cols 0-3
    # A12: rows 0-3, cols 4-5
    # A21: rows 4-5, cols 0-3
    # A22: rows 4-5, cols 4-5
    
    # B11: rows 0-3, cols 0-3
    # B12: rows 0-3, cols 4-5
    # B21: rows 4-5, cols 0-3
    # B22: rows 4-5, cols 4-5
    
    # C11: rows 0-3, cols 0-3
    # C12: rows 0-3, cols 4-5
    # C21: rows 4-5, cols 0-3
    # C22: rows 4-5, cols 4-5
    
    # C11 = A11 B11 + A12 B21
    # A11 B11: Strassen 4x4 (via 2x2 blocks)
    # A11 is 4x4. Split into 2x2 blocks.
    # A11_00: rows 0-1, cols 0-1
    # A11_01: rows 0-1, cols 2-3
    # A11_10: rows 2-3, cols 0-1
    # A11_11: rows 2-3, cols 2-3
    
    # We need to apply Strassen to the 4x4 block A11 * B11.
    # This requires decomposing A11 and B11 into 2x2 blocks.
    # A11 = [[A11_00, A11_01], [A11_10, A11_11]]
    # B11 = [[B11_00, B11_01], [B11_10, B11_11]]
    
    # C11 = A11 B11.
    # C11 = [[C11_00, C11_01], [C11_10, C11_11]]
    
    # We can use the 2x2 Strassen on the 2x2 block structure.
    # This effectively gives rank 28 for the 4x4 multiplication.
    
    # Let's call add_strassen_2x2 for each 2x2 sub-block of the 4x4 multiplication.
    # Wait, Strassen for 4x4 using 2x2 blocks means we treat the 4x4 as a 2x2 matrix of 2x2 blocks.
    # The algorithm is the same, but the "scalars" are 2x2 matrices.
    # So we need to apply the Strassen combinations to the 2x2 blocks.
    # This means for each Strassen term (e.g. M0), we multiply the corresponding 2x2 blocks.
    # But we are doing scalar decomposition.
    # So for M0 = (A11_00 + A11_11)(B11_00 + B11_11), we need to decompose the product of sums.
    # (A+B)(C+D) = AC + AD + BC + BD.
    # This is 4 trivial multiplications of 2x2 blocks.
    # Each 2x2 block multiplication has rank 4 (trivial) or 7 (Strassen).
    # If we use trivial for the 2x2 blocks inside Strassen, rank is 4 * 7 = 28.
    # If we use Strassen for the 2x2 blocks, rank is 7 * 7 = 49.
    # So we should use trivial for the inner 2x2 blocks.
    
    # So for each Strassen term (7 terms), we have 4 sub-terms (AC, AD, BC, BD).
    # Each sub-term is a product of two 2x2 blocks.
    # We decompose each 2x2 block product using trivial rank 4.
    # Total 7 * 4 * 4 = 112? No.
    # Strassen term M0 involves sum of blocks.
    # (A11_00 + A11_11) is a 2x2 matrix.
    # (B11_00 + B11_11) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # We need to decompose this product into rank-1 tensors.
    # The rank of 2x2 matmul is 7 (Strassen) or 4 (Trivial).
    # If we use Trivial, rank 4.
    # So for M0, we need to compute (A11_00 + A11_11)(B11_00 + B11_11).
    # This is a single 2x2 matmul.
    # So we just call add_strassen_2x2 for this 2x2 matmul?
    # No, the inputs are sums.
    # U coefficients for A11_00 and A11_11 are both 1.
    # So we can just call add_trivial for the 2x2 blocks?
    # If we use trivial for the 2x2 block multiplication, rank is 4.
    # So M0 contributes 4 to the rank.
    # Total for 4x4 Strassen with trivial inner: 7 * 4 = 28.
    
    # So I will implement the 4x4 multiplication as 7 Strassen terms, each of which is a 2x2 block multiplication done trivially (rank 4).
    
    # Let's refine add_strassen_2x2 to accept block indices.
    
    # I will rewrite the logic to be cleaner.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

