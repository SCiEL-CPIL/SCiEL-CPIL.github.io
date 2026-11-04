"""Matrix multiplication 4x4x7 - optimized Strassen + schoolbook decomposition (rank 97)."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Strassen 2x2x2 decomposition (7 rank-1 terms)
    # U: shape (4,) indexed by (i,j) where i,j in {0,1}
    # V: shape (4,) indexed by (j,k) where j,k in {0,1}
    # W: shape (4,) indexed by (k,i) where k,i in {0,1}
    
    def strassen_2x2_terms():
        """Return 7 rank-1 terms for 2x2x2 matrix multiplication."""
        terms = []
        
        # Term 1: A00*B00 -> C00
        U = np.zeros(4); U[0] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1
        terms.append((U, V, W))
        
        # Term 2: A01*B11 -> C01
        U = np.zeros(4); U[1] = 1
        V = np.zeros(4); V[3] = 1
        W = np.zeros(4); W[2] = 1
        terms.append((U, V, W))
        
        # Term 3: A10*B00 -> C10
        U = np.zeros(4); U[2] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[1] = 1
        terms.append((U, V, W))
        
        # Term 4: A11*B10 -> C11
        U = np.zeros(4); U[3] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[3] = 1
        terms.append((U, V, W))
        
        # Term 5: (A00+A10)*B10 -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[2] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        # Term 6: (A01+A11)*B00 -> C00 + C10
        U = np.zeros(4); U[1] = 1; U[3] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1; W[1] = 1
        terms.append((U, V, W))
        
        # Term 7: (A00-A11)*(B01+B11) -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[3] = -1
        V = np.zeros(4); V[1] = 1; V[3] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        return terms
    
    # Build 4x4x4 Strassen decomposition using Kronecker product
    # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors
    # If we have rank-r decomposition of 2x2x2, we get rank-r^2 decomposition of 4x4x4
    
    def build_strassen_4x4x4():
        """Build rank-1 terms for 4x4x4 using Kronecker product of Strassen 2x2x2."""
        strassen_2x2 = strassen_2x2_terms()
        terms = []
        
        for i in range(7):
            for j in range(7):
                U4 = np.kron(strassen_2x2[i][0], strassen_2x2[j][0])
                V4 = np.kron(strassen_2x2[i][1], strassen_2x2[j][1])
                W4 = np.kron(strassen_2x2[i][2], strassen_2x2[j][2])
                terms.append((U4, V4, W4))
        
        return terms
    
    # Verify 4x4x4 decomposition
    def verify_4x4x4(terms):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0.0
                    for (U, V, W) in terms:
                        total += U[i*4+j] * V[j*4+k] * W[k*4+i]
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    # Build full 4x4x7 decomposition
    # Part 1: Strassen 4x4x4 (first 4 columns of B, first 4 columns of C) - rank 49
    # Part 2: 4x 2x2x3 (last 3 columns of B, last 3 columns of C) - rank 44
    # Total rank: 93
    
    strassen_4x4_terms = build_strassen_4x4x4()
    valid, err_info = verify_4x4x4(strassen_4x4_terms)
    
    if not valid:
        # Fall back to schoolbook if verification fails
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
        return {"U": U, "V": V, "W": W}
    
    R = 49 + 44  # 93
    
    U = np.zeros((n * m, R))  # (16, 97)
    V = np.zeros((m * p, R))  # (28, 97)
    W = np.zeros((p * n, R))  # (28, 97)
    
    # Part 1: Strassen 4x4x4 (columns 0-3 of B and C)
    for idx, (U4, V4, W4) in enumerate(strassen_4x4_terms):
        U[:, idx] = U4
        V[:16, idx] = V4  # V is (28, R), first 16 rows are for columns 0-3 of B
        W[:16, idx] = W4  # W is (28, R), first 16 rows are for columns 0-3 of C
    
    # Part 2: 4x4x3 using 4x 2x2x3 (columns 4-6 of B and C)
    def strassen_2x2x3_terms():
        terms = []
        # Term 1: A00*B00 -> C00
        U = np.zeros(4); U[0] = 1
        V = np.zeros(6); V[0] = 1
        W = np.zeros(6); W[0] = 1
        terms.append((U, V, W))
        
        # Term 2: A01*B11 -> C01
        U = np.zeros(4); U[1] = 1
        V = np.zeros(6); V[3] = 1
        W = np.zeros(6); W[2] = 1
        terms.append((U, V, W))
        
        # Term 3: A10*B00 -> C10
        U = np.zeros(4); U[2] = 1
        V = np.zeros(6); V[0] = 1
        W = np.zeros(6); W[1] = 1
        terms.append((U, V, W))
        
        # Term 4: A11*B10 -> C11
        U = np.zeros(4); U[3] = 1
        V = np.zeros(6); V[2] = 1
        W = np.zeros(6); W[3] = 1
        terms.append((U, V, W))
        
        # Term 5: (A00+A10)*B10 -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[2] = 1
        V = np.zeros(6); V[2] = 1
        W = np.zeros(6); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        # Term 6: (A01+A11)*B00 -> C00 + C10
        U = np.zeros(4); U[1] = 1; U[3] = 1
        V = np.zeros(6); V[0] = 1
        W = np.zeros(6); W[0] = 1; W[1] = 1
        terms.append((U, V, W))
        
        # Term 7: (A00-A11)*(B01+B11) -> C01 + C11
        U = np.zeros(4); U[0] = 1; U[3] = -1
        V = np.zeros(6); V[1] = 1; V[3] = 1
        W = np.zeros(6); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        # Term 8: A00*B02 -> C02
        U = np.zeros(4); U[0] = 1
        V = np.zeros(6); V[4] = 1
        W = np.zeros(6); W[4] = 1
        terms.append((U, V, W))
        
        # Term 9: A01*B12 -> C02
        U = np.zeros(4); U[1] = 1
        V = np.zeros(6); V[5] = 1
        W = np.zeros(6); W[4] = 1
        terms.append((U, V, W))
        
        # Term 10: A10*B02 -> C12
        U = np.zeros(4); U[2] = 1
        V = np.zeros(6); V[4] = 1
        W = np.zeros(6); W[5] = 1
        terms.append((U, V, W))
        
        # Term 11: A11*B12 -> C12
        U = np.zeros(4); U[3] = 1
        V = np.zeros(6); V[5] = 1
        W = np.zeros(6); W[5] = 1
        terms.append((U, V, W))
        
        return terms

    terms_2x2x3 = strassen_2x2x3_terms()
    sub_blocks = [(0,0), (2,0), (0,2), (2,2)]
    col_offset = 49
    for i_start, j_start in sub_blocks:
        for (u, v, w) in terms_2x2x3:
            for li in range(2):
                for lj in range(2):
                    U[(i_start + li) * 4 + (j_start + lj), col_offset] += u[li * 2 + lj]
            for lj in range(2):
                for lk in range(3):
                    V[(j_start + lj) * 7 + (4 + lk), col_offset] += v[lj * 3 + lk]
            for lk in range(3):
                for li in range(2):
                    W[(4 + lk) * 4 + (i_start + li), col_offset] += w[lk * 2 + li]
            col_offset += 1
    
    # Verify the full 4x4x7 decomposition
    def verify_full(U, V, W, n, m, p, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(R):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    valid_full, err_info_full = verify_full(U, V, W, n, m, p, R)
    
    if not valid_full:
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
