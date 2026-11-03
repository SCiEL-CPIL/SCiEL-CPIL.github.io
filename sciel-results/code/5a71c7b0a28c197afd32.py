import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 135

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_decomp(U, V, W, tol=1e-8):
        # Check all (i,j,k) entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Ur = U[i * m + j, :]
                    Vr = V[j * p + k, :]
                    Wr = W[k * n + i, :]
                    val = np.dot(Ur * Vr, Wr)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_strassen_2x2():
        """
        Strassen 2x2x2 rank-7 decomposition.
        U shape (4,7): U[i*2+j, t] for A[i,j]
        V shape (4,7): V[j*2+k, t] for B[j,k]
        W shape (4,7): W[k*2+i, t] for C[k,i] (note: W index is k*n+i = k*2+i)
        
        Wait, the convention is W[k*n + i, r] where C[i,k] is the output.
        So W index for C[i,k] is k*n + i. With n=2, that's k*2 + i.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
        # A[0,0] -> U[0*2+0] = U[0], A[1,1] -> U[1*2+1] = U[3]
        # B[1,0] -> V[1*2+0] = V[2], B[1,1] -> V[1*2+1] = V[3]
        # C[0,0] -> W[0*2+0] = W[0], C[1,1] -> W[1*2+1] = W[3]
        U[0, 0] = 1; U[3, 0] = -1
        V[2, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        # A[1,0] -> U[1*2+0] = U[2], A[1,1] -> U[3]
        # B[0,0] -> V[0*2+0] = V[0]
        # C[1,0] -> W[0*2+1] = W[1]  (k=0, i=1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        # A[0,0] -> U[0]
        # B[0,1] -> V[0*2+1] = V[1], B[1,1] -> V[3]
        # C[0,1] -> W[1*2+0] = W[2] (k=1, i=0), C[1,1] -> W[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = -1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        # A[1,1] -> U[3]
        # B[1,0] -> V[2], B[0,0] -> V[0]
        # C[0,0] -> W[0], C[1,0] -> W[1]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # A[0,0] -> U[0], A[1,1] -> U[3]
        # B[0,0] -> V[0], B[1,1] -> V[3]
        # C[0,0] -> W[0] (coeff -1), C[0,1] -> W[2] (coeff 1), C[1,1] -> W[3] (coeff 1)
        U[0, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1; V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1; W[3, 4] = 1
        
        # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # A[0,1] -> U[0*2+1] = U[1], A[1,1] -> U[3]
        # B[1,0] -> V[2], B[1,1] -> V[3]
        # C[1,1] -> W[3]
        U[1, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # A[1,0] -> U[2], A[0,0] -> U[0]
        # B[0,0] -> V[0], B[0,1] -> V[1]
        # C[0,0] -> W[0]
        U[2, 6] = 1; U[0, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def make_strassen_4x4():
        """
        Build 4x4x4 rank-25 decomposition from 2x2 Strassen blocks with sharing.
        The 4x4 matrix is divided into four 2x2 blocks.
        A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
        B = [[B11, B12], [B21, B22]]
        C = A*B where Ci = sum over blocks.
        
        Standard 4x4x4 rank 25 construction:
        We can use 7 Strassen terms for each of the 4 block-products, 
        but share some. Actually, a cleaner way:
        
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
        
        Using Strassen on the block level (7 terms for 4x4):
        M1 = (A11-A22)(B21+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A22)(B11+B22)
        M6 = (A12-A22)(B21+B22)
        M7 = (A21-A11)(B11+B12)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M3 + M5 + M6
        
        But each Mi is a 2x2 matrix product, which itself needs 7 terms (or 4 for schoolbook).
        If we use schoolbook (4 terms) for each block product:
        Total = 7 * 4 = 28 terms. Not 25.
        
        If we use Strassen (7 terms) for each block product:
        Total = 7 * 7 = 49 terms. Too many.
        
        The rank-25 for 4x4x4 comes from a more careful construction. 
        One known approach: use 25 terms by combining.
        
        Let me use a different approach. I'll build the 4x4x4 decomposition 
        by treating it as a tensor and using a known rank-25 construction.
        
        Actually, let me just use the simple tiling: 4 block products, each with Strassen rank 7.
        But that's 28. To get 25, we need to share 3 terms.
        
        Alternatively, I can just use the 28-rank construction for 4x4x4,
        then embed 2x4x8 into it.
        
        4x4x4 with rank 28:
        For C11 = A11*B11 + A12*B21:
          A11*B11: 7 Strassen terms
          A12*B21: 7 Strassen terms
        Total for C11: 14
        Similarly C12, C21, C22 each 14.
        Total: 56. That's the same as before!
        
        Wait, I need to be more careful. Let me think about the dimensions.
        
        For 2x4x8:
        A is 2x4, B is 4x8, C is 2x8.
        
        I can split A into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
        B into two 2x8 blocks: B_top (rows 0-1), B_bot (rows 2-3)
        C = A_left * B_top + A_right * B_bot
        
        A_left * B_top: 2x2 * 2x8 = 2x8
        A_right * B_bot: 2x2 * 2x8 = 2x8
        
        For 2x2 * 2x8:
        Split B into four 2x2 blocks: B0, B1, B2, B3 (cols 0-1, 2-3, 4-5, 6-7)
        C_block = A * B0 + A * B1 + A * B2 + A * B3
        Each A * Bi is 2x2 * 2x2, using Strassen with 7 terms.
        Total for one 2x2*2x8: 4*7 = 28 terms.
        Two such products: 56 terms total.
        
        To do better, I need to share terms between the two 2x2*2x8 products.
        
        Alternative: Use the 4x4x4 rank-25 construction.
        
        Let me construct a 4x4x4 rank-25 decomposition explicitly.
        
        I'll use the following known construction for 4x4x4:
        Split into 2x2 blocks. Use the following 25 terms:
        
        Actually, let me just hardcode a verified rank-25 construction.
        I'll use the approach from the literature where 4x4x4 has rank 25.
        
        For now, let me use the 56-rank Strassen tiling as a baseline,
        and try to find a better construction.
        """
        # For simplicity and correctness, let me just use the block-Strassen approach
        # which gives rank 56, and see if I can do better.
        
        # Actually, let me try a different tiling.
        # A is 2x4, B is 4x8.
        # I can split B into four 4x2 blocks.
        # Then C = sum of A * B_j for j=0..3, each A*B_j is 2x2.
        # Each 2x2 product uses 7 Strassen terms.
        # Total: 4*7 = 28 terms.
        
        # That's better! Let me implement this.
        
        U_s, V_s, W_s = make_strassen_2x2()
        
        rank_total = 28
        U_full = np.zeros((U_dim, rank_total))
        V_full = np.zeros((V_dim, rank_total))
        W_full = np.zeros((W_dim, rank_total))
        
        col_offset = 0
        for b_block in range(4):  # 4 blocks of 2 columns in B
            b_col_base = b_block * 2
            for t in range(7):
                # U: A[i, j] for i in {0,1}, j in {0,1,2,3}
                # Strassen U_block has entries for j in {0,1} (2x2 block)
                # But A is 2x4, so j ranges 0..3.
                # For this tiling, we use the full A (2x4) with each 2x2 block of B.
                # Wait, that doesn't work directly with Strassen which expects 2x2 A.
                
                # Let me reconsider. Strassen works for 2x2 * 2x2.
                # A is 2x4, so I can't directly use Strassen with the full A.
                
                # I need to split A into 2x2 blocks too.
                # A = [A0 | A1] where A0, A1 are 2x2.
                # B = [B0; B1] where B0, B1 are 2x8.
                # C = A0*B0 + A1*B1
                # A0*B0: 2x2 * 2x8, split B0 into 2x2 blocks: 4 products, each 7 terms = 28
                # A1*B1: 2x2 * 2x8, similarly 28
                # Total: 56.
                
                # To get 28, I need a different approach.
                pass
        
        # Let me just go with the 56-rank construction for now,
        # but implement it correctly.
        
        rank_total = 56
        U_full = np.zeros((U_dim, rank_total))
        V_full = np.zeros((V_dim, rank_total))
        W_full = np.zeros((W_dim, rank_total))
        
        col_offset = 0
        for a_block in range(2):  # A split into 2x2 blocks
            a_col_base = a_block * 2
            b_row_base = a_block * 2  # B split into 2x8 blocks
            for c_block in range(4):  # B 2x8 split into 2x2 blocks
                c_col_base = c_block * 2
                for t in range(7):
                    # U: A[i, a_col_base + j]
                    for i in range(2):
                        for j in range(2):
                            U_full[i * 4 + a_col_base + j, col_offset] += U_s[i * 2 + j, t]
                    # V: B[b_row_base + j, c_col_base + k]
                    for j in range(2):
                        for k in range(2):
                            V_full[(b_row_base + j) * 8 + c_col_base + k, col_offset] += V_s[j * 2 + k, t]
                    # W: C[i, c_col_base + k] -> W[(c_col_base+k)*2 + i]
                    for i in range(2):
                        for k in range(2):
                            W_full[(c_col_base + k) * 2 + i, col_offset] += W_s[k * 2 + i, t]
                    col_offset += 1
        
        if check_decomp(U_full, V_full, W_full, tol=1e-10):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
        # Fallback
        U0, V0, W0, r0 = make_schoolbook()
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    U0, V0, W0, r0 = make_schoolbook()
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
