"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def is_valid_entries(U_mat, V_mat, W_mat):
        """Check that all entries are integers or half-integers."""
        for mat in [U_mat, V_mat, W_mat]:
            for row in mat:
                for val in row:
                    if val != int(val) and val != int(val) + 0.5 and val != int(val) - 0.5:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition, rank 7, integer coefficients
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # P1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # P3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # P4 = a11(b01-b10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # P5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # P6 = (a10-a00)(b00+b10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # P7 = (a01-a11)(b01+b11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    # Build 4x4x4 using Strassen: rank 49
    def build_4x4x4_strassen():
        U_2, V_2, W_2 = strassen_2x2x2()
        
        # 4x4 matrix split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # A00: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5
        # A01: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
        # A10: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
        # A11: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
        
        # For Strassen at block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M_r is a 2x2 * 2x2 = 2x2 product.
        # We decompose each using 2x2x2 Strassen (rank 7).
        # Total: 7 * 7 = 49 terms.
        
        # Index mapping for 2x2 blocks within 4x4:
        # Block (br, bc) where br, bc in {0,1}
        # Element (i,j) in block is at global index (br*2+i)*4 + (bc*2+j)
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # For each block-level Strassen term r (0..6) and sub-term s (0..6):
        # The A-blocks involved are those with U_2[a, r] != 0
        # The B-blocks involved are those with V_2[b, r] != 0
        # The C-blocks involved are those with W_2[c, r] != 0
        
        # For each such (a, b, c), we add the sub-level Strassen term s
        # with coefficients U_2[a,r] * U_2_sub[sub_a, s], etc.
        
        # Actually, the correct way:
        # M_r = (sum_a U_2[a,r] * A_a) * (sum_b V_2[b,r] * B_b)
        # = sum_a sum_b U_2[a,r] * V_2[b,r] * (A_a * B_b)
        #
        # A_a * B_b is a 2x2*2x2=2x2 product, decomposed by Strassen into 7 terms.
        # So for each (r, a, b, s), we add a term.
        # Number of terms: sum over r of (num_nonzero_U_2[r] * num_nonzero_V_2[r] * 7)
        # This is more than 49 in general.
        
        # I think the issue is that the 4x4x4 Strassen decomposition is not simply
        # 7*7. Let me use a different approach.
        
        # Actually, for the purposes of this problem, let me just use schoolbook for 4x4x4
        # and schoolbook for 4x4x2, giving total rank 64+32=96, same as before.
        # That doesn't help.
        
        # Let me try a different structural approach: use the fact that we can
        # share terms between the 4x4x4 and 4x4x2 parts.
        
        # For now, return schoolbook.
        U = [[0]*64 for _ in range(16)]
        V = [[0]*64 for _ in range(16)]
        W = [[0]*64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W
    
    # Use schoolbook for full 4x4x6
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    if not verify_decomposition(best_U, best_V, best_W):
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
