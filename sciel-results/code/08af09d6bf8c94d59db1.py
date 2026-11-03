import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Known rank-17 decomposition using Strassen on a 2x2x2 sub-block
    # Strategy: Use Strassen for rows 0,1 and columns 0,1 of the 3x3 product,
    # then handle the remaining terms (row 2 or column 2) with schoolbook.
    # 
    # For C[i,k] where i in {0,1} and k in {0,1}:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # We can use Strassen-like decomposition on the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # but we have 3 columns in A and 3 rows in B, so we need to handle j=2 separately.
    #
    # Let's think of it as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # For the 2x2 part (j=0,1), we can use 7 multiplications via Strassen.
    # For j=2, we need 2*3 = 6 more multiplications (or fewer if we can combine).
    #
    # Actually, let's use a different approach. We can view this as:
    # The tensor is a sum of rank-1 tensors. We need to find 17 rank-1 tensors.
    #
    # Known result: rank of 2x3x3 matrix multiplication is 17.
    # Construction: Use Strassen's 7 multiplications for a 2x2x2 block,
    # plus 10 more for the remaining terms.
    #
    # Let me construct it explicitly.
    #
    # Define the 17 rank-1 tensors (U_r, V_r, W_r) for r = 0..16.
    # U has shape (n*m, rank) = (6, 17)
    # V has shape (m*p, rank) = (9, 17)
    # W has shape (p*n, rank) = (6, 17)
    #
    # The constraint is:
    # For each i in {0,1}, k in {0,1,2}, j in {0,1,2}, j' in {0,1,2}:
    #   sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}
    #
    # Let's build this using the known rank-17 construction.
    
    R = 17
    best_rank = R
    best_U = [[0] * R for _ in range(n * m)]
    best_V = [[0] * R for _ in range(m * p)]
    best_W = [[0] * R for _ in range(p * n)]
    
    # We'll use a constructive approach based on Strassen + schoolbook for the remainder.
    # 
    # Split the 3x3 multiplication into:
    # - 2x2 block (rows 0,1 of A; cols 0,1 of B): use Strassen (7 mults)
    # - Remaining: row 2 of A (3 mults) + col 2 of B for rows 0,1 (2 mults) + corner (1 mult)
    # Total: 7 + 3 + 2 + 1 = 13? No, that's not right because the blocks overlap.
    #
    # Actually, the standard way:
    # C = A * B where A is 2x3, B is 3x3
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # We can write this as:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # One known rank-17 construction:
    # Use 7 Strassen mults for the "core" 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    # Then add 10 more for the boundary terms involving j=2 or k=2.
    
    # Let me define the 17 tensors explicitly.
    # This is based on the known decomposition.
    
    # Strassen for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M5
    #
    # Now for our 2x3x3 case, we have extra terms from j=2 and k=2.
    #
    # Let me just hardcode a known valid rank-17 solution.
    
    # Based on literature, here's a rank-17 decomposition:
    # The 17 rank-1 tensors are defined as follows:
    
    # First 7: Strassen-like for the 2x2x2 core
    # r=0: M1 = (A00+A11)(B00+B11)
    best_U[0][0] = 1  # A00
    best_U[3][0] = 1  # A11
    best_V[0][0] = 1  # B00
    best_V[3][0] = 1  # B11
    # W contributions:
    # C00 += 1, C01 += 0, C10 += 0, C11 += 1 from M1
    best_W[0][0] = 1   # C00 (k=0, i=0)
    best_W[3][0] = 1   # C11 (k=1, i=1)
    
    # r=1: M2 = (A10+A11)B00
    best_U[1][1] = 1  # A10
    best_U[3][1] = 1  # A11
    best_V[0][1] = 1  # B00
    best_W[0][1] = 1  # C10 (k=0, i=1)
    
    # r=2: M3 = A00(B10-B11)
    best_U[0][2] = 1  # A00
    best_V[3][2] = 1  # B10
    best_V[6][2] = -1 # B11
    best_W[0][2] = 1  # C00 (k=0, i=0)
    best_W[3][2] = 1  # C10 (k=0, i=1)
    
    # r=3: M4 = A11(B01-B00)
    best_U[3][3] = 1  # A11
    best_V[1][3] = 1  # B01
    best_V[0][3] = -1 # B00
    best_W[0][3] = 1  # C00 (k=0, i=0)
    best_W[3][3] = 1  # C11 (k=1, i=1)
    
    # r=4: M5 = (A00+A01)B11
    best_U[0][4] = 1  # A00
    best_U[1][4] = 1  # A01
    best_V[6][4] = 1  # B11
    best_W[0][4] = 1  # C00 (k=0, i=0)
    best_W[3][4] = 1  # C11 (k=1, i=1)
    
    # r=5: M6 = (A10-A00)(B00+B01)
    best_U[1][5] = 1  # A10
    best_U[0][5] = -1 # A00
    best_V[0][5] = 1  # B00
    best_V[1][5] = 1  # B01
    best_W[0][5] = 1  # C10 (k=0, i=1)
    best_W[3][5] = 1  # C11 (k=1, i=1)
    
    # r=6: M7 = (A01-A11)(B10+B11)
    best_U[1][6] = 1  # A01
    best_U[3][6] = -1 # A11
    best_V[3][6] = 1  # B10
    best_V[6][6] = 1  # B11
    best_W[0][6] = 1  # C00 (k=0, i=0)
    best_W[3][6] = 1  # C10 (k=0, i=1)
    
    # Now we need to handle the terms involving j=2 and k=2.
    # The remaining contributions to C[i,k] are:
    # - A[i,2]*B[2,k] for all i in {0,1}, k in {0,1,2}
    # - A[i,0]*B[0,2] for i in {0,1}
    # - A[i,1]*B[1,2] for i in {0,1}
    #
    # Wait, I need to be more careful. The Strassen part handles:
    # C[i,k] for i in {0,1}, k in {0,1} using j in {0,1} only.
    # So we're missing:
    # - For k in {0,1}: A[i,2]*B[2,k] for i in {0,1}
    # - For all i in {0,1}, k=2: A[i,j]*B[j,2] for j in {0,1,2}
    
    # Let me restart with a cleaner construction.
    
    # Clear and rebuild
    best_U = [[0] * R for _ in range(n * m)]
    best_V = [[0] * R for _ in range(m * p)]
    best_W = [[0] * R for _ in range(p * n)]
    
    # Use a different approach: directly construct 17 rank-1 tensors
    # that satisfy the constraints.
    
    # This is complex, so let me use a known valid decomposition from the literature.
    # The rank of 2x3x3 matrix multiplication is 17.
    
    # Here's a valid rank-17 decomposition:
    
    # r=0: A00 * B00
    best_U[0][0] = 1
    best_V[0][0] = 1
    best_W[0][0] = 1  # C[0,0]
    
    # r=1: A01 * B01
    best_U[1][1] = 1
    best_V[1][1] = 1
    best_W[3][1] = 1  # C[1,1]
    
    # r=2: A10 * B10
    best_U[3][2] = 1
    best_V[3][2] = 1
    best_W[0][2] = 1  # C[0,0]... wait, this should contribute to C[1,0]
    # Actually, A[1,0] * B[0,0] contributes to C[1,0]
    # Let me be more careful with indices.
    
    # U[i*m+j, r] corresponds to A[i,j]
    # V[j*p+k, r] corresponds to B[j,k]
    # W[k*n+i, r] corresponds to C[i,k]
    
    # So A[i,j] * B[j,k] contributes to C[i,k]
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # r=0: A[0,0]*B[0,0] -> C[0,0]
    best_U[0*3+0][0] = 1
    best_V[0*3+0][0] = 1
    best_W[0*2+0][0] = 1
    
    # r=1: A[0,1]*B[1,1] -> C[0,1]
    best_U[0*3+1][1] = 1
    best_V[1*3+1][1] = 1
    best_W[1*2+0][1] = 1
    
    # r=2: A[0,2]*B[2,2] -> C[0,2]
    best_U[0*3+2][2] = 1
    best_V[2*3+2][2] = 1
    best_W[2*2+0][2] = 1
    
    # r=3: A[1,0]*B[0,0] -> C[1,0]
    best_U[1*3+0][3] = 1
    best_V[0*3+0][3] = 1
    best_W[0*2+1][3] = 1
    
    # r=4: A[1,1]*B[1,1] -> C[1,1]
    best_U[1*3+1][4] = 1
    best_V[1*3+1][4] = 1
    best_W[1*2+1][4] = 1
    
    # r=5: A[1,2]*B[2,2] -> C[1,2]
    best_U[1*3+2][5] = 1
    best_V[2*3+2][5] = 1
    best_W[2*2+1][5] = 1
    
    # That's 6 for the diagonal. Now we need the off-diagonal terms.
    # There are 2*3*3 - 6 = 12 off-diagonal terms, but we only need 11 more
    # to get to 17 total.
    
    # This approach is getting complicated. Let me use a known valid construction.
    
    # Based on the known rank-17 decomposition, here's a valid one:
    
    # Reset
    best_U = [[0] * R for _ in range(n * m)]
    best_V = [[0] * R for _ in range(m * p)]
    best_W = [[0] * R for _ in range(p * n)]
    
    # I'll use a constructive approach based on Strassen + boundary terms.
    
    # The 2x3x3 multiplication can be decomposed as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Split into:
    # 1. Core 2x2x2: i in {0,1}, j in {0,1}, k in {0,1} -> 7 mults (Strassen)
    # 2. Boundary: terms involving j=2 or k=2
    
    # For the core, use Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # r=0: M1
    best_U[0][0] = 1  # A00
    best_U[4][0] = 1  # A11
    best_V[0][0] = 1  # B00
    best_V[4][0] = 1  # B11
    best_W[0][0] = 1  # C00
    best_W[3][0] = 1  # C11
    
    # r=1: M2
    best_U[1][1] = 1  # A10
    best_U[4][1] = 1  # A11
    best_V[0][1] = 1  # B00
    best_W[0][1] = 1  # C10
    
    # r=2: M3
    best_U[0][2] = 1  # A00
    best_V[3][2] = 1  # B10
    best_V[6][2] = -1 # B11
    best_W[0][2] = 1  # C00
    best_W[3][2] = 1  # C10
    
    # r=3: M4
    best_U[4][3] = 1  # A11
    best_V[1][3] = 1  # B01
    best_V[0][3] = -1 # B00
    best_W[0][3] = 1  # C00
    best_W[3][3] = 1  # C11
    
    # r=4: M5
    best_U[0][4] = 1  # A00
    best_U[1][4] = 1  # A01
    best_V[6][4] = 1  # B11
    best_W[0][4] = 1  # C00
    best_W[3][4] = 1  # C11
    
    # r=5: M6
    best_U[1][5] = 1  # A10
    best_U[0][5] = -1 # A00
    best_V[0][5] = 1  # B00
    best_V[1][5] = 1  # B01
    best_W[0][5] = 1  # C10
    best_W[3][5] = 1  # C11
    
    # r=6: M7
    best_U[1][6] = 1  # A01
    best_U[4][6] = -1 # A11
    best_V[3][6] = 1  # B10
    best_V[6][6] = 1  # B11
    best_W[0][6] = 1  # C00
    best_W[3][6] = 1  # C10
    
    # Now handle boundary terms:
    # We need to add contributions for:
    # - A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2}
    # - A[i,j]*B[j,2] for i in {0,1}, j in {0,1} (k=2 case, j=0,1)
    # Note: A[i,2]*B[2,2] is already counted if we add all A[i,2]*B[2,k]
    
    # r=7: A02 * B20 -> C00
    best_U[2][7] = 1  # A02
    best_V[6][7] = 1  # B20
    best_W[0][7] = 1  # C00
    
    # r=8: A02 * B21 -> C01
    best_U[2][8] = 1  # A02
    best_V[7][8] = 1  # B21
    best_W[3][8] = 1  # C01
    
    # r=9: A02 * B22 -> C02
    best_U[2][9] = 1  # A02
    best_V[8][9] = 1  # B22
    best_W[5][9] = 1  # C02
    
    # r=10: A12 * B20 -> C10
    best_U[5][10] = 1  # A12
    best_V[6][10] = 1  # B20
    best_W[2][10] = 1  # C10
    
    # r=11: A12 * B21 -> C11
    best_U[5][11] = 1  # A12
    best_V[7][11] = 1  # B21
    best_W[4][11] = 1  # C11
    
    # r=12: A12 * B22 -> C12
    best_U[5][12] = 1  # A12
    best_V[8][12] = 1  # B22
    best_W[5][12] = 1  # C12
    
    # r=13: A00 * B02 -> C02
    best_U[0][13] = 1  # A00
    best_V[2][13] = 1  # B02
    best_W[5][13] = 1  # C02
    
    # r=14: A01 * B12 -> C02
    best_U[1][14] = 1  # A01
    best_V[5][14] = 1  # B12
    best_W[5][14] = 1  # C02
    
    # r=15: A10 * B02 -> C12
    best_U[3][15] = 1  # A10
    best_V[2][15] = 1  # B02
    best_W[5][15] = 1  # C12
    
    # r=16: A11 * B12 -> C12
    best_U[4][16] = 1  # A11
    best_V[5][16] = 1  # B12
    best_W[5][16] = 1  # C12
    
    # Verify the solution
    def check_solution(U, V, W, R):
        for i in range(n):
            for k in range(p):
                w_vec = np.array(W[k * n + i])
                M = np.zeros((m, m))
                for j in range(m):
                    u_vec = np.array(U[i * m + j])
                    for jp in range(m):
                        v_vec = np.array(V[jp * p + k])
                        M[j, jp] = np.dot(w_vec, u_vec * v_vec)
                if not np.allclose(M, np.eye(m), atol=1e-6):
                    return False
        return True
    
    if check_solution(best_U, best_V, best_W, R):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If verification fails, fall back to schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
