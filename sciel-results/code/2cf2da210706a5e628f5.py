"""Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 145s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 60 as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Strategy: Split B (3x10) into five 3x2 blocks B_0..B_4.
    # C = A * B = A * [B_0 B_1 B_2 B_3 B_4] = [A*B_0, A*B_1, ..., A*B_4]
    # Each A*B_t is a 2x2 matrix multiplication.
    # We use Strassen's algorithm for each 2x2 block to get rank 7 per block.
    # Total rank = 5 * 7 = 35.
    
    # Strassen's algorithm for 2x2 matrix multiplication C = A * B:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indexing:
    # U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    #   U[i*m + j] corresponds to A[i][j]
    # V has shape (m*p, rank) = (3*10, rank) = (30, rank)
    #   V[j*p + k] corresponds to B[j][k]
    # W has shape (p*n, rank) = (10*2, rank) = (20, rank)
    #   W[k*n + i] corresponds to C[i][k]
    
    # For block t (0..4), B_t consists of columns 2t and 2t+1 of B.
    # C_t = A * B_t is a 2x2 matrix with entries:
    #   C_t[0][0] = C[0][2t]   -> W index (2t)*n + 0 = 2t
    #   C_t[0][1] = C[0][2t+1] -> W index (2t+1)*n + 0 = 2t+1
    #   C_t[1][0] = C[1][2t]   -> W index (2t)*n + 1 = 2t+1
    #   C_t[1][1] = C[1][2t+1] -> W index (2t+1)*n + 1 = 2t+2
    
    # Wait, W index is k*n + i where i is row, k is column of C.
    # C is n x p = 2 x 10.
    # W[k*n + i] where k in [0,10), i in [0,2).
    # So C[i][k] corresponds to W[k*n + i].
    
    # For block t:
    # C[0][2t]   -> W[2t*2 + 0] = W[4t]
    # C[0][2t+1] -> W[(2t+1)*2 + 0] = W[4t+2]
    # C[1][2t]   -> W[2t*2 + 1] = W[4t+1]
    # C[1][2t+1] -> W[(2t+1)*2 + 1] = W[4t+3]
    
    # A indices in U:
    # A[0][0] -> U[0*3+0] = U[0]
    # A[0][1] -> U[0*3+1] = U[1]
    # A[0][2] -> U[0*3+2] = U[2]
    # A[1][0] -> U[1*3+0] = U[3]
    # A[1][1] -> U[1*3+1] = U[4]
    # A[1][2] -> U[1*3+2] = U[5]
    
    # For block t, B_t is a 2x2 matrix:
    # B_t[0][0] = B[0][2t]   -> V[0*10+2t] = V[2t]
    # B_t[0][1] = B[0][2t+1] -> V[0*10+2t+1] = V[2t+1]
    # B_t[1][0] = B[1][2t]   -> V[1*10+2t] = V[10+2t]
    # B_t[1][1] = B[1][2t+1] -> V[1*10+2t+1] = V[10+2t+1]
    # B_t[2][0] = B[2][2t]   -> V[2*10+2t] = V[20+2t]
    # B_t[2][1] = B[2][2t+1] -> V[2*10+2t+1] = V[20+2t+1]
    
    # Strassen's algorithm operates on 2x2 matrices. But A is 2x3 and B_t is 3x2.
    # So C_t = A * B_t is 2x2. The multiplication is (2x3) * (3x2) -> 2x2.
    # This is not a 2x2 * 2x2 multiplication, it's 2x3 * 3x2.
    # Strassen's algorithm doesn't directly apply to non-square or non-2x2 matrices.
    # We need a rank-7 decomposition for 2x3 * 3x2 matrix multiplication.
    
    # Let's find the rank of 2x3 * 3x2 matrix multiplication.
    # The tensor is T_{i,j,k,l} where i in {0,1}, j in {0,1,2}, k in {0,1,2}, l in {0,1}
    # T_{i,j,k,l} = 1 if the indices are consistent, i.e., sum_j A_{ij} B_{jk} = C_{il}
    # Wait, B_t is 3x2, so B_t[j][l] where j in {0,1,2}, l in {0,1}.
    # C_t[i][l] = sum_{j=0}^{2} A[i][j] * B_t[j][l]
    # This is a bilinear map from (R^2 x R^3) x (R^3 x R^2) -> R^2 x R^2.
    # The tensor rank of 2x3 * 3x2 multiplication.
    
    # For m x n * n x p, the rank is known to be at most mn + np + mp - 1 for some cases,
    # but for 2x3 * 3x2, let's compute.
    # Actually, the rank of 2x3 * 3x2 is known to be 11.
    # Let's verify: the schoolbook rank is 2*3*2 = 12.
    # Can we do it in 11? 
    
    # Let's try to find a rank-11 decomposition for 2x3 * 3x2.
    # This is a small enough problem that we can use a known result or construct it.
    # The rank of 2x2 * 2x2 is 7 (Strassen).
    # The rank of 2x3 * 3x2 can be related to 2x2 * 2x2 by padding or splitting.
    
    # Actually, let's just use the schoolbook rank 12 for each block, giving total rank 5*12 = 60.
    # That's the same as the fallback. Not an improvement.
    
    # Let's try to find a better decomposition.
    # For 2x3 * 3x2, we can think of it as:
    # C = A * B where A is 2x3, B is 3x2.
    # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
    # C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]
    # C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
    
    # This is equivalent to two 2x3 * 3x1 multiplications (for C[:,0] and C[:,1]).
    # The rank of 2x3 * 3x1 is 6 (schoolbook), and it's known that the rank is 6.
    # So the rank of 2x3 * 3x2 is at most 12.
    
    # Can we do better? The rank of 2x3 * 3x2 is known to be 11 in some literature.
    # Let's try to construct a rank-11 decomposition.
    
    # Actually, let's just use a random search with a small rank to see if we can find something.
    # But the previous random search didn't find anything below 60.
    
    # Let's try a different approach: use the known rank-7 decomposition for 2x2 * 2x2
    # and extend it to 2x3 * 3x2.
    
    # Split A into A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    # Split B into B = [B1; B2] where B1 is 2x2 and B2 is 1x2.
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x2 * 2x2, rank 7.
    # A2*B2 is 2x1 * 1x2, rank 2 (outer product).
    # Total rank = 7 + 2 = 9.
    
    # Let's verify this.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    
    # A1 = [[a00, a01], [a10, a11]], B1 = [[b00, b01], [b10, b11]]
    # A2 = [[a02], [a12]], B2 = [[b20, b21]]
    
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x2, A2*B2 is 2x2 (outer product of 2x1 and 1x2).
    
    # So the rank of 2x3 * 3x2 is at most 7 + 2 = 9.
    # Total rank for 5 blocks: 5 * 9 = 45.
    
    # Let's construct this decomposition.
    
    rank_per_block = 9
    total_rank = 5 * rank_per_block  # 45
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen's algorithm for A1*B1 (2x2 * 2x2):
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Outer product for A2*B2 (2x1 * 1x2):
    # M8 = a02 * b20 -> C[0][0]
    # M9 = a02 * b21 -> C[0][1]
    # Wait, A2*B2 is [[a02*b20, a02*b21], [a12*b20, a12*b21]]
    # So:
    # C[0][0] += a02*b20
    # C[0][1] += a02*b21
    # C[1][0] += a12*b20
    # C[1][1] += a12*b21
    
    # We need 4 terms for the outer product, but we can combine:
    # M8 = a02 * b20 -> contributes to C[0][0] and C[1][0]? No, a02 is only in row 0.
    # Actually, A2*B2 = [[a02*b20, a02*b21], [a12*b20, a12*b21]]
    # This requires 4 rank-1 terms:
    # a02*b20, a02*b21, a12*b20, a12*b21
    
    # But we can use 2 terms if we allow the W vectors to have multiple non-zero entries.
    # M8 = (a02, a12) * (b20, 0) -> no, that's not a rank-1 term.
    # A rank-1 term is u * v * w where u is a vector in A-space, v in B-space, w in C-space.
    # For A2*B2, the A-space is spanned by e_a02 and e_a12, B-space by e_b20 and e_b21.
    # We need 4 rank-1 terms for the outer product of two 2-vectors.
    
    # So total rank per block = 7 (Strassen) + 4 (outer product) = 11.
    # Total rank = 5 * 11 = 55.
    
    # Let's use rank 11 per block.
    
    rank_per_block = 11
    total_rank = 5 * rank_per_block  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for t in range(5):
        base = t * rank_per_block
        
        # Strassen terms (7 terms)
        # M1 = (a00 + a11) * (b00 + b11)
        U[0][base+0] += 1  # a00
        U[4][base+0] += 1  # a11
        V[2*t][base+0] += 1    # b00
        V[10+2*t][base+0] += 1  # b11
        W[4*t][base+0] += 1    # C[0][2t]
        W[4*t+2][base+0] += 1  # C[0][2t+1]
        W[4*t+1][base+0] += 1  # C[1][2t]
        W[4*t+3][base+0] += 1  # C[1][2t+1]
        
        # M2 = (a10 + a11) * b00
        U[3][base+1] += 1  # a10
        U[4][base+1] += 1  # a11
        V[2*t][base+1] += 1    # b00
        W[4*t+1][base+1] += 1  # C[1][2t]
        W[4*t+3][base+1] += -1 # C[1][2t+1]
        
        # M3 = a00 * (b01 - b11)
        U[0][base+2] += 1  # a00
        V[2*t+1][base+2] += 1    # b01
        V[10+2*t][base+2] += -1  # b11
        W[4*t+2][base+2] += 1  # C[0][2t+1]
        W[4*t+3][base+2] += 1  # C[1][2t+1]
        
        # M4 = a11 * (b10 - b00)
        U[4][base+3] += 1  # a11
        V[10+2*t][base+3] += 1  # b10
        V[2*t][base+3] += -1    # b00
        W[4*t][base+3] += 1    # C[0][2t]
        W[4*t+1][base+3] += 1  # C[1][2t]
        
        # M5 = (a00 + a01) * b11
        U[0][base+4] += 1  # a00
        U[1][base+4] += 1  # a01
        V[10+2*t][base+4] += 1  # b11
        W[4*t][base+4] += -1    # C[0][2t]
        W[4*t+2][base+4] += 1  # C[0][2t+1]
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[3][base+5] += 1  # a10
        U[0][base+5] += -1  # a00
        V[2*t][base+5] += 1    # b00
        V[2*t+1][base+5] += 1  # b01
        W[4*t+3][base+5] += 1  # C[1][2t+1]
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1][base+6] += 1  # a01
        U[4][base+6] += -1  # a11
        V[10+2*t][base+6] += 1  # b10
        V[10+2*t+1][base+6] += 1  # b11
        W[4*t][base+6] += 1    # C[0][2t]
        W[4*t+3][base+6] += 1  # C[1][2t+1]
        
        # Outer product terms (4 terms)
        # M8 = a02 * b20 -> C[0][2t]
        U[2][base+7] += 1  # a02
        V[20+2*t][base+7] += 1  # b20
        W[4*t][base+7] += 1    # C[0][2t]
        
        # M9 = a02 * b21 -> C[0][2t+1]
        U[2][base+8] += 1  # a02
        V[20+2*t+1][base+8] += 1  # b21
        W[4*t+2][base+8] += 1  # C[0][2t+1]
        
        # M10 = a12 * b20 -> C[1][2t]
        U[5][base+9] += 1  # a12
        V[20+2*t][base+9] += 1  # b20
        W[4*t+1][base+9] += 1  # C[1][2t]
        
        # M11 = a12 * b21 -> C[1][2t+1]
        U[5][base+10] += 1  # a12
        V[20+2*t+1][base+10] += 1  # b21
        W[4*t+3][base+10] += 1  # C[1][2t+1]
    
    # Verify the decomposition
    def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
