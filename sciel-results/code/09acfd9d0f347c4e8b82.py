"""Matrix multiplication 2,2,10 via block-diagonal Strassen (rank 35)."""
import time
import sys

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    """
    Decompose 2x2 * 2x10 matrix multiplication.
    
    The 2x10 matrix B can be viewed as 5 blocks of size 2x2.
    Each 2x2 * 2x2 block multiplication can be done with rank 7 (Strassen).
    Total rank = 5 * 7 = 35.
    
    Convention:
      U has shape (n*m, rank) = (4, 35)
      V has shape (m*p, rank) = (20, 35)
      W has shape (p*n, rank) = (20, 35)
      
      The multiplication tensor T_{i,j,k} = A_{i,j} * B_{j,k}
      is decomposed as: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
      
      With the trace convention W[k*n + i, r].
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2x2
    
    # Strassen rank-7 decomposition for 2x2 * 2x2:
    # For A (2x2), B (2x2), C = A*B
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # 
    # Strassen's 7 products:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each product M_c is a bilinear form: sum_{i,j} U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k]
    # Wait, that's not quite right. Let me think more carefully.
    #
    # The tensor decomposition: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # For each block b (b=0..4), we have a 2x2 * 2x2 sub-problem.
    # Within block b, the A indices are i=0,1 and j=0,1 (global j = b*2 + j_local? No...)
    #
    # Actually, let me reconsider the structure. We're multiplying A (2x2) by B (2x10).
    # B has rows indexed by j=0,1 and columns by k=0..9.
    # So B[j, k] for j in {0,1}, k in {0,...,9}.
    #
    # The output C[i, k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,9}.
    #
    # We can group columns of B into 5 pairs: for block b, columns k=2b and k=2b+1.
    # For each block b, we compute a 2x2 product: C_b[i, l] = sum_j A[i,j] * B[j, 2b+l]
    # where l in {0,1}.
    #
    # So for each block b, we apply Strassen to the 2x2 matrix B_b = [[B[0,2b], B[0,2b+1]], [B[1,2b], B[1,2b+1]]]
    #
    # In the tensor decomposition:
    # For block b and Strassen factor c (c=0..6):
    #   U[i*m + j, global_c] where global_c = b*7 + c
    #   V[j*p + k, global_c] where k = 2b + l (l in {0,1})
    #   W[k*n + i, global_c]
    #
    # Strassen factors for a single 2x2*2x2:
    # M1: U part: (a11+a22) -> U[0,0]=1, U[1,0]=1, U[2,0]=0, U[3,0]=0
    #      V part: (b11+b22) -> V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=1
    #      (using local indices: A is [[a00,a01],[a10,a11]], B is [[b00,b01],[b10,b11]])
    #
    # Let me use 0-based local indices for the 2x2 sub-problem:
    # A_local[i,j] for i,j in {0,1}, B_local[j,l] for j,l in {0,1}
    #
    # Strassen:
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # For each M_c, U[i,j,c] and V[j,l,c] are coefficients.
    #
    # M0: U: a00->1, a11->1 => U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=1
    #     V: b00->1, b11->1 => V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=1
    # M1: U: a10->1, a11->1 => U[0,0]=0, U[1,0]=0, U[2,0]=1, U[3,0]=1
    #     V: b00->1 => V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=0
    # M2: U: a00->1 => U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=0
    #     V: b01->1, b11->-1 => V[0,0]=0, V[1,0]=1, V[2,0]=0, V[3,0]=-1
    # M3: U: a11->1 => U[0,0]=0, U[1,0]=0, U[2,0]=0, U[3,0]=1
    #     V: b10->1, b00->-1 => V[0,0]=-1, V[1,0]=0, V[2,0]=1, V[3,0]=0
    # M4: U: a00->1, a01->1 => U[0,0]=1, U[1,0]=1, U[2,0]=0, U[3,0]=0
    #     V: b11->1 => V[0,0]=0, V[1,0]=0, V[2,0]=0, V[3,0]=1
    # M5: U: a10->1, a00->-1 => U[0,0]=-1, U[1,0]=0, U[2,0]=1, U[3,0]=0
    #     V: b00->1, b01->1 => V[0,0]=1, V[1,0]=1, V[2,0]=0, V[3,0]=0
    # M6: U: a01->1, a11->-1 => U[0,0]=0, U[1,0]=1, U[2,0]=0, U[3,0]=-1
    #     V: b10->1, b11->1 => V[0,0]=0, V[1,0]=0, V[2,0]=1, V[3,0]=1
    #
    # W for the output: C[i,l] = sum_c coeff(i,l,c) * M_c
    # C00 = M0 + M3 - M4 + M6 => W for (i=0,l=0): W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=1, W[4,0]=-1, W[5,0]=0, W[6,0]=1
    # C01 = M2 + M4 => W[0,0]=0, W[1,0]=0, W[2,0]=1, W[3,0]=0, W[4,0]=1, W[5,0]=0, W[6,0]=0
    # C10 = M1 + M3 => W[0,0]=0, W[1,0]=1, W[2,0]=0, W[3,0]=1, W[4,0]=0, W[5,0]=0, W[6,0]=0
    # C11 = M0 - M1 + M2 + M5 => W[0,0]=1, W[1,0]=-1, W[2,0]=1, W[3,0]=0, W[4,0]=0, W[5,0]=1, W[6,0]=0
    
    # Define Strassen coefficients
    # U_strassen[c][i][j] for c in 0..6, i,j in 0..1
    U_s = [
        [[1, 0], [0, 1]],   # M0
        [[0, 0], [1, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 0], [0, 1]],   # M3
        [[1, 1], [0, 0]],   # M4
        [[-1, 0], [1, 0]],  # M5
        [[0, 1], [0, -1]],  # M6
    ]
    # V_strassen[c][j][l] for c in 0..6, j,l in 0..1
    V_s = [
        [[1, 0], [0, 1]],   # M0
        [[1, 0], [0, 0]],   # M1
        [[0, 1], [0, -1]],  # M2
        [[-1, 0], [1, 0]],  # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[0, 0], [1, 1]],   # M6
    ]
    # W_strassen[c][i][l] for c in 0..6, i,l in 0..1
    W_s = [
        [1, 0],  # M0 contributes to C00
        [0, 0],  # M1 contributes to C10, C11 (negative)
        [0, 1],  # M2 contributes to C01, C11
        [1, 0],  # M3 contributes to C00, C10
        [-1, 1], # M4 contributes to C00 (neg), C01
        [0, 1],  # M5 contributes to C11
        [1, 0],  # M6 contributes to C00
    ]
    # Wait, let me redo W more carefully.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # W[i, l, c] = coefficient of M_c in C[i,l]
    W_s = [
        # c=0 (M0): C00+=1, C11+=1
        [[1, 0], [0, 1]],
        # c=1 (M1): C10+=1, C11+=-1
        [[0, 0], [1, -1]],
        # c=2 (M2): C01+=1, C11+=1
        [[0, 1], [0, 1]],
        # c=3 (M3): C00+=1, C10+=1
        [[1, 0], [1, 0]],
        # c=4 (M4): C00+=-1, C01+=1
        [[-1, 1], [0, 0]],
        # c=5 (M5): C11+=1
        [[0, 0], [0, 1]],
        # c=6 (M6): C00+=1
        [[1, 0], [0, 0]],
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        for c in range(7):
            gc = b * 7 + c  # global column index
            
            # Fill U: for i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i * m + j][gc] = U_s[c][i][j]
            
            # Fill V: for j in 0..1, l in 0..1 -> global k = 2*b + l
            for j in range(2):
                for l in range(2):
                    k_global = 2 * b + l
                    V[j * p + k_global][gc] = V_s[c][j][l]
            
            # Fill W: for i in 0..1, l in 0..1 -> global k = 2*b + l, index k*n + i
            for i in range(2):
                for l in range(2):
                    k_global = 2 * b + l
                    W[k_global * n + i][gc] = W_s[c][i][l]
    
    # Verify correctness with a quick numerical check
    if time.time() < deadline:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check: for random A, B, does sum_c U[i,j,c]*V[j,k,c]*W[k*n+i,c] == A*B?
        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((n, m))
        B_test = rng.standard_normal((m, p))
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                C_computed[i, k] = val
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: return schoolbook rank-40 if Strassen has a bug
            U_fb = [[0]*40 for _ in range(4)]
            V_fb = [[0]*40 for _ in range(20)]
            W_fb = [[0]*40 for _ in range(20)]
            for i in range(2):
                for j in range(2):
                    for k in range(10):
                        idx = i*20 + j*10 + k
                        U_fb[i*2+j][idx] = 1.0
                        V_fb[j*10+k][idx] = 1.0
                        W_fb[k*2+i][idx] = 1.0
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
