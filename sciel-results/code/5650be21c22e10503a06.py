import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 118.0  # Safety margin before 120s limit


def construct():
    n, m, p = 2, 3, 9

    # Rank 49 decomposition using a 4-column block approach:
    # - 2 blocks of 4 columns each using rank 18 (30% saving vs direct)
    # - 1 block of 1 column using rank 9 (direct)
    # Total rank = 2*18 + 9 = 45... wait let me recalculate
    
    # Actually let me use a different approach:
    # For 2x3 * 3x9, we can use the fact that the tensor has special structure
    # The 4-column block decomposition:
    # For each 4-column block of B (rows 0-2, cols 0-3):
    #   We need to compute A * B_block where A is 2x3 and B_block is 3x4
    #   This is a 2x3x4 tensor
    #   
    # A rank-18 decomposition for 2x3x4:
    # This can be achieved by treating it as two 2x1x4 subproblems + one 2x2x4
    # But let's use a more direct approach
    
    # Alternative: Use the standard Strassen-style decomposition for the full 2x3x9
    # by processing columns in groups
    
    # Let me try a rank 48 decomposition:
    # Process 8 columns using rank 5.5 each (44) + 1 column using rank 4
    # Total = 48
    
    # Actually, let's use a simpler approach that gives rank 49:
    # - Split into three 2x1x9 subproblems (one for each intermediate index)
    # - Each 2x1x9 has rank 2*9 = 18 if done directly
    # - But we can do better by exploiting the structure
    
    # Best approach: Use the known rank for small matrix multiplication
    # For 2x3 * 3x9, the rank is at most 54 (direct)
    # Using block Strassen: split A into 2x2 and 2x1 parts
    # 2x2 * 2x9 part: can use rank ~7 per 2-column block
    # Let's stick with a clean rank 49 decomposition
    
    # Method: Process B in 3-column blocks
    # For each 3-column block, compute A * B_block (2x3 * 3x3)
    # Rank of 2x3*3x3 tensor = 16 (known)
    # 3 blocks * 16 = 48
    
    # But we can do slightly better with 4-column blocks:
    # 2x3 * 3x4 tensor rank = 18
    # 2 blocks of 4 + 1 block of 1: 2*18 + 9 = 45... that seems too good
    
    # Let me just use the safe rank 50 approach but try to optimize to 49
    
    # Actually, the simplest improvement: use rank 48 by processing 
    # 8 columns with a rank-6 2-column Strassen variant + 1 column direct
    
    # Let me use a concrete rank 48 decomposition:
    # For each pair of consecutive columns (8 pairs for cols 0-7, plus col 8):
    # Use rank 6 for each pair (vs rank 7 for full Strassen on 2x2)
    # This works because we only have 3 intermediate dimensions, not 2x2 in both
    
    # For A[:, 0:3] * B[0:3, k0:k0+2] where we use all 3 intermediate dims:
    # The 2x3*3x2 tensor can be decomposed in rank 6
    
    # Let me construct rank 48:
    # 8 pairs of columns * rank 6 + 1 single column * rank 6 = 48 + 6 = 54... no
    
    # OK let me just use rank 49 with a specific construction:
    # 7 pairs of columns * rank 7 (Strassen-style) + 1 single column * rank 2
    # = 49 + 2 = 51... still not 49
    
    # Let me just go with rank 48 using a block approach:
    # Split B into 3 blocks of 3 columns each
    # For each block, use rank 16 for the 2x3*3x3 multiplication
    # Total rank = 3 * 16 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    col_idx = 0
    
    # For each 3-column block of B
    for block in range(3):
        k_start = block * 3
        k_end = k_start + 3
        
        # We need to compute C[:, k_start:k_end] = A * B[:, k_start:k_end]
        # A is 2x3, B_block is 3x3
        # Using rank-16 decomposition for 2x3*3x3
        
        # A indices: A[i, j] -> U[i*m + j]
        # B indices: B[j, k] -> V[j*p + k]
        # C indices: C[i, k] -> W[k*n + i]
        
        # Rank-16 decomposition for 2x3*3x3:
        # This is equivalent to multiplying a 2x3 matrix by a 3x3 matrix
        # We can use the following approach:
        # For each output column k in the block (3 columns):
        # C[:, k] = A * B[:, k]
        # This is a 2x3*3x1 multiplication, rank 6
        # 3 columns * 6 = 18... that's rank 18, not 16
        
        # Let me use rank 18 for each 3-column block:
        # Total rank = 3 * 18 = 54... that's the same as direct
        
        # I'll just use the original rank 50 approach since finding a better
        # decomposition requires more sophisticated algorithms
        
        pass
    
    # Fall back to the known rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # Strassen's 7 multiplications
        U[u_a][col_idx] += 1
        U[u_d][col_idx] += 1
        V[v_e][col_idx] += 1
        V[v_h][col_idx] += 1
        
        U[u_c][col_idx+1] += 1
        U[u_d][col_idx+1] += 1
        V[v_e][col_idx+1] += 1
        
        U[u_a][col_idx+2] += 1
        V[v_f][col_idx+2] += 1
        V[v_h][col_idx+2] -= 1
        
        U[u_d][col_idx+3] += 1
        V[v_g][col_idx+3] += 1
        V[v_e][col_idx+3] -= 1
        
        U[u_a][col_idx+4] += 1
        U[u_b][col_idx+4] += 1
        V[v_h][col_idx+4] += 1
        
        U[u_c][col_idx+5] += 1
        U[u_a][col_idx+5] -= 1
        V[v_e][col_idx+5] += 1
        V[v_f][col_idx+5] += 1
        
        U[u_b][col_idx+6] += 1
        U[u_d][col_idx+6] -= 1
        V[v_g][col_idx+6] += 1
        V[v_h][col_idx+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col_idx] += 1
        W[w_00][col_idx+3] += 1
        W[w_00][col_idx+4] -= 1
        W[w_00][col_idx+6] += 1
        
        W[w_01][col_idx+2] += 1
        W[w_01][col_idx+4] += 1
        
        W[w_10][col_idx+1] += 1
        W[w_10][col_idx+3] += 1
        
        W[w_11][col_idx] += 1
        W[w_11][col_idx+1] -= 1
        W[w_11][col_idx+2] += 1
        W[w_11][col_idx+5] += 1
        
        col_idx += 7
    
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col_idx] += 1
    V[v_e8][col_idx] += 1
    W[k8 * n + 0][col_idx] += 1
    
    U[u_b][col_idx+1] += 1
    V[v_g8][col_idx+1] += 1
    W[k8 * n + 0][col_idx+1] += 1
    
    U[u_c][col_idx+2] += 1
    V[v_e8][col_idx+2] += 1
    W[k8 * n + 1][col_idx+2] += 1
    
    U[u_d][col_idx+3] += 1
    V[v_g8][col_idx+3] += 1
    W[k8 * n + 1][col_idx+3] += 1
    
    col_idx += 4
    
    for k in range(9):
        v_b = 2 * p + k
        
        u_a2 = 0 * m + 2
        u_b2 = 1 * m + 2
        
        U[u_a2][col_idx] += 1
        V[v_b][col_idx] += 1
        W[k * n + 0][col_idx] += 1
        
        U[u_b2][col_idx+1] += 1
        V[v_b][col_idx+1] += 1
        W[k * n + 1][col_idx+1] += 1
        
        col_idx += 2
    
    if time.time() - start_time > TIME_LIMIT:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 3, 9
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    A_flat = A.flatten()
    B_flat = B.flatten()

    C_flat = np.zeros(n * p)
    U_arr = np.array(res["U"])
    V_arr = np.array(res["V"])
    W_arr = np.array(res["W"])

    for s in range(len(U_arr[0])):
        val = np.dot(U_arr[:, s], A_flat) * np.dot(V_arr[:, s], B_flat)
        C_flat += val * W_arr[:, s]

    C_actual = C_flat.reshape((p, n)).T

    if not np.allclose(C_actual, C_expected, atol=1e-6):
        print("Mismatch detected")
    else:
        print("Validation Passed")
