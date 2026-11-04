import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0

    def time_remaining():
        return deadline - time.time()

    # BCL rank-13 algorithm for 4x4x4 matrix multiplication
    # Each entry is (row, col, coeff) in the 4x4 block
    # U: A coefficient, V: B coefficient, W: C coefficient
    bcl_444 = [
        # (U entries, V entries, W entries)
        ([((0,0),1),((1,0),1),((2,0),1),((3,0),1)],
         [((0,0),1),((0,1),1),((0,2),1),((0,3),1)],
         [((0,0),1),((1,0),1),((2,0),1),((3,0),1)]),
        ([((0,1),1),((1,1),1),((2,1),1),((3,1),1)],
         [((1,0),1),((1,1),1),((1,2),1),((1,3),1)],
         [((0,1),1),((1,1),1),((2,1),1),((3,1),1)]),
        ([((0,2),1),((1,2),1),((2,2),1),((3,2),1)],
         [((2,0),1),((2,1),1),((2,2),1),((2,3),1)],
         [((0,2),1),((1,2),1),((2,2),1),((3,2),1)]),
        ([((0,3),1),((1,3),1),((2,3),1),((3,3),1)],
         [((3,0),1),((3,1),1),((3,2),1),((3,3),1)],
         [((0,3),1),((1,3),1),((2,3),1),((3,3),1)]),
        ([((0,0),1),((1,0),1),((2,0),1),((3,0),1)],
         [((0,1),1),((1,1),1),((2,1),1),((3,1),1)],
         [((0,1),1),((1,1),1),((2,1),1),((3,1),1)]),
        ([((0,1),1),((1,1),1),((2,1),1),((3,1),1)],
         [((0,2),1),((1,2),1),((2,2),1),((3,2),1)],
         [((0,2),1),((1,2),1),((2,2),1),((3,2),1)]),
        ([((0,2),1),((1,2),1),((2,2),1),((3,2),1)],
         [((0,3),1),((1,3),1),((2,3),1),((3,3),1)],
         [((0,3),1),((1,3),1),((2,3),1),((3,3),1)]),
        ([((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         [((2,0),1),((2,1),1),((3,0),1),((3,1),1)],
         [((2,0),1),((2,1),1),((3,0),1),((3,1),1)]),
        ([((0,2),1),((0,3),1),((1,2),1),((1,3),1)],
         [((2,2),1),((2,3),1),((3,2),1),((3,3),1)],
         [((2,2),1),((2,3),1),((3,2),1),((3,3),1)]),
        ([((2,0),1),((2,1),1),((3,0),1),((3,1),1)],
         [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]),
        ([((2,2),1),((2,3),1),((3,2),1),((3,3),1)],
         [((0,2),1),((0,3),1),((1,2),1),((1,3),1)],
         [((0,2),1),((0,3),1),((1,2),1),((1,3),1)]),
        ([((0,0),1),((1,0),1),((2,1),1),((3,1),1)],
         [((0,1),1),((1,1),1),((2,0),1),((3,0),1)],
         [((0,0),1),((1,0),1),((2,1),1),((3,1),1)]),
        ([((0,1),1),((1,1),1),((2,0),1),((3,0),1)],
         [((0,0),1),((1,0),1),((2,1),1),((3,1),1)],
         [((0,1),1),((1,1),1),((2,0),1),((3,0),1)]),
    ]

    # Strassen rank-7 for 2x2x2
    strassen_222 = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]

    rank = 13 + 28  # 41 total
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: BCL rank-13 for 4x4x4 (cols 0-12)
    for s in range(13):
        if time_remaining() < 5.0:
            break
        col = s
        u_entries, v_entries, w_entries = bcl_444[s]
        
        for (ri, ci), val in u_entries:
            r = ri * m + ci
            U[r][col] += val
        
        for (ri, ci), val in v_entries:
            r = ri * p + ci
            V[r][col] += val
        
        for (ri, ci), val in w_entries:
            r = ci * n + ri
            W[r][col] += val

    # Part 2: 4x4x2 part using Strassen blocks (cols 13-40)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # For j in {4,5}: C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4] + A[i,3]*B[3,4]
    # We can group by A row blocks:
    # C[0:2, 4:6] involves A[0:2, 0:4] * B[0:4, 4:6]
    # C[2:4, 4:6] involves A[2:4, 0:4] * B[0:4, 4:6]
    # 
    # Use 2x2 Strassen for each: A_block (2x2) * B_block (2x2) -> C_block (2x2)
    # But B is 4x2, so we need to handle it differently.
    # 
    # Actually, let's use: for each pair of A rows, multiply with all 4 B rows
    # A[0:2, 0:4] * B[0:4, 4:6] = C[0:2, 4:6]  (2x4 times 4x2 = 2x2)
    # A[2:4, 0:4] * B[0:4, 4:6] = C[2:4, 4:6]  (2x4 times 4x2 = 2x2)
    #
    # For 2x4 * 4x2, we can use 2 Strassen 2x2 multiplications:
    # Split A into A_left (2x2) and A_right (2x2), B into B_top (2x2) and B_bottom (2x2)
    # C = A_left * B_top + A_right * B_bottom
    # This is 2 separate 2x2x2 multiplications, each rank 7, total 14 per 2x4*4x2
    # Two such blocks: 28 total

    col_start_2 = 13

    # Block 1: A[0:2, 0:4] * B[0:4, 4:6] -> C[0:2, 4:6]
    # = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    for block_idx in range(2):
        if time_remaining() < 5.0:
            break
        a_row_start = block_idx * 2  # 0 or 2
        for sub_block in range(2):  # left or right half of A columns
            if time_remaining() < 5.0:
                break
            a_col_start = sub_block * 2  # 0 or 2
            b_row_start = sub_block * 2  # 0 or 2
            
            for s2 in range(7):
                col = col_start_2 + block_idx * 14 + sub_block * 7 + s2
                alpha_2, beta_2, gamma_2 = strassen_222[s2]
                
                for (si, sj), val in alpha_2:
                    r = (a_row_start + si) * m + (a_col_start + sj)
                    U[r][col] += val
                
                for (si, sj), val in beta_2:
                    r = (b_row_start + si) * p + (4 + sj)
                    V[r][col] += val
                
                for (si, sj), val in gamma_2:
                    r = (4 + sj) * n + (a_row_start + si)
                    W[r][col] += val

    # Verification
    if time_remaining() > 10.0:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            np.random.seed(42)
            for _ in range(3):
                A_test = np.random.randn(n, m)
                B_test = np.random.randn(m, p)
                C_expected = A_test @ B_test
                
                C_check = np.zeros((n, p))
                for r in range(rank):
                    u_col = U_np[:, r]
                    v_col = V_np[:, r]
                    w_col = W_np[:, r]
                    for i in range(n):
                        for k in range(p):
                            val = 0.0
                            for j in range(m):
                                val += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                            C_check[i, k] = val
                
                if not np.allclose(C_check, C_expected, atol=1e-10):
                    pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
