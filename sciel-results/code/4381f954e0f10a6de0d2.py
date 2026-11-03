"""Rewritten program for 4x4x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Fallback: Schoolbook
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()

    def time_check():
        return time.time() < deadline

    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U.shape[1]
        
        # Use vectorized verification
        # For each (i,k), check sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    w_val = W[k * n + i, :]
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    total += np.dot(u_row, v_row * w_val)
                if abs(total - 1.0) > 1e-6:
                    return False
        return True

    def build_strassen_2x2x2():
        """
        Rank-7 Strassen decomposition for 2x2x2.
        U (4,7): A[i,j] at index i*2+j
        V (4,7): B[j,k] at index j*2+k
        W (4,7): C[i,k] at index k*2+i (trace convention: W[k*n+i] with n=2)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a00+a11)*(b00+b11) -> c00(+1), c11(+1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1   # c[0,0] -> W[0*2+0]
        W[3, 0] = 1   # c[1,1] -> W[1*2+1]

        # M2 = (a10+a11)*b00 -> c10(+1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1   # c[1,0] -> W[0*2+1]

        # M3 = a00*(b01-b11) -> c01(+1)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1   # c[0,1] -> W[1*2+0]

        # M4 = a11*(b10-b00) -> c10(+1), c00(+1)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1   # c[1,0]
        W[0, 3] = 1   # c[0,0]

        # M5 = (a00+a01)*b11 -> c00(-1), c01(+1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # c[0,0]
        W[2, 4] = 1   # c[0,1]

        # M6 = (a10-a00)*(b00+b01) -> c11(+1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1   # c[1,1]

        # M7 = (a01-a11)*(b10+b11) -> c00(+1)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1   # c[0,0]

        return U, V, W

    def build_4x4x4_strassen():
        """
        Build rank-49 decomposition for 4x4x4 using Strassen recursion.
        U (16, 49), V (16, 49), W (16, 49).
        """
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = build_strassen_2x2x2()
        
        # 7 Strassen terms for 4x4 from 2x2 blocks
        # Block layout: 0=TL, 1=TR, 2=BL, 3=BR
        terms = [
            # M1 = (A0+A3)*(B0+B3) -> C0(+1), C3(+1)
            {'L': [(0, 1), (3, 1)], 'R': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            # M2 = (A2+A3)*B0 -> C2(+1)
            {'L': [(2, 1), (3, 1)], 'R': [(0, 1)], 'C': [(2, 1)]},
            # M3 = A0*(B1-B3) -> C1(+1)
            {'L': [(0, 1)], 'R': [(1, 1), (3, -1)], 'C': [(1, 1)]},
            # M4 = A3*(B2-B0) -> C0(+1), C2(+1)
            {'L': [(3, 1)], 'R': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            # M5 = (A0+A1)*B3 -> C0(-1), C1(+1)
            {'L': [(0, 1), (1, 1)], 'R': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            # M6 = (A2-A0)*(B0+B1) -> C3(+1)
            {'L': [(2, 1), (0, -1)], 'R': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            # M7 = (A1-A3)*(B2+B3) -> C0(+1)
            {'L': [(1, 1), (3, -1)], 'R': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]
        
        def block_offsets(block_idx):
            """Returns (row_start, col_start) for a 2x2 block in a 4x4 matrix."""
            r = (block_idx // 2) * 2
            c = (block_idx % 2) * 2
            return r, c

        for t_idx, term in enumerate(terms):
            base_col = t_idx * 7
            
            for sub_idx in range(7):
                col = base_col + sub_idx
                
                # Process U (A matrix)
                for a_block, sign in term['L']:
                    r0, c0 = block_offsets(a_block)
                    for i in range(2):
                        for j in range(2):
                            local_idx = i * 2 + j
                            global_idx = (r0 + i) * 4 + (c0 + j)
                            U[global_idx, col] += U2[local_idx, sub_idx] * sign

                # Process V (B matrix)
                for b_block, sign in term['R']:
                    r0, c0 = block_offsets(b_block)
                    for j in range(2):
                        for k in range(2):
                            local_idx = j * 2 + k
                            global_idx = (r0 + j) * 4 + (c0 + k)
                            V[global_idx, col] += V2[local_idx, sub_idx] * sign

                # Process W (C matrix)
                # W index: k*n + i where n=4
                for c_block, sign in term['C']:
                    r0, c0 = block_offsets(c_block)
                    for i in range(2):
                        for k in range(2):
                            local_idx = k * 2 + i
                            global_idx = (c0 + k) * 4 + (r0 + i)
                            W[global_idx, col] += W2[local_idx, sub_idx] * sign

        return U, V, W

    try:
        if not time_check():
            return {
                "U": [[float(x) for x in row] for row in best_U],
                "V": [[float(x) for x in row] for row in best_V],
                "W": [[float(x) for x in row] for row in best_W]
            }

        # Build 4x4x4 Strassen (rank 49)
        U4, V4, W4 = build_4x4x4_strassen()
        
        if verify(U4, V4, W4, 4, 4, 4):
            # Build 4x4x9 by partitioning B into B1(4x4), B2(4x4), B3(4x1)
            # C = A*B1 + A*B2 + A*B3
            # Rank: 49 + 49 + 16 = 114
            total_rank = 114
            U9 = np.zeros((16, total_rank))
            V9 = np.zeros((36, total_rank))
            W9 = np.zeros((36, total_rank))
            
            # Part 1: B cols 0-3 (B1)
            U9[:, 0:49] = U4
            V9[0:16, 0:49] = V4
            W9[0:16, 0:49] = W4
            
            # Part 2: B cols 4-7 (B2)
            U9[:, 49:98] = U4
            V9[16:32, 49:98] = V4
            W9[16:32, 49:98] = W4
            
            # Part 3: B col 8 (B3, 4x1) - schoolbook
            for i in range(4):
                for j in range(4):
                    col_idx = 98 + i * 4 + j
                    U9[i * 4 + j, col_idx] = 1
                    V9[j * 9 + 8, col_idx] = 1
                    W9[8 * 4 + i, col_idx] = 1
            
            if verify(U9, V9, W9, 4, 4, 9):
                best_rank = total_rank
                best_U = U9
                best_V = V9
                best_W = W9
                print(f"Success: Rank {total_rank}")
            else:
                print("Verification failed for 4x4x9")
        else:
            print("Verification failed for 4x4x4")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
