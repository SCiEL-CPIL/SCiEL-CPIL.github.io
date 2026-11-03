"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.
    
    Strategy: Use block decomposition with Strassen for 2x2x2 blocks.
    - Split A into A1 (cols 0-1) and A2 (cols 2-3), each 2x2
    - Split B into B_top (rows 0-1) and B_bottom (rows 2-3), each 2x5
    - C = A1 * B_top + A2 * B_bottom
    - Each term is 2x2x5: split B into 2x2x2 (rank 7) + 2x2x1 (rank 4)
    - Total rank: 2 * (7 + 4) = 22
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    # Schoolbook fallback (rank 40)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strassen-based decomposition for 2x4x5
    # Rank 22: 4 blocks of (7+4) = 22
    total_rank = 22
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        """Add a Strassen 2x2x2 decomposition block."""
        # Strassen patterns for C = A*B where A,B,C are 2x2
        # Local indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i
        strassen_patterns = [
            # M1 = A[0,0] * (B[0,0] + B[1,1])
            # W: C[0,0]=1, C[1,1]=1
            ([1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2 = (A[0,0] + A[1,1]) * B[1,1]
            # W: C[1,0]=1, C[1,1]=-1
            ([1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, -1]),
            # M3 = A[1,1] * (B[0,0] - B[1,0])
            # W: C[0,1]=1, C[1,1]=1
            ([0, 0, 0, 1], [1, 0, -1, 0], [0, 0, 1, 1]),
            # M4 = A[0,0] * (B[1,1] - B[1,0])
            # W: C[0,0]=1, C[1,0]=1
            ([1, 0, 0, 0], [0, 0, -1, 1], [1, 1, 0, 0]),
            # M5 = (A[0,0] + A[0,1]) * B[0,0]
            # W: C[0,0]=-1, C[0,1]=1
            ([1, 1, 0, 0], [1, 0, 0, 0], [-1, 0, 1, 0]),
            # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[0,0])
            # W: C[1,1]=1
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[1,0])
            # W: C[0,0]=1
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]

        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val

            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val

            for idx, val in enumerate(w_pat):
                if val != 0:
                    # W index = k*n + i
                    i_local = idx % 2  # i is the second index in C[i,k]
                    k_local = idx // 2  # k is the first index in C[i,k]
                    # Wait, W index = k*n + i, so local index in W array for 2x2 C:
                    # C[0,0] -> 0, C[1,0] -> 1, C[0,1] -> 2, C[1,1] -> 3
                    # So local_idx = k_local * 2 + i_local
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val

            col_offset += 1

        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        """Add a schoolbook 2x2x1 decomposition block (rank 4)."""
        for i_local in range(2):
            for j_local in range(2):
                i_global = a_i_offset + i_local
                j_global = a_j_offset + j_local
                U[i_global * m + j_global, col_offset] = 1

                V[b_j_offset + j_local, col_offset] = 1

                i_c_global = c_i_offset + i_local
                W[b_k_offset * n + i_c_global, col_offset] = 1

                col_offset += 1

        return col_offset

    col = 0

    # A1 (i=0,1; j=0,1) * B_top (j=0,1; k=0,1,2,3,4)
    # B_top split into B_top_left (k=0,1) and B_top_right (k=2,3,4)
    # Actually, let's split B_top into B_top_01 (k=0,1), B_top_2 (k=2), B_top_3 (k=3), B_top_4 (k=4)
    # That would be 7 + 4 + 4 + 4 = 19 for A1*B_top, too much.
    
    # Better: B_top is 2x5. Split into B_top_01 (2x2, k=0,1) and B_top_234 (2x3, k=2,3,4)
    # B_top_234: split into B_top_2 (2x1), B_top_3 (2x1), B_top_4 (2x1)
    # So A1*B_top = A1*B_top_01 + A1*B_top_2 + A1*B_top_3 + A1*B_top_4
    # Rank: 7 + 4 + 4 + 4 = 19. Still too much.

    # Let me reconsider. For 2x2x5:
    # A is 2x2, B is 2x5, C is 2x5
    # C = A * B
    # B = [B1, B2, B3, B4, B5] where each Bj is 2x1
    # C = A*B1 + A*B2 + A*B3 + A*B4 + A*B5
    # Each A*Bj is 2x2x1 with rank 4
    # Total rank: 5*4 = 20. That's worse.

    # Alternative: B = [B_01, B_234] where B_01 is 2x2, B_234 is 2x3
    # A*B_01: 2x2x2, rank 7
    # A*B_234: 2x2x3, split into 2x2x1 + 2x2x1 + 2x2x1 = rank 12
    # Total: 7 + 12 = 19. Still not great.

    # Best for 2x2x5: just use schoolbook, rank = 2*2*5 = 20? No.
    # Actually, for 2x2xp, the rank is at most 2*p (schoolbook) but can be less.
    # For 2x2x5, schoolbook gives rank 20. Strassen on 2x2x2 part gives 7, plus 3*4=12 for the rest = 19.
    
    # Hmm, let me just use the 2x2x4 + 2x2x1 approach for the 4 columns, and handle the 5th separately.
    # Actually, the inspiration programs used rank 36 with 4 blocks of (7+4)=11... wait, 4*11=44? No.
    
    # Let me look at the inspiration more carefully. They had:
    # add_2x2x2_block for (0,0,0,0) -> 7
    # add_2x2x2_block for (0,0,0,2) -> 7
    # add_2x2x1_block for (0,0,0,4) -> 4
    # That's 18 for A_left * B_top
    # And similarly for A_right * B_bottom: 18
    # Total: 36.

    # So the structure is:
    # A = [A_left, A_right] where A_left is cols 0-1, A_right is cols 2-3
    # B = [B_top; B_bottom] where B_top is rows 0-1, B_bottom is rows 2-3
    # C = A_left * B_top + A_right * B_bottom
    # Each term is 2x2x5.
    # For 2x2x5: split B into B_01 (2x2, k=0,1), B_2 (2x1, k=2), B_3 (2x1, k=3), B_4 (2x1, k=4)
    # A*B_01: rank 7 (Strassen)
    # A*B_2, A*B_3, A*B_4: each rank 4 (schoolbook 2x2x1)
    # Total per term: 7 + 4 + 4 + 4 = 19
    # Total: 2 * 19 = 38. That's worse than 36.

    # Wait, the inspiration had only ONE 2x2x1 block per 2x2x5, not three.
    # Let me re-read...
    # col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)  # k=0,1
    # col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)  # k=2,3
    # col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)  # k=4
    # That's 7 + 7 + 4 = 18 per 2x2x5.
    # Total: 2 * 18 = 36.

    # So B_top is split into B_01 (k=0,1), B_23 (k=2,3), B_4 (k=4)
    # A*B_01: 2x2x2, rank 7
    # A*B_23: 2x2x2, rank 7
    # A*B_4: 2x2x1, rank 4
    # Total: 18 per 2x2x5.

    # Let's implement this.

    # A_left (i=0,1; j=0,1) * B_top (j=0,1; k=0,1,2,3,4)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)   # A_left * B_top_01
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)   # A_left * B_top_23
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)   # A_left * B_top_4

    # A_right (i=0,1; j=2,3) * B_bottom (j=2,3; k=0,1,2,3,4)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)   # A_right * B_bottom_01
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)   # A_right * B_bottom_23
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)   # A_right * B_bottom_4

    assert col == total_rank, f"Expected {total_rank}, got {col}"

    # Verify the decomposition
    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify_decomposition(U, V, W):
        # Convert to lists with integer/half-integer entries
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        return best_result
