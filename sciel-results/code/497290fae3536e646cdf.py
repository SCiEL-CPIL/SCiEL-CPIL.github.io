import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    t_limit = 115.0

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W indexed by W[k*n + i, r] where C = A*B, C is n x p
    # C[i, k] = sum_{j} A[i, j] * B[j, k]
    # W should satisfy: for each r, sum_r U[i,j,r] * V[j,k,r] * W[r, k, i] = C[k,i] ... wait
    # Actually the tensor is: T[i, j, k] = delta (it's the multiplication tensor)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C = sum_r U[r] outer V[r] outer W[r] (in some indexing)
    # Standard: for matrix mult (n x m) * (m x p) -> (n x p)
    # T in R^{n x m x p}, T[i,j,k] = 1 if ... 
    # Actually the tensor multiplication: C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k in the output space
    # Decomposition rank: find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    # For each i, j, k: sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since C_{ik} = sum_j A_{ij} B_{jk}, the tensor coefficient for (i,j,k) is 1 for all i,j,k)

    # Let's use a different partitioning: partition the m=4 dimension differently
    # Or try a direct rank search with small modifications

    # Strassen-based: partition into 2x2 blocks
    # A is 2x4, B is 4x8
    # Partition A as [A00 A01; A10 A11] where each block is 1x2
    # Partition B as 2x4 blocks: B[j,k] where j in 0..3, k in 0..7
    # Group j into pairs: j=0,1 is top, j=2,3 is bottom
    # Group k into groups of 2: k=0,1; 2,3; 4,5; 6,7

    # Actually let me try a 4x4x8 approach: treat A as 2x4 = (2 blocks of 1x4)
    # Or better: partition into 1x2 x 2x4 subproblems
    # A[i, j] * B[j, k] where i in {0,1}, j in {0,1,2,3}, k in {0..7}

    # Let's try partitioning j into two groups of 2: j in {0,1} and j in {2,3}
    # For each group of 2 j-values, and each group of 2 k-values, we have a 1x2 * 2x2 -> 1x2 subproblem
    # A[i, j:j+2] (1x2) * B[j:j+2, k:k+2] (2x2) -> C[i, k:k+2] (1x2)

    # 1x2 * 2x2 = rank 4 schoolbook, or can we do better?
    # 1x2 * 2x2: A is 1x2, B is 2x2, C is 1x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # This is just 4 scalar multiplications: a0*b00, a0*b01, a1*b10, a1*b11
    # Can we do it in 3? Let's see... probably not easily.

    # Alternative: partition into 2x2 * 2x2 subproblems
    # A is 2x4: [A_left | A_right] where each is 2x2
    # B is 4x8: [B_top; B_bottom] where each is 2x8
    # C = A_left * B_top + A_right * B_bottom, each term is 2x8
    # A_left * B_top: 2x2 * 2x8 -> 2x8
    # This can be partitioned into 2 subproblems of 2x2 * 2x4 (each giving 2x4 result)
    # Or 4 subproblems of 2x2 * 2x2 (each giving 2x2 result)

    # 2x2 * 2x2: rank 7 (Strassen)
    # 2x2 * 2x4: can partition B into two 2x2 blocks, so 2 * rank(2x2*2x2) = 14
    #   or find a direct decomposition: 2x2 * 2x4 -> 2x4
    #   T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    #   This is equivalent to 2 independent 2x2*2x2 problems (columns 0-1 and 2-3)
    #   So rank = 7 + 7 = 14

    # So: A_left * B_top: partition B_top (2x8) into 4 blocks of 2x2
    #   rank = 4 * 7 = 28
    # A_right * B_bottom: same, rank = 28
    # Total = 56

    # This is what the current program does. Let me try a different partition.

    # Alternative: Don't partition A at all. Treat it as 2x4 * 4x8 directly.
    # Partition only B: into 4 groups of 4 columns.
    # A (2x4) * B[:, k0:k0+4] (4x4) -> 2x4
    # 2x4 * 4x4: rank?
    # Schoolbook: 2*4*4 = 32 per group, total 128. Worse.

    # Try: partition A into 2 groups of 2 rows? No, A is 2x4, only 2 rows.

    # Let me try a 4x8 perspective:
    # Treat A^T (4x2) * ... no, let's think differently.

    # Try partitioning k into groups of 4:
    # A (2x4) * B[:, 0:4] (4x4) -> 2x4
    # A (2x4) * B[:, 4:8] (4x4) -> 2x4
    # 2x4 * 4x4: this is (2x4) * (4x4) = 2x4
    # Partition 4x4 into four 2x2 blocks:
    # 2x4 * 2x2 = ?
    #   A (2x4) * B (2x2) where the 4 columns of A pair with... 
    #   Actually 2x4 * 4x2 -> 2x2 if B is 4x2
    #   But B is 2x2 here.

    # Let me try: 2x4 * 4x4
    # = (2x2 | 2x2) * (4x4)
    # = 2x2 * top2rows(4x4) + 2x2 * bottom2rows(4x4)
    # Each 2x2 * 2x4 -> 2x4
    # 2x2 * 2x4: partition 2x4 into two 2x2 -> 2 * 7 = 14
    # So 2x4 * 4x4 = 14 + 14 = 28
    # Two such groups: 28 + 28 = 56. Same as before.

    # Try a completely different approach: use 4x4 Strassen-like on a subblock.
    # Actually, let's try to find a rank-54 or rank-52 decomposition by mixing blocks.

    # Idea: Some of the 2x2*2x2 subproblems might share structure if we reorganize.
    # For instance, the top-left 2x2 of A times top-left 2x2 of B gives top-left 2x2 of C.
    # The top-right 2x2 of A times top-left 2x2 of B gives top-right 2x2 of C.
    # These are different subproblems with different A blocks.

    # Let me try: group by k instead.
    # C[:, 0:2] = A * B[:, 0:2]  (2x4 * 4x2 -> 2x2)
    # C[:, 2:4] = A * B[:, 2:4]  (2x4 * 4x2 -> 2x2)
    # C[:, 4:6] = A * B[:, 4:6]  (2x4 * 4x2 -> 2x2)
    # C[:, 6:8] = A * B[:, 6:8]  (2x4 * 4x2 -> 2x2)

    # 2x4 * 4x2: rank?
    # A (2x4) * B (4x2) -> C (2x2)
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0] + A[1,3]B[3,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1] + A[1,3]B[3,1]

    # Schoolbook: 2*4*2 = 16
    # Can we do better? Partition B into two 2x2 blocks:
    #   B_top (2x2) = B[0:2, :], B_bot (2x2) = B[2:4, :]
    #   A_left (2x2) = A[:, 0:2], A_right (2x2) = A[:, 2:4]
    #   C = A_left * B_top + A_right * B_bot
    #   Each 2x2 * 2x2 -> 2x2: rank 7
    #   Total: 7 + 7 = 14
    # Four such groups (for k=0-1, 2-3, 4-5, 6-7): 4 * 14 = 56. Same.

    # Try 3 groups? No, 8/3 is not integer.

    # What if we use a 2x4 * 4x4 -> 2x4 but with a rank-28 decomposition that's shared?
    # Or try to find if rank(2x4 * 4x4) < 28.
    # 2x4 * 4x4: n=2, m=4, p=4. Schoolbook = 32.
    # Partition: 2x4 * 4x4 = (2x2|2x2) * 4x4 = 2x2*2x4 + 2x2*2x4
    # 2x2*2x4: partition 2x4 into 2x2|2x2 -> 2*(2x2*2x2) = 14
    # So 2x4*4x4 = 28. Can we do 2x4*4x4 in < 28?
    # Maybe 27? 26? Hard to say without search.

    # For now, let's stick with 56 but try a different construction that's structurally different.
    # Use the k-grouping approach (4 groups of 2x4*4x2 -> 2x2).

    # Strassen 2x2x2
    U_strassen = np.zeros((4, 7), dtype=np.int64)
    V_strassen = np.zeros((4, 7), dtype=np.int64)
    W_strassen = np.zeros((4, 7), dtype=np.int64)

    # r=0: (a00+a11)(b00+b11)
    U_strassen[0,0] = 1; U_strassen[3,0] = 1
    V_strassen[0,0] = 1; V_strassen[3,0] = 1
    W_strassen[0,0] = 1; W_strassen[3,0] = 1

    # r=1: (a10+a11)b00
    U_strassen[2,1] = 1; U_strassen[3,1] = 1
    V_strassen[0,1] = 1
    W_strassen[2,1] = 1; W_strassen[3,1] = -1

    # r=2: a00(b01-b11)
    U_strassen[0,2] = 1
    V_strassen[1,2] = 1; V_strassen[3,2] = -1
    W_strassen[1,2] = 1; W_strassen[3,2] = 1

    # r=3: a11(b10-b00)
    U_strassen[3,3] = 1
    V_strassen[2,3] = 1; V_strassen[0,3] = -1
    W_strassen[0,3] = 1; W_strassen[2,3] = 1

    # r=4: (a00+a01)b11
    U_strassen[0,4] = 1; U_strassen[1,4] = 1
    V_strassen[3,4] = 1
    W_strassen[0,4] = -1; W_strassen[1,4] = 1

    # r=5: (a10-a00)(b00+b01)
    U_strassen[2,5] = 1; U_strassen[0,5] = -1
    V_strassen[0,5] = 1; V_strassen[1,5] = 1
    W_strassen[3,5] = 1

    # r=6: (a01-a11)(b10+b11)
    U_strassen[1,6] = 1; U_strassen[3,6] = -1
    V_strassen[2,6] = 1; V_strassen[3,6] = 1
    W_strassen[0,6] = 1

    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)

    # For each k_group (4 groups of 2 columns each):
    #   C[:, k0:k0+2] = A * B[:, k0:k0+2]
    #   = A_left * B_top + A_right * B_bot
    #   where A_left = A[:, 0:2], A_right = A[:, 2:4]
    #         B_top = B[0:2, k0:k0+2], B_bot = B[2:4, k0:k0+2]

    for kg in range(4):
        k0 = kg * 2
        # A_left * B_top: 2x2 * 2x2 -> 2x2
        base1 = kg * 14
        for l in range(7):
            r = base1 + l
            # A_left indices: [0,1] for row 0, [2,3] for row 1 of A (2x4)
            # a_map: (a00, a01, a10, a11) -> (0, 1, 4, 5) in U (2x4 = 8 rows)
            # U is indexed by i*m + j where i in {0,1}, j in {0,1,2,3}
            # A_left[0,0] = A[0,0] -> U[0*4+0] = U[0]
            # A_left[0,1] = A[0,1] -> U[0*4+1] = U[1]
            # A_left[1,0] = A[1,0] -> U[1*4+0] = U[4]
            # A_left[1,1] = A[1,1] -> U[1*4+1] = U[5]
            a_idx = [0, 1, 4, 5]
            # B_top[0,0] = B[0, k0], B_top[0,1] = B[0, k0+1]
            # B_top[1,0] = B[1, k0], B_top[1,1] = B[1, k0+1]
            # V is indexed by j*p + k
            # B[0, k0] -> V[0*8 + k0] = V[k0]
            # B[0, k0+1] -> V[k0+1]
            # B[1, k0] -> V[1*8 + k0] = V[8+k0]
            # B[1, k0+1] -> V[8+k0+1]
            b_idx = [k0, k0+1, 8+k0, 8+k0+1]
            # W is indexed by k*n + i
            # C[0, k0] = 0, C[0, k0+1] = 1
            # C[1, k0] = 2, C[1, k0+1] = 3
            # W[0] = C[0,0], W[1] = C[0,1], W[2] = C[1,0], W[3] = C[1,1]
            c_idx = [0, 1, 2, 3]  # local indices for 2x2 C block
            # But W is global: k*n + i
            # C[i, k] -> W[k*n + i]
            # C[0, k0] -> W[k0*2 + 0] = W[2*k0]
            # C[0, k0+1] -> W[(k0+1)*2 + 0] = W[2*k0+2]
            # C[1, k0] -> W[k0*2 + 1] = W[2*k0+1]
            # C[1, k0+1] -> W[(k0+1)*2 + 1] = W[2*k0+3]
            w_idx = [2*k0, 2*k0+2, 2*k0+1, 2*k0+3]
            
            for la in range(4):
                val = U_strassen[la, l]
                if val != 0:
                    U[a_idx[la], r] += val
            for lb in range(4):
                val = V_strassen[lb, l]
                if val != 0:
                    V[b_idx[lb], r] += val
            for lc in range(4):
                val = W_strassen[lc, l]
                if val != 0:
                    W[w_idx[lc], r] += val

        # A_right * B_bot: 2x2 * 2x2 -> 2x2
        base2 = base1 + 7
        for l in range(7):
            r = base2 + l
            # A_right[0,0] = A[0,2] -> U[0*4+2] = U[2]
            # A_right[0,1] = A[0,3] -> U[3]
            # A_right[1,0] = A[1,2] -> U[6]
            # A_right[1,1] = A[1,3] -> U[7]
            a_idx = [2, 3, 6, 7]
            # B_bot[0,0] = B[2, k0], B_bot[0,1] = B[2, k0+1]
            # B_bot[1,0] = B[3, k0], B_bot[1,1] = B[3, k0+1]
            # V[2*8+k0] = V[16+k0], V[16+k0+1]
            # V[3*8+k0] = V[24+k0], V[24+k0+1]
            b_idx = [16+k0, 16+k0+1, 24+k0, 24+k0+1]
            w_idx = [2*k0, 2*k0+2, 2*k0+1, 2*k0+3]
            
            for la in range(4):
                val = U_strassen[la, l]
                if val != 0:
                    U[a_idx[la], r] += val
            for lb in range(4):
                val = V_strassen[lb, l]
                if val != 0:
                    V[b_idx[lb], r] += val
            for lc in range(4):
                val = W_strassen[lc, l]
                if val != 0:
                    W[w_idx[lc], r] += val

    if time.time() - start_time > t_limit:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
