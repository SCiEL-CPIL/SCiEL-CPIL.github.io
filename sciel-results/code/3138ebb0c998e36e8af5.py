import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 8

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strategy: Karatsuba on intermediate dimension m=4.
    # A (2x4) = [A0 | A1] where A0, A1 are 2x2.
    # B (4x8) = [B0; B1] where B0, B1 are 2x8.
    # C = A0*B0 + A1*B1 = P1 + (P3 - P1 - P2) where:
    #   P1 = A0*B0  (2x2 * 2x8, rank 14 via Kronecker 7*2)
    #   P2 = A1*B1  (2x2 * 2x8, rank 14)
    #   P3 = (A0+A1)*(B0+B1)  (2x2 * 2x8, rank 14)
    # Total rank = 3 * 14 = 42.
    #
    # The Kronecker product rank-14 decomposition for 2x2 * 2x8:
    #   Rank-7 Strassen for 2x2*2x2, applied to each of 2 column-pairs of B.
    #   Total = 7 * 2 = 14 per product.
    #
    # The three products share U and V rows (linear combinations),
    # but each has independent rank-14 structure.

    # Strassen 2x2*2x2 factors (local indices 0..3 for (0,0),(0,1),(1,0),(1,1))
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))

    # M1 = (A00+A11)(B00+B11)
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1; W_22[3, 0] = 1

    # M2 = (A10+A11)B00
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[2, 1] = 1; W_22[3, 1] = -1

    # M3 = A00(B01-B11)
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    W_22[1, 2] = 1; W_22[3, 2] = 1

    # M4 = A11(B10-B00)
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[0, 3] = -1
    W_22[0, 3] = 1; W_22[2, 3] = 1

    # M5 = (A00+A01)B11
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    W_22[0, 4] = -1; W_22[1, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    W_22[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    W_22[0, 6] = 1; W_22[1, 6] = -1

    # Global dimensions:
    # U: 8 rows (i*m+j), V: 32 rows (j*p+k), W: 16 rows (k*n+i)
    total_rank = 42
    U_global = np.zeros((8, total_rank))
    V_global = np.zeros((32, total_rank))
    W_global = np.zeros((16, total_rank))

    # Helper: add a 2x2*2x8 Kronecker-Strassen block
    # u_map: 4 local A-indices -> 4 global U-row indices
    # v_maps: list of 2, each 4 local B-indices -> 4 global V-row indices (for 2 column-pairs)
    # w_maps: list of 2, each 4 local C-indices -> 4 global W-row indices
    # scale_u, scale_v: multipliers for U and V rows (for Karatsuba sums)
    # sign_w: multiplier for W rows (for Karatsuba difference)
    def add_22x28_block(U_g, V_g, W_g, u_map, v_maps, w_maps, start_col,
                        scale_u=1.0, scale_v=1.0, sign_w=1.0):
        for pair_idx in range(2):
            v_map = v_maps[pair_idx]
            w_map = w_maps[pair_idx]
            base = start_col + pair_idx * 7
            for c in range(7):
                for l in range(4):
                    val_u = U_22[l, c]
                    val_v = V_22[l, c]
                    val_w = W_22[l, c]
                    if val_u != 0:
                        U_g[u_map[l], base + c] += val_u * scale_u
                    if val_v != 0:
                        V_g[v_map[l], base + c] += val_v * scale_v
                    if val_w != 0:
                        W_g[w_map[l], base + c] += val_w * sign_w

    # Define index maps for the three Karatsuba products.
    #
    # A0: A rows 0-1, cols 0-1 -> U indices [0, 1, 4, 5]
    # A1: A rows 0-1, cols 2-3 -> U indices [2, 3, 6, 7]
    # A0+A1: U indices for A0 have +1, A1 have +1 -> [0,1,4,5] with scale 1, [2,3,6,7] with scale 1
    #   But in the tensor decomposition, we need U rows that represent A0+A1.
    #   Since U rows are per-element, we add contributions from both A0 and A1 rows.
    #
    # B0: B rows 0-1, cols 0-7 -> V indices:
    #   col-pair 0 (cols 0-1): [0, 1, 8, 9]
    #   col-pair 1 (cols 2-3): [2, 3, 10, 11]
    #   col-pair 2 (cols 4-5): [4, 5, 12, 13]
    #   col-pair 3 (cols 6-7): [6, 7, 14, 15]
    # B1: B rows 2-3, cols 0-7 -> V indices:
    #   col-pair 0 (cols 0-1): [16, 17, 24, 25]
    #   col-pair 1 (cols 2-3): [18, 19, 26, 27]
    #   col-pair 2 (cols 4-5): [20, 21, 28, 29]
    #   col-pair 3 (cols 6-7): [22, 23, 30, 31]
    #
    # C: 2x8 output. W indices (k*n+i):
    #   col-pair 0 (k=0,1): [0, 1, 2, 3]
    #   col-pair 1 (k=2,3): [4, 5, 6, 7]
    #   col-pair 2 (k=4,5): [8, 9, 10, 11]
    #   col-pair 3 (k=6,7): [12, 13, 14, 15]

    # For the 2x2*2x8 block, we split B (2x8) into 4 column-pairs of 2x2.
    # The Kronecker structure: for each of the 4 column-pairs, we apply rank-7 Strassen.
    # But wait: the standard Kronecker product for 2x2 * 2x8 uses 2 column-pairs,
    # each with rank 7, giving rank 14. Let me re-examine.
    #
    # Actually, 2x2 * 2x8: A is 2x2, B is 2x8.
    # Split B into two 2x4 matrices: B = [B_left | B_right].
    # Then A*B = A*B_left + A*B_right.
    # Each A*B_i is 2x2 * 2x4.
    # Split B_i into two 2x2: B_i = [B_i1 | B_i2].
    # A*B_i = A*B_i1 + A*B_i2, each 2x2*2x2 with rank 7.
    # So A*B_i has rank 14, and A*B has rank 28.
    #
    # Wait, that gives rank 28 per 2x2*2x8 product, not 14.
    # Let me reconsider. The Kronecker product rank(r1, r2) = r1*r2.
    # rank(2x2*2x2) = 7, rank(2x1*1x4) = 1 (trivial).
    # So rank(2x2*2x8) via Kronecker = 7 * 1 * ... hmm.
    #
    # Actually, the tensor for 2x2 * 2x8 multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, j in {0,1}, k in {0..7}.
    # This is a 2*1 * 1*8 tensor product of the 2x2*2x2 tensor? No.
    #
    # Let me think differently. The 2x2*2x8 tensor has 2*1*8 = 16 entries (one per i,j,k).
    # The Kronecker product of the 2x2*2x2 tensor (rank 7) with a trivial 1x1*1x4 tensor (rank 1)
    # would give a 2x2*2x8 tensor of rank 7*1 = 7? No, that's not right either.
    #
    # The correct way: 2x2 * 2x8 can be seen as (2x2 * 2x2) ⊗ (1x1 * 1x4) in some sense,
    # but the dimensions don't factor that way.
    #
    # Let me just directly construct it. For 2x2 * 2x8:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # We can write this as: for each k, C[:,k] = A * B[:,k] (2x2 * 2x1).
    # The 2x2*2x1 tensor has rank... let's see. C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is a 2*1*1 tensor (for fixed k) with rank at most 2, but actually it's just
    # the standard matrix-vector product which has tensor rank 2 (since it's bilinear
    # in A and B[:,k]).
    #
    # Actually, the full 2x2*2x8 tensor: T[i, j, k] = delta(i,j) for the multiplication? No.
    # T[i*m+j, j*p+k, k*n+i] where n=2, m=2, p=8.
    # T[i*2+j, j*8+k, k*2+i] = 1.
    #
    # The rank of this tensor: we can decompose it as sum over j of (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ ...
    # Actually, the standard decomposition has rank n*m*p = 2*2*8 = 32 (schoolbook).
    #
    # Using Strassen on the 2x2 part: for each pair of k-values, we can apply rank-7.
    # There are 8 k-values. If we group them into pairs: 4 groups.
    # Each group of 2 k-values: 2x2 * 2x2 (for those 2 k's), rank 7.
    # Total: 4 * 7 = 28.
    #
    # Can we do better? The tensor 2x2*2x8:
    # T[i*2+j, j*8+k, k*2+i] = 1 for i,j in {0,1}, k in {0..7}.
    #
    # Alternative: use the fact that B has 8 columns. We can group k into 4 pairs.
    # For each pair (k, k+1), we have a 2x2*2x2 sub-tensor.
    # 4 such sub-tensors, each rank 7, total rank 28.
    #
    # Or group into 2 groups of 4: 2x2 * 2x4.
    # 2x2*2x4: T[i*2+j, j*4+k, k*2+i] for k in {0,1,2,3}.
    # This can be decomposed as two 2x2*2x2 (k=0,1 and k=2,3), each rank 7, total 14.
    # So 2x2*2x8 = 2 * 14 = 28. Same as before.
    #
    # So rank(2x2*2x8) = 28, not 14. My earlier calculation was wrong.
    #
    # Therefore:
    # P1 = A0*B0: 2x2 * 2x8, rank 28
    # P2 = A1*B1: 2x2 * 2x8, rank 28
    # P3 = (A0+A1)*(B0+B1): 2x2 * 2x8, rank 28
    # Total rank = 3 * 28 = 84. That's WORSE than 56!
    #
    # The Karatsuba trick only helps when the sub-products are independent and the
    # sum/difference structure allows sharing. But here, the three products each have
    # rank 28, and they don't share tensor rank because the U and V factors are different
    # (A0 vs A1 vs A0+A1, B0 vs B1 vs B0+B1).
    #
    # Wait, but in a tensor decomposition, the rank is the number of terms in the sum.
    # If we write C = P1 + (P3 - P1 - P2), we need to express this as a sum of rank-1 tensors.
    # P1 has 28 rank-1 terms, P2 has 28, P3 has 28. Total 84 rank-1 terms.
    # But some of these might be linearly dependent or cancel?
    #
    # Actually, in the tensor decomposition framework, we just need:
    # sum_r u_r ⊗ v_r ⊗ w_r = T
    # where T is the multiplication tensor.
    #
    # The Karatsuba approach: C = A0*B0 + A1*B1.
    # We compute P1 = A0*B0, P2 = A1*B1, P3 = (A0+A1)*(B0+B1) = A0*B0 + A0*B1 + A1*B0 + A1*B1.
    # Then A0*B1 + A1*B0 = P3 - P1 - P2.
    # And C = P1 + A1*B1 = P1 + P2. (This doesn't use P3!)
    #
    # I confused myself. The Karatsuba trick is for when you need to compute
    # A0*B0 + A1*B1 AND you also need A0*B1 + A1*B0 (e.g., in full matrix multiplication
    # where all four products appear). For C = A0*B0 + A1*B1, you just need P1 and P2.
    #
    # So Karatsuba doesn't help here! The current approach of 2 independent 2x2*2x8
    # products (each rank 28) giving total rank 56 is already optimal for this splitting.
    #
    # Let me reconsider the problem. Can we do better than 56?
    #
    # The tensor is 2x4 * 4x8. Rank lower bound: max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    # Also, rank >= n*p = 16 (from the output dimension).
    #
    # Known results: rank of m×n × n×p multiplication.
    # For 2×4 × 4×8: this is a specific case.
    #
    # The current best is 56 (from the seed program). Can we do 48, 40, 32?
    #
    # Let me try a different decomposition. Instead of splitting m=4 into two 2's,
    # what if we use the full 2x4 * 4x8 structure more cleverly?
    #
    # Alternative: view C[i,k] = sum_j A[i,j] B[j,k] as a bilinear map.
    # The tensor has 8*32*16 = 4096 entries, but only 2*4*8 = 64 are nonzero.
    #
    # Let me try: split k=8 into four 2's. For each pair of k's (k0, k0+1),
    # C[:, k0:k0+2] = A * B[:, k0:k0+2] (2x4 * 4x2).
    # The 2x4 * 4x2 tensor: rank?
    # C[i, k] = sum_j A[i,j] B[j,k].
    # This is like 2x4 * 4x2. We can split A into [A_left | A_right] (2x2 each)
    # and B into [B_top; B_bottom] (2x2 each).
    # C = A_left * B_top + A_right * B_bottom.
    # Each 2x2*2x2 has rank 7. Total rank 14 per k-pair.
    # 4 k-pairs: 4 * 14 = 56. Same as before.
    #
    # What if we don't split independently? The four k-pairs share the same A and B rows.
    # But in tensor rank, sharing doesn't reduce rank unless we can factor it.
    #
    # Let me try yet another approach: use the 4x4 Strassen or other known decompositions.
    #
    # Actually, let me reconsider. The problem is 2x4 * 4x8.
    # What if we view it as (2x4 * 4x4) + (2x4 * 4x4)?
    # Split B into two 4x4 blocks: B = [B1 | B2].
    # C = A*B1 + A*B2.
    # Each 2x4 * 4x4: split A into [A1|A2] (2x2 each), B_i into [B_i1; B_i2] (2x2 each).
    # A*B_i = A1*B_i1 + A2*B_i2. Each 2x2*2x2 rank 7. Total 14 per 2x4*4x4.
    # Two such: 28. Wait, that's only 28? Let me recount.
    #
    # A*B1: A is 2x4, B1 is 4x4.
    # A = [A1 | A2] (2x2 each), B1 = [B11; B12] (2x2 each).
    # A*B1 = A1*B11 + A2*B12. Two 2x2*2x2 products, each rank 7. Total 14.
    # Similarly A*B2: 14.
    # Total: 28. But that can't be right because the output is 2x8, not 2x4.
    #
    # Oh wait, B1 is 4x4, so A*B1 is 2x4. And B2 is 4x4, so A*B2 is 2x4.
    # C = [A*B1 | A*B2] which is 2x8. Yes!
    # So total rank = 14 + 14 = 28?? That seems too good.
    #
    # Let me verify. The tensor for 2x4 * 4x8:
    # T[i*4+j, j*8+k, k*2+i] = 1.
    #
    # If I decompose as:
    # For B1 (cols 0-3 of B, i.e., k=0..3):
    #   A1*B11: A1 is A[:, 0:2] (2x2), B11 is B[0:2, 0:4] (2x4).
    #   Wait, B11 should be 2x2 for the 2x2*2x2 decomposition.
    #
    # I think I'm confusing myself. Let me be very careful.
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k], i in {0,1}, k in {0..7}.
    #
    # Split k into {0,1,2,3} and {4,5,6,7}.
    # C[:, 0:4] = A * B[:, 0:4] (2x4 * 4x4 = 2x4)
    # C[:, 4:8] = A * B[:, 4:8] (2x4 * 4x4 = 2x4)
    #
    # For C[:, 0:4] = A * B[:, 0:4]:
    #   Let B_left = B[:, 0:4] (4x4).
    #   Split A = [A0 | A1] where A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2).
    #   Split B_left = [B0; B1] where B0 = B_left[0:2, :] (2x4), B1 = B_left[2:4, :] (2x4).
    #   C[:, 0:4] = A0 * B0 + A1 * B1.
    #   Each term is 2x2 * 2x4.
    #
    # For 2x2 * 2x4: split the 2x4 into two 2x2's (by columns).
    #   A0 * B0 = A0 * B0[:, 0:2] + A0 * B0[:, 2:4].
    #   Each 2x2 * 2x2, rank 7.
    #   So A0 * B0 has rank 14.
    #   Similarly A1 * B1 has rank 14.
    #   Total for C[:, 0:4]: 28.
    #
    # Similarly for C[:, 4:8]: 28.
    # Total: 56. Same as before.
    #
    # OK so 56 seems to be the natural decomposition. Let me just implement it correctly
    # and verify. The current program already does this, so let me make sure it's correct
    # and return it. If it scores 1.0, then 56 is the baseline and we need a different approach.
    #
    # Let me try a completely different strategy: use the fact that n=2 is small.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # This is like two independent 1x4 * 4x8 multiplications (one for each row of C).
    # 1x4 * 4x8: C[k] = sum_j A[j] B[j,k]. This is a dot product for each k.
    # Tensor rank of 1x4 * 4x8: T[j, j*8+k, k] = 1 (using 1x4*4x8 indexing).
    # Actually, for 1x4 * 4x8: n=1, m=4, p=8.
    # T[i*4+j, j*8+k, k*1+i] = T[j, j*8+k, k] = 1 for j in 0..3, k in 0..7.
    # This is just the "outer product" structure: sum_j (e_j ⊗ e_{j*8} ⊗ e_k) ... no.
    # T[j, j*8+k, k] = 1. For fixed j and k, there's one nonzero entry.
    # The rank is 4*8 = 32 (schoolbook), and it's actually minimal since
    # each of the 32 entries is independent (no two share the same j,k pair in a way
    # that allows cancellation).
    #
    # So two such: 64. Worse than 56.
    #
    # I think 56 is likely close to optimal for this problem. Let me just implement
    # the correct 56-rank decomposition and verify it.

    # Let me implement the 56-rank decomposition properly.
    # Structure:
    # - Split j (intermediate dim 4) into two groups of 2: j_group in {0, 1}.
    #   j_group=0: j in {0,1}, j_group=1: j in {2,3}.
    # - For each j_group, split k (output dim 8) into four groups of 2: k_group in {0,1,2,3}.
    #   k_group=g: k in {2g, 2g+1}.
    # - Each (j_group, k_group) gives a 2x2 * 2x2 multiplication (rank 7).
    # - Total: 2 * 4 * 7 = 56.

    # Global index mappings:
    # U rows: i*m + j = 4*i + j, i in {0,1}, j in {0,1,2,3}.
    # V rows: j*p + k = 8*j + k, j in {0,1,2,3}, k in {0..7}.
    # W rows: k*n + i = 2*k + i, k in {0..7}, i in {0,1}.

    # For a given (j_group, k_group):
    # j_group=0: j_local 0->j_global 0, j_local 1->j_global 1.
    # j_group=1: j_local 0->j_global 2, j_local 1->j_global 3.
    # k_group=g: k_local 0->k_global 2g, k_local 1->k_global 2g+1.

    # Local 2x2*2x2 indices:
    # A (2x2): (i, j_local) -> u_local = 2*i + j_local.
    # B (2x2): (j_local, k_local) -> v_local = 2*j_local + k_local.
    # C (2x2): (i, k_local) -> w_local = 2*i + k_local.

    # Global U row for local (i, j_local) in j_group:
    #   u_global = 4*i + (2*j_group + j_local)
    # Global V row for local (j_local, k_local) in j_group, k_group:
    #   v_global = 8*(2*j_group + j_local) + (2*k_group + k_local)
    # Global W row for local (i, k_local) in k_group:
    #   w_global = 2*(2*k_group + k_local) + i

    U_global = np.zeros((8, 56))
    V_global = np.zeros((32, 56))
    W_global = np.zeros((16, 56))

    col = 0
    for jg in range(2):
        for kg in range(4):
            for c in range(7):
                for l in range(4):
                    # l encodes local (i, j_local) for U, (j_local, k_local) for V, (i, k_local) for W
                    # But U_22, V_22, W_22 use different local indexings!
                    # U_22[l, c]: l is local A index (2*i + j_local)
                    # V_22[l, c]: l is local B index (2*j_local + k_local)
                    # W_22[l, c]: l is local C index (2*i + k_local)
                    # So we need to map each l separately.

                    # U mapping: l = 2*i + j_local
                    i_u = l // 2
                    jl_u = l % 2
                    u_g = 4 * i_u + (2 * jg + jl_u)
                    if U_22[l, c] != 0:
                        U_global[u_g, col] += U_22[l, c]

                    # V mapping: l = 2*j_local + k_local
                    jl_v = l // 2
                    kl_v = l % 2
                    v_g = 8 * (2 * jg + jl_v) + (2 * kg + kl_v)
                    if V_22[l, c] != 0:
                        V_global[v_g, col] += V_22[l, c]

                    # W mapping: l = 2*i + k_local
                    i_w = l // 2
                    kl_w = l % 2
                    w_g = 2 * (2 * kg + kl_w) + i_w
                    if W_22[l, c] != 0:
                        W_global[w_g, col] += W_22[l, c]

                col += 1

    # Verify
    if time.time() < deadline - 5.0:
        T_recon = np.zeros((8, 32, 16))
        for c in range(56):
            T_recon += np.einsum('a,b,c->abc', U_global[:, c], V_global[:, c], W_global[:, c])

        T_orig = np.zeros((8, 32, 16))
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    T_orig[i*4+j, j*8+k, k*2+i] = 1

        if not np.allclose(T_recon, T_orig):
            return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
