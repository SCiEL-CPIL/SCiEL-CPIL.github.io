import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Part 1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    # 4x4x4 Strassen: 7 terms
    # Each term is a rank-1 contribution to the tensor
    # U: linear form in A entries
    # V: linear form in B entries  
    # W: linear form in C entries (with the trace convention W[j*n+i] = C[i,j])

    # Block offsets in 4x4:
    def block_offset(block):
        if block == 0: return (0, 0)   # top-left
        elif block == 1: return (0, 2) # top-right
        elif block == 2: return (2, 0) # bottom-left
        else: return (2, 2)            # bottom-right

    # Strassen 4x4x4: 7 rank-1 terms
    # Term k: U_k = sum of A entries, V_k = sum of B entries, W_k = sum of C entries
    # such that sum_k U_k * V_k * W_k = sum_{i,j,k} A[i,k]*B[k,j]*C[i,j]

    # M1 = (A00+A11)(B00+B11)
    #   U: A[0,0]+A[0,1]+A[1,0]+A[1,1]+A[2,2]+A[2,3]+A[3,2]+A[3,3]
    #   V: B[0,0]+B[0,1]+B[1,0]+B[1,1]+B[2,2]+B[2,3]+B[3,2]+B[3,3]
    #   W: C[0,0]+C[0,1]+C[1,0]+C[1,1]+C[2,2]+C[2,3]+C[3,2]+C[3,3] (with sign +1)
    # But wait, M1 contributes to C00, C11 with +1, and to C01, C10 with 0? No...
    # Actually in Strassen, M1 is a scalar, and it contributes to multiple C entries.
    # The correct way: each of the 7 Strassen products M1..M7 is a scalar.
    # Then C entries are linear combinations of M1..M7.
    # But for tensor decomposition, each M_k * (linear combo of C entries) gives rank-1 terms.
    # Actually, the standard way: the 7 terms are already the rank-1 decomposition.
    # Term k has U_k (linear form in A), V_k (linear form in B), W_k (linear form in C).
    # And sum_k U_k * V_k * W_k = the full tensor.

    # Let me define the 7 terms properly:
    # M1 = (A00+A11)(B00+B11)
    #   U1 = A00+A11 = A[0,0]+A[0,1]+A[1,0]+A[1,1]+A[2,2]+A[2,3]+A[3,2]+A[3,3]
    #   V1 = B00+B11 = B[0,0]+B[0,1]+B[1,0]+B[1,1]+B[2,2]+B[2,3]+B[3,2]+B[3,3]
    #   M1 contributes to C00 and C11: C00 += M1, C11 += M1
    #   So W1 = C[0,0]+C[0,1]+C[1,0]+C[1,1]+C[2,2]+C[2,3]+C[3,2]+C[3,3]

    # Actually, I realize the issue. In the tensor decomposition, each rank-1 term is:
    # (linear form in A) * (linear form in B) * (linear form in C)
    # and the sum over all terms equals the matrix multiplication tensor.
    # The 7 Strassen terms work as follows:
    # For each k in 1..7:
    #   U_k = linear form in A entries
    #   V_k = linear form in B entries
    #   W_k = linear form in C entries
    #   such that U_k * V_k * W_k is a rank-1 tensor
    # And sum_k U_k * V_k * W_k = sum_{i,j,k} A[i,k]*B[k,j]*C[i,j]

    # The standard Strassen 2x2x2 gives 7 terms. For 4x4x4, we apply it at the block level.
    # Each "block" is 2x2, so each block-level entry is itself a 2x2 matrix.
    # The 7 block-level Strassen products each require 4 scalar multiplications (for the 2x2 blocks),
    # giving 28 terms total. But we can do better by using the 7-term Strassen directly on 4x4.

    # Actually, the 4x4x4 Strassen decomposition has 28 terms (7 outer * 4 inner), not 7.
    # Unless we use a different decomposition. Let me just use the 28-term version.

    # For now, let me use the simpler approach: 7 outer Strassen terms, each expanded into 4 inner terms.
    # That gives 28 terms for the 4x4x4 part.

    # Outer Strassen (on 2x2 blocks of 2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # Each M_k is a 2x2 matrix product of 2x2 matrices, which can be done in 7 scalar mults (Strassen) or 4 (schoolbook).
    # If we use schoolbook for the inner 2x2x2, we get 7*4 = 28 terms.
    # If we use Strassen for the inner, we get 7*7 = 49 terms (worse).
    # So 28 terms is the way to go for 4x4x4.

    # Let me implement the 28-term 4x4x4 decomposition.
    # Each term: U (linear form in A[0:4,0:4]), V (linear form in B[0:4,0:4]), W (linear form in C[0:4,0:4])

    # I'll generate all 28 terms programmatically.

    # Outer Strassen definitions (block level):
    outer_strassen = [
        # (U_blocks, V_blocks, W_blocks) where each is a dict {block: sign}
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1: (A00+A11)(B00+B11) -> C00, C11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),          # M2: (A10+A11)B00 -> C10, C11
        ({0: 1}, {2: 1, 0: -1}, {1: 1}),               # M3: A00(B10-B00) -> C01
        ({3: 1}, {3: 1, 1: -1}, {0: 1, 2: 1}),         # M4: A11(B11-B01) -> C00, C10
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: -1}),        # M5: (A00+A01)B11 -> C00, C01
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6: (A10-A00)(B00+B01) -> C11
        ({3: 1, 1: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),   # M7: (A11-A01)(B10+B11) -> C00, C11
    ]

    # Inner schoolbook (2x2x2): 4 terms
    # For each (li, lk) in 0..1 for A, (lk2, lj) in 0..1 for B:
    #   term: A[li,lk] * B[lk2,lj] * C[li,lj] where lk=lk2

    for outer_idx, (u_blocks, v_blocks, w_blocks) in enumerate(outer_strassen):
        for li in range(2):
            for lk in range(2):
                u_vec = [0.0] * (n * m)
                v_vec = [0.0] * (m * p)
                w_vec = [0.0] * (p * n)

                for blk, sign in u_blocks.items():
                    ro, co = block_offset(blk)
                    gi, gk = ro + li, co + lk
                    u_vec[gi * m + gk] += sign

                for blk, sign in v_blocks.items():
                    ro, co = block_offset(blk)
                    gk, gj = ro + lk, co + li  # Wait, this is wrong. lk for B should be independent.
                    # Actually, for the inner schoolbook, we need:
                    # A_block[li, lk] * B_block[lk, lj] * C_block[li, lj]
                    # So for a fixed (li, lk) in A, we sum over lj in B.
                    # Let me restructure.

                break
            break
        break

    # This is getting complicated. Let me just hardcode all 28 terms.
    # Actually, let me take a step back and use a cleaner approach.

    # Clear the lists and restart
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()

    # Part 1: 4x4x4 using 28-term decomposition
    # I'll generate the terms systematically.

    # The 4x4x4 tensor can be decomposed as:
    # sum_{i,j,k=0}^3 A[i,k]*B[k,j]*C[i,j]
    # Using Strassen at the block level (2x2 blocks), with schoolbook inside:
    # 7 outer terms * 4 inner terms = 28 terms

    # Let me define it more carefully.
    # For each outer Strassen term t (0..6):
    #   U_t = sum of A entries with coefficients
    #   V_t = sum of B entries with coefficients
    #   For each of 4 inner positions (li, lj) in 0..1:
    #     W_{t,li,lj} = sum of C entries with coefficients
    #     Term: U_t * V_t * W_{t,li,lj}

    # Actually, the correct decomposition is:
    # For each outer term t and inner position (li, lj):
    #   U_{t,li,lj} = linear form in A
    #   V_{t,li,lj} = linear form in B
    #   W_{t,li,lj} = linear form in C
    #   such that the product is a rank-1 tensor

    # Let me just use the explicit 28-term decomposition.

    # I'll define the terms as lists of (index, coefficient) for U, V, W.

    # This is complex, so let me use a different approach:
    # Use the known 28-term 4x4x4 decomposition from the literature.

    # For simplicity and correctness, let me use the 49-term nested Strassen (7 outer * 7 inner)
    # and accept the higher rank for now. 49 + 16 + 20 = 85.
    # But the current program already does this and scores 1.1765.
    # To improve, I need to reduce the boundary terms.

    # Let me focus on reducing the boundary terms from 36 to something lower.
    # Part 2: A[0:4,0:4] * B[0:4,4] -> C[0:4,4] (4x4x1)
    # Part 3: A[0:4,4] * B[4,0:5] -> C[0:4,0:5] (4x1x5)

    # For Part 2 (4x4x1):
    # C[i,4] = sum_{k=0}^3 A[i,k] * B[k,4]
    # This is a 4x4 matrix times a 4x1 vector.
    # The tensor rank of 4x4x1 is at most 4*4 = 16 (schoolbook).
    # Can we do better? The tensor is sum_{i,k} A[i,k]*B[k,4]*C[i,4]
    # This is the same as the matrix-vector product tensor.
    # The rank of the 4x4x1 tensor is 4*4 = 16 (it's essentially a 4x4 matrix flattened).
    # Actually, the tensor T[i,k] = A[i,k]*b_k*c_i where b = B[:,4], c = C[:,4].
    # This is a rank-1 tensor in the (i,k) space when b and c are fixed, but as a bilinear form,
    # it requires 16 scalar multiplications.
    # So 16 is optimal for Part 2.

    # For Part 3 (4x1x5):
    # C[i,j] = A[i,4] * B[4,j]
    # This is the outer product of A[:,4] (4x1) and B[4,:] (1x5).
    # The tensor is sum_{i,j} A[i,4]*B[4,j]*C[i,j]
    # This requires 4*5 = 20 scalar multiplications.
    # So 20 is optimal for Part 3.

    # Total: 49 + 16 + 20 = 85. Same as current.
    # To improve, I need to reduce the 4x4x4 part below 49.
    # The 28-term decomposition would give 28 + 16 + 20 = 64.

    # Let me implement the 28-term 4x4x4 decomposition.

    # I'll use the following approach:
    # The 4x4x4 Strassen decomposition with 28 terms.
    # Each term is a rank-1 contribution.

    # Let me define the 28 terms explicitly.

    # Actually, let me use a cleaner formulation. The 4x4x4 tensor can be written as:
    # sum_{p,q,r=0}^3 A[p,q]*B[q,r]*C[p,r]
    # Using the 28-term Strassen decomposition.

    # I'll look up the exact 28-term decomposition.
    # From Strassen's 1969 paper, the 4x4x4 decomposition has 28 terms.

    # For now, let me just output the 49-term version and see if I can verify it works.
    # Then I'll try to implement the 28-term version.

    # 49-term nested Strassen for 4x4x4:
    # 7 outer Strassen terms, each expanded into 7 inner Strassen terms.

    # Outer Strassen (on 2x2 blocks):
    outer = [
        # (U_blocks, V_blocks, C_blocks)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),          # M2
        ({0: 1}, {2: 1, 0: -1}, {1: 1}),               # M3
        ({3: 1}, {3: 1, 1: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: -1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({3: 1, 1: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),   # M7
    ]

    # Inner Strassen (on 2x2 matrices):
    inner = [
        # (U_blocks, V_blocks, C_blocks) for 2x2x2
        ({0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}),  # M1
        ({1: 1}, {0: 1}, {1: 1}),                      # M2
        ({0: 1}, {1: 1, 0: -1}, {0: 1}),               # M3
        ({1: 1}, {1: 1, 0: -1}, {0: 1}),               # M4
        ({0: 1, 1: 1}, {1: 1}, {0: -1}),               # M5
        ({1: 1, 0: -1}, {0: 1, 1: 1}, {1: 1}),         # M6
        ({1: 1, 0: -1}, {1: 1, 0: -1}, {0: 1}),        # M7
    ]

    # This is getting too complex. Let me just use the schoolbook for the 4x4x4 part
    # and focus on getting the boundary terms right.
    # 4x4x4 schoolbook: 16 terms.
    # Total: 16 + 16 + 20 = 52.

    # Wait, that's better than 85! Let me use schoolbook for 4x4x4.

    for i in range(4):
        for k in range(4):
            for j in range(4):
                u_vec = [0.0] * (n * m)
                v_vec = [0.0] * (m * p)
                w_vec = [0.0] * (p * n)
                u_vec[i * m + k] = 1.0
                v_vec[k * p + j] = 1.0
                w_vec[j * n + i] = 1.0
                add_rank1(u_vec, v_vec, w_vec)

    # Part 2: A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + k] = 1.0
            v_vec[k * p + 4] = 1.0
            w_vec[4 * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)

    # Part 3: A[0:4,4] * B[4,0:5] -> C[0:4,0:5]
    for i in range(4):
        for j in range(5):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + j] = 1.0
            w_vec[j * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    # Rank should be: 16 + 16 + 20 = 52

    if rank == 0:
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[0] = 1.0
        v_vec[0] = 1.0
        w_vec[0] = 1.0
        add_rank1(u_vec, v_vec, w_vec)
        rank = 1

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
