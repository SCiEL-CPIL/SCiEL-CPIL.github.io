import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2 columns each

    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Local A indices: 0=a, 1=b, 2=c, 3=d
    # Local B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # U coefficients for each Strassen term (A-side)
    U_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {0: -1, 2: 1},      # M6
        {1: 1, 3: -1},      # M7
    ]

    # V coefficients for each Strassen term (B-side, local indices)
    V_strassen = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]

    # W coefficients for each Strassen term (C-side, local indices)
    W_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1},             # M7
    ]

    # Output equation for each C entry in block t:
    # C[0, 2t]   = M1 + M4 - M5 + M7
    # C[0, 2t+1] = M3 + M5
    # C[1, 2t]   = M2 + M4
    # C[1, 2t+1] = M1 - M2 + M3 + M6
    #
    # W coefficient vectors for each output (indexed by Strassen term 0-6):
    w_out = {
        0: [1, 0, 0, 1, -1, 0, 1],   # C[0, 2t]
        1: [0, 0, 1, 0, 1, 0, 0],    # C[0, 2t+1]
        2: [0, 1, 0, 1, 0, 0, 0],    # C[1, 2t]
        3: [1, -1, 1, 0, 0, 1, 0],   # C[1, 2t+1]
    }

    # Map local B index to global V row
    # B is 2x12: row 0 -> cols 0-11, row 1 -> cols 12-23
    def local_b_to_global(local_b, t):
        if local_b == 0:
            return 0 * p + 2 * t        # B[0, 2t]
        elif local_b == 1:
            return 0 * p + 2 * t + 1    # B[0, 2t+1]
        elif local_b == 2:
            return 1 * p + 2 * t        # B[1, 2t]
        else:
            return 1 * p + 2 * t + 1    # B[1, 2t+1]

    # Map local C index to global W row
    # W[k*n + i] where k=col, i=row
    def local_c_to_global(local_c, t):
        if local_c == 0:
            return (2 * t) * n + 0       # C[0, 2t]
        elif local_c == 1:
            return (2 * t + 1) * n + 0   # C[0, 2t+1]
        elif local_c == 2:
            return (2 * t) * n + 1       # C[1, 2t]
        else:
            return (2 * t + 1) * n + 1   # C[1, 2t+1]

    # For a given rank r, we assign each block t a Strassen term s(t) in 0..6
    # The r terms are: (t, s(t)) for t=0..5, plus possibly extra terms
    # Actually, let's think of it as: we have r "slots", each slot is assigned
    # a Strassen term type (0-6). Multiple blocks can share the same slot.
    #
    # For a valid decomposition with rank r:
    # - We have r slots, each with a Strassen type s_r in {0,...,6}
    # - Each block t must be "covered" by some subset of slots
    # - For block t, the W equation must be satisfied:
    #   For output o (0-3): sum over slots r that "serve" block t of
    #     W_strassen[s_r][o] * (V contribution) = w_out[o]
    #
    # Wait, this is getting complicated. Let me think differently.
    #
    # Actually, the key constraint is:
    # For each block t and each output o:
    #   sum_{r: slot r serves block t} W_coeff[s_r][o] = w_out[o]
    #
    # where W_coeff[s][o] is the coefficient of output o in Strassen term s.
    #
    # But also, the V side must match: for each slot r with type s_r,
    # the V coefficients must correspond to the B entries of the block(s) it serves.
    #
    # If slot r serves block t with type s_r, then V[r] must equal
    # V_strassen[s_r] mapped to block t's B entries.
    #
    # If slot r serves multiple blocks, V[r] must be consistent with all of them.
    # Since V_strassen[s] uses local indices {0,1,2,3} which map to different
    # global indices for different blocks, a slot can only serve one block
    # unless the V coefficients happen to be compatible (which they generally aren't
    # since different blocks use different B columns).
    #
    # EXCEPTION: If V_strassen[s] only uses entries from one row of B, and
    # multiple blocks share that row pattern... but no, each block uses different columns.
    #
    # So actually, each slot can only serve ONE block. This means we need at least
    # 6*4 = 24 slots? No wait, that's not right either.
    #
    # Let me reconsider. The Strassen decomposition for block t gives:
    # C_t = sum_{s=0}^{6} M_s(t) * W_s
    # where M_s(t) = U_s(A) * V_s(B_t)
    #
    # If we want to share across blocks, we need M_s(t1) and M_s(t2) to use
    # the same scalar product. But V_s(B_t1) and V_s(B_t2) are different linear
    # forms on different B entries, so they can't be the same scalar.
    #
    # UNLESS we restructure. What if we define new scalar products that combine
    # B entries from multiple blocks?
    #
    # For example, M1 for block 0 is (a+d)(B[0,0]+B[1,1])
    # M1 for block 1 is (a+d)(B[0,2]+B[1,3])
    # These are different because they use different B entries.
    #
    # But what if we define: P1 = (a+d) * (B[0,0]+B[1,1]+B[0,2]+B[1,3]+...)
    # Then we'd need to subtract off the unwanted parts, which requires more terms.
    #
    # This suggests the rank 42 (independent Strassen per block) might be close to optimal
    # for this structure. Let me verify by trying a computational search for small cases.

    # Let's try a direct search: for rank r from 24 to 42, check if a valid
    # decomposition exists using the constraint that each slot serves exactly one block.
    #
    # If each slot serves one block, then for block t we need to choose a subset
    # S_t of slots such that:
    #   sum_{r in S_t} W_strassen[type(r)][o] = w_out[o] for o = 0,1,2,3
    # where type(r) is the Strassen type assigned to slot r.
    #
    # And the total number of slots is sum_t |S_t| = r.
    #
    # For each block, we need to find a multiset of Strassen types whose
    # W-vectors sum to the required w_out vectors.
    #
    # The W vectors for the 7 Strassen terms are:
    # s=0: [1, 0, 0, 1]
    # s=1: [0, 0, 1, -1]
    # s=2: [0, 1, 0, 1]
    # s=3: [1, 0, 1, 0]
    # s=4: [-1, 1, 0, 0]
    # s=5: [0, 0, 0, 1]
    # s=6: [1, 0, 0, 0]
    #
    # We need: sum of selected W-vectors = [1, 0, 0, 1] (for C[0,2t])
    #                                    + [0, 1, 0, 0] (for C[0,2t+1])
    #                                    + [0, 0, 1, 0] (for C[1,2t])
    #                                    + [1, -1, 1, 0] (for C[1,2t+1])
    # Wait, no. Each output has its own equation.
    #
    # For output 0 (C[0,2t]): sum_{r in S_t} W_s[r][0] = 1
    # For output 1 (C[0,2t+1]): sum_{r in S_t} W_s[r][1] = 1
    # For output 2 (C[1,2t]): sum_{r in S_t} W_s[r][2] = 1
    # For output 3 (C[1,2t+1]): sum_{r in S_t} W_s[r][3] = 1
    #
    # Where W_s[r][o] is the coefficient of output o in Strassen term s.
    #
    # Let me denote the W matrix (7 terms x 4 outputs):
    # s\o:  0  1  2  3
    #  0:   1  0  0  1
    #  1:   0  0  1 -1
    #  2:   0  1  0  1
    #  3:   1  0  1  0
    #  4:  -1  1  0  0
    #  5:   0  0  0  1
    #  6:   1  0  0  0
    #
    # We need non-negative integer counts n_0,...,n_6 (number of times each
    # Strassen type is used for this block) such that:
    # n_0*1 + n_3*1 + n_4*(-1) + n_6*1 = 1  (output 0)
    # n_2*1 + n_4*1 = 1                        (output 1)
    # n_1*1 + n_3*1 = 1                        (output 2)
    # n_0*1 + n_1*(-1) + n_2*1 + n_5*1 = 1    (output 3)
    #
    # From output 1: n_2 + n_4 = 1
    # From output 2: n_1 + n_3 = 1
    #
    # Case A: n_2=1, n_4=0
    #   Output 0: n_0 + n_3 + n_6 = 1
    #   Output 3: n_0 - n_1 + 1 + n_5 = 1 => n_0 - n_1 + n_5 = 0
    #   With n_1 + n_3 = 1:
    #     Sub-case A1: n_1=1, n_3=0
    #       Output 0: n_0 + n_6 = 1
    #       Output 3: n_0 - 1 + n_5 = 0 => n_0 + n_5 = 1
    #       So n_0 + n_6 = 1 and n_0 + n_5 = 1
    #       If n_0=0: n_6=1, n_5=1. Total: n_1+n_2+n_5+n_6 = 1+1+1+1 = 4
    #       If n_0=1: n_6=0, n_5=0. Total: n_0+n_1+n_2 = 1+1+1 = 3
    #     Sub-case A2: n_1=0, n_3=1
    #       Output 0: n_0 + 1 + n_6 = 1 => n_0 + n_6 = 0 => n_0=0, n_6=0
    #       Output 3: 0 - 0 + n_5 = 0 => n_5=0
    #       Total: n_2+n_3 = 1+1 = 2
    #
    # Case B: n_2=0, n_4=1
    #   Output 0: n_0 + n_3 - 1 + n_6 = 1 => n_0 + n_3 + n_6 = 2
    #   Output 3: n_0 - n_1 + 0 + n_5 = 1
    #   With n_1 + n_3 = 1:
    #     Sub-case B1: n_1=1, n_3=0
    #       Output 0: n_0 + n_6 = 2
    #       Output 3: n_0 - 1 + n_5 = 1 => n_0 + n_5 = 2
    #       Min total: n_0=0 => n_6=2, n_5=2. Total: n_1+n_4+n_5+n_6 = 1+1+2+2 = 6
    #       Or n_0=1 => n_6=1, n_5=1. Total: 1+1+1+1+1 = 5
    #       Or n_0=2 => n_6=0, n_5=0. Total: n_0+n_1+n_4 = 2+1+1 = 4
    #     Sub-case B2: n_1=0, n_3=1
    #       Output 0: n_0 + 1 + n_6 = 2 => n_0 + n_6 = 1
    #       Output 3: n_0 + n_5 = 1
    #       If n_0=0: n_6=1, n_5=1. Total: n_3+n_4+n_5+n_6 = 1+1+1+1 = 4
    #       If n_0=1: n_6=0, n_5=0. Total: n_0+n_3+n_4 = 1+1+1 = 3

    # Minimum per block is 2 (Case A2: n_2=1, n_3=1)
    # Let's verify: use Strassen terms 2 and 3 only.
    # W contributions:
    #   s=2: [0, 1, 0, 1]
    #   s=3: [1, 0, 1, 0]
    #   Sum: [1, 1, 1, 1]
    # But we need: output0=1, output1=1, output2=1, output3=1
    # Wait, that gives [1,1,1,1] which matches! Let me double-check.
    #
    # Actually wait, let me re-examine the W coefficients.
    # C00=M1+M4-M5+M7 -> W for output 0: M1:+1, M4:+1, M5:-1, M7:+1
    # C01=M3+M5 -> W for output 1: M3:+1, M5:+1
    # C10=M2+M4 -> W for output 2: M2:+1, M4:+1
    # C11=M1-M2+M3+M6 -> W for output 3: M1:+1, M2:-1, M3:+1, M6:+1
    #
    # So the W matrix (rows=Strassen term 0-6, cols=output 0-3):
    # s=0 (M1): [1, 0, 0, 1]
    # s=1 (M2): [0, 0, 1, -1]
    # s=2 (M3): [0, 1, 0, 1]
    # s=3 (M4): [1, 0, 1, 0]
    # s=4 (M5): [-1, 1, 0, 0]
    # s=5 (M6): [0, 0, 0, 1]
    # s=6 (M7): [1, 0, 0, 0]
    #
    # Using s=2 and s=3:
    #   Output 0: 0+1 = 1 ✓
    #   Output 1: 1+0 = 1 ✓
    #   Output 2: 0+1 = 1 ✓
    #   Output 3: 1+0 = 1 ✓
    #
    # So we can do each block with just 2 Strassen terms!
    # But wait, we also need the U and V sides to work out.
    #
    # For s=2 (M3=a*(f-h)): U={a:1}, V={f:1, h:-1}
    # For s=3 (M4=d*(g-e)): U={d:1}, V={g:1, e:-1}
    #
    # For block t, the scalar products are:
    #   P1 = a * (B[0,2t+1] - B[1,2t+1])
    #   P2 = d * (B[1,2t] - B[0,2t])
    #
    # And C[0,2t] = P2, C[0,2t+1] = P1, C[1,2t] = P2, C[1,2t+1] = P1
    #
    # But that's NOT correct! C[0,2t] should be a*B[0,2t] + b*B[1,2t], not d*(B[1,2t]-B[0,2t]).
    #
    # The issue is that the W coefficients tell us how to COMBINE the scalar products
    # to get the outputs, but the scalar products themselves must correctly compute
    # the needed quantities.
    #
    # The Strassen identity says:
    # C = sum_s M_s * w_s
    # where M_s are the 7 scalar products and w_s are the W vectors.
    #
    # If we only use 2 of the 7 terms, we're NOT computing the correct C unless
    # the other 5 terms happen to be zero, which they're not in general.
    #
    # So the W-coefficient matching is necessary but not sufficient. We also need
    # the actual values to match, which requires all 7 terms in general.
    #
    # This means rank 42 (6 blocks × 7 terms) is likely the minimum for this
    # "independent blocks" approach. Let me just implement the clean version.

    rank = num_blocks * 7  # 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s

            # U (A indices are global)
            for a_idx, val in U_strassen[s].items():
                U[a_idx][col] = val

            # V (map local B index to global)
            for local_b, val in V_strassen[s].items():
                global_b = local_b_to_global(local_b, t)
                V[global_b][col] = val

            # W (map local C index to global W row)
            for local_c, val in W_strassen[s].items():
                global_w = local_c_to_global(local_c, t)
                W[global_w][col] += val

    # The decomposition is optimal for the block structure.
    # U, V, W are lists of lists of integers.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
