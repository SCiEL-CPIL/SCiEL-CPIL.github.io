"""Tensor decomposition for 4x5 by 5x5 matrix multiplication.
Implements a rank-85 decomposition using:
  - Strassen (rank 7) for the 2x2x2 core
  - Recursion for 4x4x4 (rank 49)
  - Rank-1 terms for the remaining mixed products
"""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Block split: m1=4, m2=1; p1=4, p2=1
    m1, m2 = 4, 1
    p1, p2 = 4, 1
    
    # terms: list of (u, v, w) where
    #   u: dict {(i,j): val}  for i in [0,n), j in [0,m)
    #   v: dict {(j,k): val}  for j in [0,m), k in [0,p)
    #   w: dict {(k,i): val}  for k in [0,p), i in [0,n)
    terms = []
    
    def strassen_2x2_terms():
        """Return 7 rank-1 terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) with u, v, w dicts over {(0,0),(0,1),(1,0),(1,1)}.
        
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        terms = []
        # M1: (A00+A11)(B00+B11)
        terms.append((
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1}
        ))
        # M2: (A10+A11)B00
        terms.append((
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1}
        ))
        # M3: A00(B01-B11)
        terms.append((
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1}
        ))
        # M4: A11(B10-B00)
        terms.append((
            {(1,1):1},
            {(1,0):1, (0,0):-1},
            {(0,0):1}
        ))
        # M5: (A00+A01)B11
        terms.append((
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(1,1):-1, (0,1):1}
        ))
        # M6: (A10-A00)(B00+B01)
        terms.append((
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}
        ))
        # M7: (A01-A11)(B10+B11)
        terms.append((
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1, (1,0):1}
        ))
        return terms
    
    def strassen_4x4_terms():
        """Return 49 rank-1 terms for 4x4x4 matrix multiplication.
        Uses recursive Strassen: split 4x4 into four 2x2 blocks,
        apply 7 Strassen products, each expanded with 2x2 Strassen (7 terms).
        """
        # 2x2 Strassen terms (with signs in w for C contributions)
        s2 = strassen_2x2_terms()
        # s2[r] = (u_r, v_r, w_r)
        # The 7 products for 4x4 (each is a 2x2x2 product):
        # M1 = (A00+A11) * (B00+B11)
        # M2 = (A10+A11) * B00
        # M3 = A00 * (B01-B11)
        # M4 = A11 * (B10-B00)
        # M5 = (A00+A01) * B11
        # M6 = (A10-A00) * (B00+B01)
        # M7 = (A01-A11) * (B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each of the 7 products, we need to specify:
        #   - Which 2x2 block combination of A (with coefficients)
        #   - Which 2x2 block combination of B (with coefficients)
        #   - Which C block the result goes to (with coefficient)
        
        # The 7 products defined by (a_comb, b_comb, c_block, c_coeff):
        # a_comb: list of (block_idx, coeff) where block_idx in {(0,0),(0,1),(1,0),(1,1)}
        # b_comb: same for B
        # c_block: (row, col) of the 2x2 C block
        # c_coeff: coefficient for that C block
        
        products = [
            # M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], (0,0), 1),
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], (1,1), 1),
            # M2
            ([(1,0,1),(1,1,1)], [(0,0,1)], (1,0), 1),
            # M3
            ([(0,0,1)], [(0,1,1),(1,1,-1)], (0,1), 1),
            # M4
            ([(1,1,1)], [(1,0,1),(0,0,-1)], (0,0), 1),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], (1,0), 1),
            # M5
            ([(0,0,1),(0,1,1)], [(1,1,1)], (0,0), -1),
            ([(0,0,1),(0,1,1)], [(1,1,1)], (0,1), 1),
            # M6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], (1,1), 1),
            # M7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], (0,0), 1),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], (1,0), 1),
        ]
        
        # Wait, this approach is getting complicated. Let me use a cleaner method.
        # The 4x4x4 Strassen decomposition has exactly 7*7=49 terms.
        # Each of the 7 outer Strassen products P_l is itself a 2x2x2 product.
        # For each P_l, we decompose it into 7 rank-1 terms using 2x2 Strassen.
        # The key is: P_l = sum_r u_{l,r} * v_{l,r} * w_{l,r} where each term is a 2x2x2 rank-1.
        # But we need to lift these to 4x4x4 by mapping the 2x2 block indices to 4x4 indices.
        
        # Let me define the 7 outer products more carefully.
        # Each outer product P_l involves:
        #   - A linear combination of 2x2 A-blocks (4 blocks: A_{00}, A_{01}, A_{10}, A_{11})
        #   - A linear combination of 2x2 B-blocks
        #   - The result contributes to specific C-blocks with coefficients
        #
        # The 7 outer Strassen products:
        # P1 = (A00+A11)(B00+B11)  -> C00 += P1, C11 += P1
        # P2 = (A10+A11)B00        -> C10 += P2
        # P3 = A00(B01-B11)        -> C01 += P3
        # P4 = A11(B10-B00)        -> C00 += P4, C10 += P4
        # P5 = (A00+A01)B11        -> C00 -= P5, C01 += P5
        # P6 = (A10-A00)(B00+B01)  -> C11 += P6
        # P7 = (A01-A11)(B10+B11)  -> C00 += P7, C10 += P7
        
        # For each P_l, it's a 2x2x2 product: (linear combo of A blocks) * (linear combo of B blocks)
        # Each A block is 2x2, each B block is 2x2, result is 2x2.
        # We expand each P_l using the 7 Strassen terms for 2x2x2.
        
        # The 2x2x2 Strassen terms (s2 above) decompose:
        #   C = sum_r u_r(A) * v_r(B) * w_r(C)
        # where each term is a rank-1 in the 2x2x2 tensor.
        
        # For the outer product P_l, the "A" input is a linear combination of A-blocks,
        # and the "B" input is a linear combination of B-blocks.
        # The rank-1 terms of P_l are obtained by:
        #   For each 2x2 Strassen term r:
        #     u_{l,r} = sum over A-blocks involved in P_l of (coeff) * u_r mapped to that block
        #     v_{l,r} = sum over B-blocks involved in P_l of (coeff) * v_r mapped to that block
        #     w_{l,r} = sum over C-blocks that P_l contributes to of (coeff) * w_r mapped to that block
        
        # Let me define the mapping.
        # A 2x2 block at position (bi, bj) in A corresponds to global indices:
        #   (i,j) where i = 2*bi + local_i, j = 2*bj + local_j
        # Similarly for B and C.
        
        # The 7 outer products with their A-combos, B-combos, and C-contributions:
        outer = [
            # (a_combo, b_combo, c_contribs)
            # a_combo: list of (block_pos, coeff)
            # b_combo: list of (block_pos, coeff)
            # c_contribs: list of ((block_pos, coeff))
            (
                [((0,0), 1), ((1,1), 1)],
                [((0,0), 1), ((1,1), 1)],
                [((0,0), 1), ((1,1), 1)]
            ),  # P1
            (
                [((1,0), 1), ((1,1), 1)],
                [((0,0), 1)],
                [((1,0), 1)]
            ),  # P2
            (
                [((0,0), 1)],
                [((0,1), 1), ((1,1), -1)],
                [((0,1), 1)]
            ),  # P3
            (
                [((1,1), 1)],
                [((1,0), 1), ((0,0), -1)],
                [((0,0), 1), ((1,0), 1)]
            ),  # P4
            (
                [((0,0), 1), ((0,1), 1)],
                [((1,1), 1)],
                [((0,0), -1), ((0,1), 1)]
            ),  # P5
            (
                [((1,0), 1), ((0,0), -1)],
                [((0,0), 1), ((0,1), 1)],
                [((1,1), 1)]
            ),  # P6
            (
                [((0,1), 1), ((1,1), -1)],
                [((1,0), 1), ((1,1), 1)],
                [((0,0), 1), ((1,0), 1)]
            ),  # P7
        ]
        
        all_terms = []
        for l in range(7):
            a_combo, b_combo, c_contribs = outer[l]
            for r in range(7):
                u2, v2, w2 = s2[r]
                # Build u for 4x4
                u4 = {}
                for (bi, bj), coeff in a_combo:
                    for (li, lj), val in u2.items():
                        gi, gj = 2*bi + li, 2*bj + lj
                        u4[(gi, gj)] = u4.get((gi, gj), 0) + coeff * val
                
                v4 = {}
                for (bi, bj), coeff in b_combo:
                    for (li, lj), val in v2.items():
                        gi, gj = 2*bi + li, 2*bj + lj
                        v4[(gi, gj)] = v4.get((gi, gj), 0) + coeff * val
                
                w4 = {}
                for (ci, cj), coeff in c_contribs:
                    for (li, lj), val in w2.items():
                        # C block at (ci, cj): C[2*ci+li, 2*cj+lj]
                        # w is indexed by (k, i) where k=col, i=row
                        # So (k,i) = (2*cj+lj, 2*ci+li)
                        gk, gi = 2*cj + lj, 2*ci + li
                        w4[(gk, gi)] = w4.get((gk, gi), 0) + coeff * val
                
                all_terms.append((u4, v4, w4))
        
        return all_terms
    
    # Part 1: A11 * B11 (4x4 * 4x4 -> 4x4)
    s4 = strassen_4x4_terms()
    for u, v, w in s4:
        terms.append((u, v, w))
    
    # Part 2: A12 * B21 (4x1 * 1x4 -> 4x4)
    for i in range(4):
        for k in range(4):
            u = {(i, 4): 1}
            v = {(4, k): 1}
            w = {(k, i): 1}
            terms.append((u, v, w))
    
    # Part 3: A11 * B12 (4x4 * 4x1 -> 4x1)
    for i in range(4):
        for j in range(4):
            u = {(i, j): 1}
            v = {(j, 4): 1}
            w = {(4, i): 1}
            terms.append((u, v, w))
    
    # Part 4: A12 * B22 (4x1 * 1x1 -> 4x1)
    for i in range(4):
        u = {(i, 4): 1}
        v = {(4, 4): 1}
        w = {(4, i): 1}
        terms.append((u, v, w))
    
    # Convert to matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] += val
        for (j, k), val in v.items():
            V[j * p + k][r] += val
        for (k, i), val in w.items():
            W[k * n + i][r] += val
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
