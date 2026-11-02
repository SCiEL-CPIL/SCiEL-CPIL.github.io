import time
import sys
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Build the standard rank-28 block-diagonal Strassen solution
    def build_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
            a11 = A_r0 * m + A_c0
            a12 = A_r0 * m + A_c0 + 1
            a21 = (A_r0 + 1) * m + A_c0
            a22 = (A_r0 + 1) * m + A_c0 + 1
            b11 = B_r0 * p + B_c0
            b12 = B_r0 * p + B_c0 + 1
            b21 = (B_r0 + 1) * p + B_c0
            b22 = (B_r0 + 1) * p + B_c0 + 1
            c11_idx = C_c0 * n + C_r0
            c12_idx = (C_c0 + 1) * n + C_r0
            c21_idx = C_c0 * n + (C_r0 + 1)
            c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
            r = start_r
            # M1 = (A11+A22)(B11+B22)
            U[a11][r] += 1; U[a22][r] += 1; V[b11][r] += 1; V[b22][r] += 1
            W[c11_idx][r] += 1; W[c22_idx][r] += 1; r += 1
            # M2 = (A21+A22)B11
            U[a21][r] += 1; U[a22][r] += 1; V[b11][r] += 1
            W[c21_idx][r] += 1; W[c22_idx][r] += -1; r += 1
            # M3 = A11(B12-B22)
            U[a11][r] += 1; V[b12][r] += 1; V[b22][r] += -1
            W[c12_idx][r] += 1; W[c22_idx][r] += 1; r += 1
            # M4 = A22(B21-B11)
            U[a22][r] += 1; V[b21][r] += 1; V[b11][r] += -1
            W[c11_idx][r] += 1; W[c21_idx][r] += 1; r += 1
            # M5 = (A11+A12)B22
            U[a11][r] += 1; U[a12][r] += 1; V[b22][r] += 1
            W[c11_idx][r] += -1; W[c12_idx][r] += 1; r += 1
            # M6 = (A21-A11)(B11+B12)
            U[a21][r] += 1; U[a11][r] += -1; V[b11][r] += 1; V[b12][r] += 1
            W[c22_idx][r] += 1; r += 1
            # M7 = (A12-A22)(B21+B22)
            U[a12][r] += 1; U[a22][r] += -1; V[b21][r] += 1; V[b22][r] += 1
            W[c11_idx][r] += 1; r += 1
            return r
        
        r = add_strassen(0, 0, 0, 0, 0, 0, 0)
        r = add_strassen(0, 2, 2, 0, 0, 0, r)
        r = add_strassen(0, 0, 0, 2, 0, 2, r)
        r = add_strassen(0, 2, 2, 2, 0, 2, r)
        assert r == 28
        return U, V, W
    
    U28, V28, W28 = build_rank28()
    
    # Verify rank-28 solution
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    expected = 1
                    actual = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        a_idx = i * m + j
                        b_idx = j2 * p + k2
                        c_idx = k2 * n + i
                        expected = 0
                        actual = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                        if abs(actual) > 1e-9:
                            return False
        return True
    
    assert verify(U28, V28, W28, 28), "Rank-28 verification failed"
    
    # Get individual Strassen terms
    def get_strassen_terms(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        terms = []
        terms.append(({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11_idx: 1, c22_idx: 1}))
        terms.append(({a21: 1, a22: 1}, {b11: 1}, {c21_idx: 1, c22_idx: -1}))
        terms.append(({a11: 1}, {b12: 1, b22: -1}, {c12_idx: 1, c22_idx: 1}))
        terms.append(({a22: 1}, {b21: 1, b11: -1}, {c11_idx: 1, c21_idx: 1}))
        terms.append(({a11: 1, a12: 1}, {b22: 1}, {c11_idx: -1, c12_idx: 1}))
        terms.append(({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22_idx: 1}))
        terms.append(({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11_idx: 1}))
        return terms
    
    P1_terms = get_strassen_terms(0, 0, 0, 0, 0, 0)
    P2_terms = get_strassen_terms(0, 2, 2, 0, 0, 0)
    P3_terms = get_strassen_terms(0, 0, 0, 2, 0, 2)
    P4_terms = get_strassen_terms(0, 2, 2, 2, 0, 2)
    
    # First: try the original block-merge approach (fast, known to work or not)
    def try_merge(p1_terms, p2_terms, p1_idx, p2_idx):
        U1, V1, W1 = p1_terms[p1_idx]
        U2, V2, W2 = p2_terms[p2_idx]
        
        support = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    support.append((a_idx, b_idx, c_idx))
        
        a_list = sorted(set(a for a, b, c in support))
        b_list = sorted(set(b for a, b, c in support))
        c_list = sorted(set(c for a, b, c in support))
        
        R = {}
        for a, b, c in support:
            val = U1.get(a, 0) * V1.get(b, 0) * W1.get(c, 0)
            val += U2.get(a, 0) * V2.get(b, 0) * W2.get(c, 0)
            R[(a, b, c)] = val
        
        pivot = None
        for key in R:
            if R[key] != 0:
                pivot = key
                break
        
        if pivot is None:
            return ({}, {}, {})
        
        pa, pb, pc = pivot
        R_pivot = R[pivot]
        
        S_U = {}
        S_V = {}
        S_W = {}
        
        for a in a_list:
            if (a, pb, pc) in R:
                S_U[a] = R[(a, pb, pc)] / R_pivot
            else:
                S_U[a] = 0
        for b in b_list:
            if (pa, b, pc) in R:
                S_V[b] = R[(pa, b, pc)] / R_pivot
            else:
                S_V[b] = 0
        for c in c_list:
            if (pa, pb, c) in R:
                S_W[c] = R[(pa, pb, c)] / R_pivot
            else:
                S_W[c] = 0
        
        for a, b, c in support:
            expected = R.get((a, b, c), 0)
            actual = S_U.get(a, 0) * S_V.get(b, 0) * S_W.get(c, 0)
            if abs(expected - actual) > 1e-9:
                return None
        
        S_U = {k: v for k, v in S_U.items() if abs(v) > 1e-12}
        S_V = {k: v for k, v in S_V.items() if abs(v) > 1e-12}
        S_W = {k: v for k, v in S_W.items() if abs(v) > 1e-12}
        
        return (S_U, S_V, S_W)
    
    best_result = None
    
    for pair_name, (t1_list, t2_list) in [("C_L", (P1_terms, P2_terms)), ("C_R", (P3_terms, P4_terms))]:
        for i in range(7):
            for j in range(7):
                if time.time() > deadline:
                    break
                shared = try_merge(t1_list, t2_list, i, j)
                if shared is None:
                    continue
                SU, SV, SW = shared
                if not SU or not SV or not SW:
                    continue
                
                rank = 27
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                r = 0
                for pi in range(7):
                    if pi == i:
                        continue
                    Ud, Vd, Wd = t1_list[pi]
                    for a, val in Ud.items():
                        U[a][r] += val
                    for b, val in Vd.items():
                        V[b][r] += val
                    for c, val in Wd.items():
                        W[c][r] += val
                    r += 1
                for pj in range(7):
                    if pj == j:
                        continue
                    Ud, Vd, Wd = t2_list[pj]
                    for a, val in Ud.items():
                        U[a][r] += val
                    for b, val in Vd.items():
                        V[b][r] += val
                    for c, val in Wd.items():
                        W[c][r] += val
                    r += 1
                for terms in [P3_terms, P4_terms]:
                    for Ud, Vd, Wd in terms:
                        for a, val in Ud.items():
                            U[a][r] += val
                        for b, val in Vd.items():
                            V[b][r] += val
                        for c, val in Wd.items():
                            W[c][r] += val
                        r += 1
                for a, val in SU.items():
                    U[a][r] += val
                for b, val in SV.items():
                    V[b][r] += val
                for c, val in SW.items():
                    W[c][r] += val
                r += 1
                
                assert r == 27, f"Expected 27, got {r}"
                
                if verify(U, V, W, 27):
                    best_result = (U, V, W)
                    break
            if best_result is not None:
                break
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # New: Randomized algebraic solver for rank-27
    # Strategy: Take the rank-28 solution and try to find a rank-27 by
    # solving a system where we perturb one term and solve for corrections.
    
    # The matmul tensor T has entries T[a,b,c] = 1 iff (a,b,c) is a valid triple.
    # A rank-r decomposition: T[a,b,c] = sum_{r=0}^{r-1} U[a,r]*V[b,r]*W[c,r]
    # 
    # For rank 27: we have 27 terms. Let's use a "structured perturbation" approach:
    # Start with rank-28, identify two terms that could be merged, and solve
    # for a single term that replaces them.
    
    # More general approach: Use the fact that if a rank-27 decomposition exists,
    # then for any choice of 27 of the 28 rank-28 terms, we can check if the
    # remaining "residual" is rank-1. But we already tried that.
    
    # Alternative: Use a different structural ansatz. Consider that the 2x4x4
    # matmul can be seen as two 2x2x2 matmuls (left and right halves of B)
    # sharing the A matrix. The key insight is that A's rows can be reused.
    
    # Let's try: A has 2 rows. Row 0 is used in products with B columns 0,1 and 2,3.
    # Row 1 is used in products with B columns 0,1 and 2,3.
    # 
    # Consider a decomposition where we use shared "row-sum" terms.
    # For 2x2x2: M1=(a11+a22)(b11+b22), etc.
    # For 2x4x4, we have 4 such 2x2 sub-products but they share A rows.
    
    # Let me try a different structural approach:
    # Write A = [A_L | A_R] where A_L is 2x2, A_R is 2x2
    # B = [[B_11, B_12], [B_21, B_22]] where each block is 2x2
    # C = A_L @ B_11 + A_R @ B_21  (left 2 cols)
    # C = A_L @ B_12 + A_R @ B_22  (right 2 cols)
    
    # The key: A_L and A_R are different 2x2 matrices, so we can't directly share.
    # But what if we use linear combinations?
    
    # Let's try: define s = A_L + A_R, d = A_L - A_R (each 2x2)
    # Then A_L = (s+d)/2, A_R = (s-d)/2
    # C_L = (s+d)/2 @ B_11 + (s-d)/2 @ B_21 = s@(B_11+B_21)/2 + d@(B_11-B_21)/2
    # C_R = (s+d)/2 @ B_12 + (s-d)/2 @ B_22 = s@(B_12+B_22)/2 + d@(B_12-B_22)/2
    
    # Now we need to compute:
    # s @ (B_11+B_21)/2, d @ (B_11-B_21)/2, s @ (B_12+B_22)/2, d @ (B_12-B_22)/2
    # Each is a 2x2 @ 2x2 product. Using Strassen, each takes 7 multiplications.
    # Total: 4 * 7 = 28. Same as before. No gain.
    
    # What if we share more? The four products are:
    # s @ X1, d @ X2, s @ X3, d @ X4
    # where X1=(B_11+B_21)/2, X2=(B_11-B_21)/2, X3=(B_12+B_22)/2, X4=(B_12-B_22)/2
    
    # For the "s" products: s @ X1 and s @ X3 share the same left factor s.
    # For the "d" products: d @ X2 and d @ X4 share the same left factor d.
    
    # Can we compute s@X1 and s@X3 with fewer than 14 multiplications?
    # This is the problem of computing two 2x2 products sharing the left factor.
    # That's equivalent to computing s @ [X1 | X3] which is 2x2 @ 2x4.
    # The rank of 2x2x4 matmul... 
    # Schoolbook: 2*2*4 = 16
    # Strassen-like: treat as two 2x2x2 products sharing left factor.
    # M1 = (s11+s22)(x11+x31+x21+x41)? No, that's not right.
    
    # Actually, for s @ [X1 X3] where s is 2x2 and [X1 X3] is 2x4:
    # Result is 2x4. 
    # We can write: s @ [X1 X3] = [s@X1, s@X3]
    # Using the standard 2x2x2 Strassen for each: 7+7=14.
    # But maybe we can do better by exploiting the shared left factor.
    
    # Let s = [[s11, s12], [s21, s22]]
    # [X1 X3] = [[x11, x12, x31, x32], [x21, x22, x41, x42]]
    # Result:
    # r11 = s11*x11 + s12*x21
    # r12 = s11*x12 + s12*x22
    # r13 = s11*x31 + s12*x41
    # r14 = s11*x32 + s12*x42
    # r21 = s21*x11 + s22*x21
    # r22 = s21*x12 + s22*x22
    # r23 = s21*x31 + s22*x41
    # r24 = s21*x32 + s22*x42
    
    # We need 8 outputs. Each is a sum of 2 products.
    # Total bilinear terms: 16. But with sharing:
    # s11 appears in r11,r12,r13,r14 (4 times)
    # s12 appears in r11,r12,r13,r14 (4 times)
    # s21 appears in r21,r22,r23,r24 (4 times)
    # s22 appears in r21,r22,r23,r24 (4 times)
    
    # Can we factor this? 
    # r11+r13 = s11*(x11+x31) + s12*(x21+x41)
    # r12+r14 = s11*(x12+x32) + s12*(x22+x42)
    # r21+r23 = s21*(x11+x31) + s22*(x21+x41)
    # r22+r24 = s21*(x12+x32) + s22*(x22+x42)
    
    # Let u1 = x11+x31, u2 = x12+x32, u3 = x21+x41, u4 = x22+x42
    # r11+r13 = s11*u1 + s12*u3
    # r12+r14 = s11*u2 + s12*u4
    # r21+r23 = s21*u1 + s22*u3
    # r22+r24 = s21*u2 + s22*u4
    
    # Also: r11-r13 = s11*(x11-x31) + s12*(x21-x41)
    # etc.
    
    # This is getting complex. Let me try a direct algebraic approach.
    
    # For the 2x2x4 problem (left factor shared):
    # We need to compute s @ M where s is 2x2 and M is 2x4.
    # The tensor is 2x2x4. What's its rank?
    # Lower bound: max(2, 2*4) = 8? No, lower bound is max(n, m*p) for n x m @ m x p.
    # Actually the tensor rank lower bound for n x m @ m x p is max(n*p, m) ... no.
    # The standard lower bound is max(n, p) * m ... no.
    # Actually, the trivial lower bound is max(n*p, m) is wrong.
    # The correct lower bound: the tensor has "slice rank" considerations.
    # For 2x2x4: we can view it as 2 slices of 2x4, or 4 slices of 2x2.
    # Each 2x2 slice has rank at most 4 (schoolbook).
    # Total rank >= ... hmm.
    
    # Let me just try: can we do 2x2x4 with rank < 14?
    # Schoolbook: 2*2*4 = 16.
    # Strassen per 2x2: 7 each, total 14.
    # Can we do better?
    
    # Actually, let me think about this differently. The full problem is 2x4x4.
    # Known results: 
    # - Schoolbook: 32
    # - Block Strassen: 28
    # - Is 27 achievable? 
    
    # Let me try a direct search with a smarter parameterization.
    
    # Approach: Assume the rank-27 decomposition has a specific structure.
    # The 28 rank-28 terms come in 4 groups of 7 (one per 2x2x2 sub-product).
    # For rank 27, we need to "merge" one term across groups.
    
    # Let me try: take the rank-28 solution and try to find a linear dependence
    # among the 28 terms that allows us to eliminate one.
    
    # Actually, a better approach: use the fact that the problem has symmetry.
    # The 4 sub-products are:
    # P1: A_L @ B_11 -> C_L (rows 0-1 of A, cols 0-1 of B, cols 0-1 of C)
    # P2: A_R @ B_21 -> C_L (rows 0-1 of A, cols 2-3 of B, cols 0-1 of C)
    # P3: A_L @ B_12 -> C_R (rows 0-1 of A, cols 0-1 of B, cols 2-3 of C)
    # P4: A_R @ B_22 -> C_R (rows 0-1 of A, cols 2-3 of B, cols 2-3 of C)
    
    # P1 and P3 share the same A (A_L) and same B columns (0-1).
    # P2 and P4 share the same A (A_R) and same B columns (2-3).
    # P1 and P2 share the same C output (C_L).
    # P3 and P4 share the same C output (C_R).
    
    # The most promising merge is between terms that share A and B indices
    # but write to different C outputs, or share A and C but different B.
    
    # Let me try: merge a term from P1 with a term from P3 (same A_L, same B cols 0-1, different C).
    # And/or merge a term from P2 with a term from P4.
    
    # Actually, the original code tried P1/P2 (same C) and P3/P4 (same C).
    # Let me also try P1/P3 (same A, same B cols) and P2/P4.
    
    all_pairs = [
        (P1_terms, P2_terms, "P1P2"),  # same C_L
        (P3_terms, P4_terms, "P3P4"),  # same C_R
        (P1_terms, P3_terms, "P1P3"),  # same A_L, same B cols 0-1
        (P2_terms, P4_terms, "P2P4"),  # same A_R, same B cols 2-3
    ]
    
    for t1_list, t2_list, pair_name in all_pairs:
        if best_result is not None:
            break
        for i in range(7):
            for j in range(7):
                if time.time() > deadline:
                    break
                shared = try_merge(t1_list, t2_list, i, j)
                if shared is None:
                    continue
                SU, SV, SW = shared
                if not SU or not SV or not SW:
                    continue
                
                rank = 27
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                r = 0
                for pi in range(7):
                    if pi == i:
                        continue
                    Ud, Vd, Wd = t1_list[pi]
                    for a, val in Ud.items():
                        U[a][r] += val
                    for b, val in Vd.items():
                        V[b][r] += val
                    for c, val in Wd.items():
                        W[c][r] += val
                    r += 1
                for pj in range(7):
                    if pj == j:
                        continue
                    Ud, Vd, Wd = t2_list[pj]
                    for a, val in Ud.items():
                        U[a][r] += val
                    for b, val in Vd.items():
                        V[b][r] += val
                    for c, val in Wd.items():
                        W[c][r] += val
                    r += 1
                # Add the other two products
                other_products = []
                if pair_name == "P1P2":
                    other_products = [P3_terms, P4_terms]
                elif pair_name == "P3P4":
                    other_products = [P1_terms, P2_terms]
                elif pair_name == "P1P3":
                    other_products = [P2_terms, P4_terms]
                elif pair_name == "P2P4":
                    other_products = [P1_terms, P3_terms]
                for terms in other_products:
                    for Ud, Vd, Wd in terms:
                        for a, val in Ud.items():
                            U[a][r] += val
                        for b, val in Vd.items():
                            V[b][r] += val
                        for c, val in Wd.items():
                            W[c][r] += val
                        r += 1
                for a, val in SU.items():
                    U[a][r] += val
                for b, val in SV.items():
                    V[b][r] += val
                for c, val in SW.items():
                    W[c][r] += val
                r += 1
                
                assert r == 27, f"Expected 27, got {r}"
                
                if verify(U, V, W, 27):
                    best_result = (U, V, W)
                    break
            if best_result is not None:
                break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Try a more general algebraic approach:
    # Assume rank-27 with a specific "almost Strassen" structure.
    # Use 27 terms where 26 are standard Strassen terms and 1 is a
    # "correction" term that fixes the mismatch.
    
    # Strategy: Take all 28 Strassen terms. Remove one term from each of
    # two products (say P1 term i and P3 term j). The residual is a
    # sum of two rank-1 tensors. If their sum is rank-1, we're done.
    # We already tried this. Let's try removing one term and adding a
    # general rank-1 correction.
    
    # Actually, let's try a completely different approach:
    # Use the "bilinear complexity" lower bound. For 2x4x4, the rank
    # is at least max(2,4,4)*... no. The standard lower bound is:
    # rank >= max(n*p, m) is not right.
    # Actually, the trivial lower bound is max(n, m, p) which is 4.
    # A better bound: rank >= n*p = 8 (from the output dimension).
    # And rank >= m = 4.
    # So rank >= 8. But we know rank <= 28.
    
    # Let me try yet another approach: use a random linear combination
    # to test if rank 27 is achievable.
    
    # The tensor T is in R^{8 x 16 x 8} (n*m=8, m*p=16, p*n=8).
    # A rank-r decomposition has r * (8 + 16 + 8) = 32r parameters.
    # The tensor has 8*16*8 = 1024 entries.
    # For rank 27: 32*27 = 864 parameters < 1024 entries.
    # So the system is overdetermined, but the tensor is special (matmul).
    
    # Let me try: use the rank-28 solution as a starting point and
    # do a "term elimination" search.
    
    # For each pair of terms (i, j) from the 28, check if removing both
    # and adding one new term works. This is what we've been doing.
    
    # Let me try: remove one term and see if the residual can be
    # absorbed into the remaining 27 terms by modifying them.
    
    # This is a harder problem. Let's try a different structural idea.
    
    # Idea: Use the fact that for 2x2x2, Strassen uses 7 terms.
    # For 2x4x4, if we could use a "shared Strassen" structure...
    
    # Let me try: define the 27 terms explicitly with a parameterized
    # structure and solve for the parameters.
    
    # The 2x4x4 matmul:
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i,k] for i in {0,1}, k in {0,1,2,3}
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split j into {0,1} and {2,3}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Let's define:
    # a0 = A[0,0], a1 = A[0,1], a2 = A[0,2], a3 = A[0,3]
    # b0 = A[1,0], b1 = A[1,1], b2 = A[1,2], b3 = A[1,3]
    # (Note: A has only 2 rows, so these are all the A entries)
    
    # B entries: B[j,k] for j,k in {0,1,2,3}
    # 16 entries.
    
    # C entries: C[0,k] and C[1,k] for k in {0,1,2,3}
    # 8 entries.
    
    # Each C[i,k] is a bilinear form in the 8 A entries and 16 B entries.
    
    # Total bilinear terms needed: 8 * 4 = 32 (schoolbook).
    # Strassen reduces to 28.
    
    # Can we do 27? Let me try a direct algebraic search.
    
    # Parameterize 27 terms. Each term is a triple (u, v, w) where
    # u is in R^8, v is in R^16, w is in R^8.
    # Total parameters: 27 * (8 + 16 + 8) = 27 * 32 = 864.
    # Constraints: 8 * 16 * 8 = 1024 tensor entries must match.
    # But many entries are zero (only 8*4 = 32 are non-zero in the matmul).
    # Actually, the matmul tensor T[a,b,c] is non-zero only when
    # a = i*m+j, b = j*p+k, c = k*n+i for consistent i,j,k.
    # There are n*m*p = 2*4*4 = 32 such entries, all equal to 1.
    # The remaining 1024 - 32 = 992 entries must be 0.
    
    # So we have 32 equations = 1 and 992 equations = 0, total 1024 equations
    # in 864 unknowns. Overdetermined by 160.
    
    # But the equations are polynomial (bilinear in u,v,w for each term,
    # summed over terms). This is a system of polynomial equations.
    
    # For a random generic tensor, rank 27 would be impossible.
    # But the matmul tensor has special structure.
    
    # Let me try: use the rank-28 solution and try to find a rank-27
    # by solving a smaller system.
    
    # Approach: Fix 26 of the 27 terms to be close to the rank-28 terms,
    # and solve for the 27th term.
    
    # Actually, let me try a specific structural ansatz for rank 27.
    
    # Based on the analysis above, let me try:
    # Use the s/d decomposition:
    # s = A_L + A_R, d = A_L - A_R
    # X1 = (B_11 + B_21)/2, X2 = (B_11 - B_21)/2
    # X3 = (B_12 + B_22)/2, X4 = (B_12 - B_22)/2
    # C_L = s @ X1 + d @ X2
    # C_R = s @ X3 + d @ X4
    
    # Now, for s @ X1 and s @ X3 (shared left factor s):
    # Use a "shared Strassen" for 2x2x4.
    # 
    # s = [[s11, s12], [s21, s22]]
    # [X1 X3] = [[x11, x12, x31, x32], [x21, x22, x41, x42]]
    # 
    # We need:
    # r11 = s11*x11 + s12*x21
    # r12 = s11*x12 + s12*x22
    # r13 = s11*x31 + s12*x41
    # r14 = s11*x32 + s12*x42
    # r21 = s21*x11 + s22*x21
    # r22 = s21*x12 + s22*x22
    # r23 = s21*x31 + s22*x41
    # r24 = s21*x32 + s22*x42
    
    # Standard: 8 products (schoolbook for 2x2x4).
    # Can we do 7? Let's try.
    
    # Define:
    # m1 = (s11+s22)*(x11+x31)  -- covers r11, r21 partially
    # m2 = (s21+s22)*(x11+x31)  -- covers r21 partially
    # ... this is getting complicated.
    
    # Let me just try the direct algebraic approach with a time limit.
    
    # Use a "term replacement" search:
    # For each of the 28 rank-28 terms, try to remove it and find a
    # replacement that's a linear combination of the remaining terms
    # plus one new term.
    
    # Actually, the simplest thing to try: just return rank 28.
    # The improvement from 28 to 27 would be significant (1.143 -> 1.185),
    # but it may not be achievable with the current approach.
    
    return {"U": U28, "V": V28, "W": W28}
