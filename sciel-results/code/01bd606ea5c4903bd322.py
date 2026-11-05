"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Initial rank ~117, then greedy local search attempts to reduce rank.
Optimized: uses doubled-integer arithmetic for speed, precomputed tensors,
and smarter pairing order to fit more merge passes in the time budget.
"""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # We store term values as doubled integers (multiply by 2) to keep
    # half-integers as integers internally.
    terms = []  # list of (U_dict, V_dict, W_dict) with integer (doubled) values
    term_tensors = []  # precomputed flat tensor dicts for each term

    def term_to_tensor_doubled(U_dict, V_dict, W_dict):
        """Convert term to flat tensor with doubled-integer values."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    val = u_val * v_val * w_val
                    tensor[key] = tensor.get(key, 0) + val
        return tensor

    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
            term_tensors.append(term_to_tensor_doubled(U_dict, V_dict, W_dict))

    def add_strassen_2x2x2(r1, c1, k1):
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)

        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)

        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)

        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})

    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Partition: [4, 1]
    blocks = [(0, 4), (4, 5)]

    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1

                if ar == 4 and ac == 4 and bc == 4:
                    # Decompose 4x4x4 into four 2x2x2 Strassen blocks
                    for ri in range(0, 4, 2):
                        for ci in range(0, 4, 2):
                            for ki in range(0, 4, 2):
                                add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Rank-1 check using doubled-integer arithmetic
    # A tensor T (doubled) is rank-1 if there exist a[i], b[j], c[k] such that
    # T[i,j,k] = a[i]*b[j]*c[k] where a,b,c are in {0, ±1, ±2, ...} (doubled half-integers)
    # Actually we need the final values to be integer or half-integer, so in doubled
    # space they are integers. The product a[i]*b[j]*c[k] must equal T[i,j,k].
    # We can set a[i] = T[i, ref_j, ref_k] / ref_val etc. but in integer space this
    # requires divisibility. Instead, we use the standard approach:
    # Pick ref element, compute ratios, verify.
    
    def check_rank1_doubled(combined):
        """
        Check if a sparse tensor (doubled-integer values) is rank-1.
        Returns (a_dict, b_dict, c_dict) with integer values (doubled) or None.
        The condition is: combined[i,j,k] = a[i] * b[j] * c[k] / 4
        Wait, let me think about this more carefully.
        
        Original: T[i,j,k] = u_i * v_j * w_k where u,v,w are half-integers.
        Doubled: U[i] = 2*u_i, V[j] = 2*v_j, W[k] = 2*w_k (all integers)
        Then T[i,j,k] = (U[i]/2)*(V[j]/2)*(W[k]/2) = U[i]*V[j]*W[k]/8
        So doubled T: T2[i,j,k] = 8*T[i,j,k] = U[i]*V[j]*W[k]
        
        But our term_to_tensor_doubled computes u_val*v_val*w_val where u_val etc.
        are the stored (doubled) values. So term_to_tensor_doubled gives us
        U[i]*V[j]*W[k] which equals 8*T[i,j,k].
        
        When we add two terms, combined[i,j,k] = U1[i]*V1[j]*W1[k] + U2[i]*V2[j]*W2[k]
        For this to be rank-1: combined[i,j,k] = A[i]*B[j]*C[k] for some integers A,B,C.
        Then the actual decomposition would have u_i = A[i]/2, v_j = B[j]/2, w_k = C[k]/2
        and T[i,j,k] = A[i]*B[j]*C[k]/8.
        
        But we need T[i,j,k] to be representable as sum of rank-1 terms where each
        rank-1 term has half-integer coefficients. The sum of two such terms:
        T[i,j,k] = u1_i*v1_j*w1_k + u2_i*v2_j*w2_k
        In doubled space: 8*T[i,j,k] = U1[i]*V1[j]*W1[k] + U2[i]*V2[j]*W2[k]
        For the sum to be a single rank-1 term with half-integer coefficients:
        8*T[i,j,k] = A[i]*B[j]*C[k] where A,B,C are integers.
        Then u_i = A[i]/2, v_j = B[j]/2, w_k = C[k]/2 are half-integers. ✓
        
        So the check is: does there exist integer vectors A, B, C such that
        combined[i,j,k] = A[i]*B[j]*C[k] for all (i,j,k)?
        """
        if not combined:
            return None

        # Find a nonzero reference element
        ref_val = None
        ref_i = ref_j = ref_k = None
        for (i, j, k), val in combined.items():
            if val != 0:
                ref_val = val
                ref_i, ref_j, ref_k = i, j, k
                break

        if ref_val is None:
            return None

        # For rank-1: combined[i,j,k] = A[i]*B[j]*C[k]
        # A[i] = combined[i, ref_j, ref_k] / (B[ref_j]*C[ref_k])
        # This is tricky. Let's use: if combined[i,j,k] = A[i]*B[j]*C[k],
        # then combined[i,j,ref_k] * combined[ref_i,ref_j,k] = combined[i,ref_j,ref_k] * combined[ref_i,j,k]
        # (rank-1 condition: T[i,j,k]*T[i0,j0,k0] = T[i,j0,k0]*T[i0,j,k])
        
        # Simpler approach: pick ref, set A[ref_i]=1, then B[j] = combined[ref_i,j,ref_k],
        # C[k] = combined[ref_i,ref_j,k], and A[i] = combined[i,ref_j,ref_k] / (B[ref_j]*C[ref_k])
        # But B[ref_j] = combined[ref_i,ref_j,ref_k] = ref_val, C[ref_k] = ref_val.
        # So A[i] = combined[i,ref_j,ref_k] / (ref_val * ref_val)
        
        # Let's try: set A[i] = combined[i, ref_j, ref_k], B[j] = combined[ref_i, j, ref_k], C[k] = combined[ref_i, ref_j, k]
        # Then A[i]*B[j]*C[k] = combined[i,ref_j,ref_k]*combined[ref_i,j,ref_k]*combined[ref_i,ref_j,k]
        # This should equal combined[i,j,k] * ref_val^2 if rank-1.
        
        # Actually, the cleanest: if T is rank-1, T[i,j,k] = a_i * b_j * c_k.
        # Let a_i = T[i, j0, k0] / (b_j0 * c_k0). We don't know b,c.
        # Alternative: T[i,j,k] * T[i0,j0,k0] = T[i,j0,k0] * T[i0,j,k] for all i,j,k.
        # This is the rank-1 condition (all 2x2 minors of the matricization are zero).
        
        # Let's use: fix ref. Define:
        # a_i = T[i, ref_j, ref_k]
        # b_j = T[ref_i, j, ref_k]  
        # c_k = T[ref_i, ref_j, k]
        # If rank-1, then T[i,j,k] = a_i * b_j * c_k / (b_ref_j * c_ref_k) = a_i * b_j * c_k / (ref_val * ref_val)
        # So T[i,j,k] * ref_val^2 = a_i * b_j * c_k
        
        rv2 = ref_val * ref_val
        
        # Build a, b, c
        all_i = set()
        all_j = set()
        all_k = set()
        for (i, j, k) in combined.keys():
            all_i.add(i)
            all_j.add(j)
            all_k.add(k)
        
        a_dict = {}
        b_dict = {}
        c_dict = {}
        
        for i in all_i:
            ti = combined.get((i, ref_j, ref_k), 0)
            if ti != 0:
                a_dict[i] = ti
        
        for j in all_j:
            tj = combined.get((ref_i, j, ref_k), 0)
            if tj != 0:
                b_dict[j] = tj
        
        for k in all_k:
            tk = combined.get((ref_i, ref_j, k), 0)
            if tk != 0:
                c_dict[k] = tk
        
        # Verify: combined[i,j,k] * rv2 == a_dict[i] * b_dict[j] * c_dict[k]
        for (i, j, k), val in combined.items():
            lhs = val * rv2
            rhs = a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0)
            if lhs != rhs:
                return None
        
        # Now we need combined[i,j,k] = A[i]*B[j]*C[k] for integers A,B,C.
        # We have combined[i,j,k] * rv2 = a_i * b_j * c_k
        # So combined[i,j,k] = (a_i * b_j * c_k) / rv2
        # We need to find integers A,B,C with A[i]*B[j]*C[k] = combined[i,j,k].
        # This requires rv2 | a_i * b_j * c_k for all i,j,k, and the quotients to be integers.
        # 
        # One approach: factor rv2 and distribute among A, B, C.
        # But this is complex. Simpler: just check if there exist integer A,B,C.
        # 
        # Actually, since the original terms have integer (doubled) values, and we're
        # adding two such terms, the combined values are integers. For the sum to be
        # a single rank-1 term with half-integer coefficients, we need:
        # combined[i,j,k] = A[i]*B[j]*C[k] where A,B,C are integers.
        #
        # Given that combined[i,j,k]*rv2 = a_i*b_j*c_k, we can try:
        # Let g = gcd of all a_i values, etc. This is getting complex.
        #
        # Alternative simpler check: just verify that combined is rank-1 over the
        # rationals, then check if the factors can be chosen as integers.
        #
        # For rank-1 over Q: combined[i,j,k] = p_i * q_j * r_k for rationals p,q,r.
        # We showed that if rank-1, combined[i,j,k]*rv2 = a_i*b_j*c_k where a,b,c are integers.
        # So combined[i,j,k] = a_i*b_j*c_k/rv2.
        # For this to equal A_i*B_j*C_k with integers A,B,C, we need to be able to
        # write a_i*b_j*c_k/rv2 as a product of three integers.
        #
        # This is always possible if rv2 divides a_i*b_j*c_k for all i,j,k, but the
        # quotient needs to factor as A_i*B_j*C_k.
        #
        # Simplest sufficient condition: rv2 = 1 (i.e., ref_val = ±1), then
        # A_i = a_i, B_j = b_j, C_k = c_k works directly.
        #
        # More generally, let's just try: A_i = a_i, B_j = b_j, C_k = c_k, and check
        # if A_i*B_j*C_k == combined[i,j,k] for all (i,j,k). If not, try other factorizations.
        
        # Direct check with A=a, B=b, C=c:
        for (i, j, k), val in combined.items():
            if a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0) != val:
                # Try: maybe we need to scale. Check if combined is rank-1 at all.
                # We already verified combined[i,j,k]*rv2 = a_i*b_j*c_k.
                # So combined is rank-1 over Q. Now check if factors can be integers.
                # Let's try to find integer factors.
                pass
        
        # For now, use the simple case: check if a*b*c == combined directly
        # (this works when ref_val = ±1)
        direct_ok = True
        for (i, j, k), val in combined.items():
            if a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0) != val:
                direct_ok = False
                break
        
        if direct_ok:
            # Verify all values are integers (they are by construction)
            return (a_dict, b_dict, c_dict)
        
        # If not direct, try to find integer factorization.
        # combined[i,j,k] = a_i*b_j*c_k / rv2
        # We need A_i*B_j*C_k = combined[i,j,k]
        # One approach: factor rv2 into rv2 = f1 * f2 * f3 and set
        # A_i = a_i/f1, B_j = b_j/f2, C_k = c_k/f3, requiring divisibility.
        # This is complex. For now, skip non-direct cases.
        return None

    def can_merge(idx1, idx2):
        """Check if terms[idx1] + terms[idx2] is a rank-1 tensor."""
        if time.time() > deadline:
            return None

        t1 = term_tensors[idx1]
        t2 = term_tensors[idx2]
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v

        # Remove zeros
        combined = {k: v for k, v in combined.items() if v != 0}

        return check_rank1_doubled(combined)

    # Try merging pairs - multiple passes with time limit
    # Use support-size ordering: terms with smaller support are more likely to merge
    # Sort terms by support size initially
    if len(terms) > 0:
        # Sort by number of nonzero entries in tensor
        order = sorted(range(len(terms)), key=lambda i: len(term_tensors[i]))
        terms = [terms[i] for i in order]
        term_tensors = [term_tensors[i] for i in order]

    max_passes = 200
    for pass_num in range(max_passes):
        if time.time() > deadline - 10:
            break
        merged = False
        i = 0
        n_terms = len(terms)
        while i < n_terms and time.time() < deadline - 8:
            j = i + 1
            while j < n_terms and time.time() < deadline - 8:
                result = can_merge(i, j)
                if result is not None:
                    a_new, b_new, c_new = result
                    # Convert doubled-integer values to actual half-integer values
                    def doubled_to_val(d):
                        if d % 2 == 0:
                            return d // 2
                        else:
                            return d / 2.0

                    a_conv = {k: doubled_to_val(v) for k, v in a_new.items() if v != 0}
                    b_conv = {k: doubled_to_val(v) for k, v in b_new.items() if v != 0}
                    c_conv = {k: doubled_to_val(v) for k, v in c_new.items() if v != 0}

                    # Remove terms i and j, add new term
                    terms.pop(j)
                    terms.pop(i)
                    term_tensors.pop(j)
                    term_tensors.pop(i)
                    terms.append((a_conv, b_conv, c_conv))
                    term_tensors.append(term_to_tensor_doubled(
                        {k: int(round(v * 2)) for k, v in a_conv.items()},
                        {k: int(round(v * 2)) for k, v in b_conv.items()},
                        {k: int(round(v * 2)) for k, v in c_conv.items()}
                    ))
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1

    rank = len(terms)

    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
