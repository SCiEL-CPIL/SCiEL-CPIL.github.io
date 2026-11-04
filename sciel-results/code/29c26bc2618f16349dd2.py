"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Initial rank ~89, then greedy local search attempts to reduce rank.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
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
    
    # Greedy local search: try to merge pairs of terms
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert term to a flat 3D tensor representation."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return tensor
    
    def find_nonzero_ref(combined, u_indices, v_indices, w_indices):
        """Find a reference element that is non-zero."""
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    val = combined.get((i, j, k), 0)
                    if abs(val) > 1e-12:
                        return i, j, k, val
        return None
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor with integer/half-integer entries."""
        if time.time() > deadline:
            return None
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Remove near-zero entries
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            return None
        
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        # Find a non-zero reference
        ref = find_nonzero_ref(combined, u_indices, v_indices, w_indices)
        if ref is None:
            return None
        ref_i, ref_j, ref_k, ref_val = ref
        
        # Compute factors using exact arithmetic with Fractions
        from fractions import Fraction
        a = {ref_i: Fraction(1)}
        b = {ref_j: Fraction(1)}
        c = {ref_k: Fraction(1)}
        
        ref_frac = Fraction(ref_val).limit_denominator(10**6)
        if ref_frac == 0:
            return None
        
        for i in u_indices:
            if i != ref_i:
                val = combined.get((i, ref_j, ref_k), 0)
                a[i] = Fraction(val).limit_denominator(10**6) / ref_frac
        
        for j in v_indices:
            if j != ref_j:
                val = combined.get((ref_i, j, ref_k), 0)
                b[j] = Fraction(val).limit_denominator(10**6) / ref_frac
        
        for k in w_indices:
            if k != ref_k:
                val = combined.get((ref_i, ref_j, k), 0)
                c[k] = Fraction(val).limit_denominator(10**6) / ref_frac
        
        # Verify rank-1: tensor[i,j,k] = a[i]*b[j]*c[k]*ref_val
        for (i, j, k), val in combined.items():
            expected = a.get(i, Fraction(0)) * b.get(j, Fraction(0)) * c.get(k, Fraction(0)) * ref_frac
            if abs(Fraction(val).limit_denominator(10**6) - expected) > Fraction(1, 10**8):
                return None
        
        # Find common denominator to scale to integers
        from math import gcd
        all_fracs = list(a.values()) + list(b.values()) + list(c.values())
        lcm = 1
        for f in all_fracs:
            d = f.denominator
            lcm = lcm * d // gcd(lcm, d)
        
        # Scale
        a_int = {k: int(v * lcm) for k, v in a.items()}
        b_int = {k: int(v * lcm) for k, v in b.items()}
        c_int = {k: int(v * lcm) for k, v in c.items()}
        
        # Final verification with integers
        for (i, j, k), val in combined.items():
            expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
            if abs(val * lcm - expected) > 1e-6:
                return None
        
        # Check that entries are integers (they should be by construction)
        return (lcm, a_int, b_int, c_int)
    
    # Try merging pairs - multiple passes
    for pass_num in range(50):
        if time.time() > deadline - 10:
            break
        merged = False
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    _, a_new, b_new, c_new = result
                    # Remove terms i and j, add new term
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    # Remove redundant terms: if a term is zero, remove it
    new_terms = []
    for U_dict, V_dict, W_dict in terms:
        # Check if term is effectively zero
        is_zero = True
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    if abs(u_val * v_val * w_val) > 1e-12:
                        is_zero = False
                        break
                if not is_zero:
                    break
            if not is_zero:
                break
        if not is_zero:
            new_terms.append((U_dict, V_dict, W_dict))
    terms = new_terms
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            # Round to nearest integer if very close
            if abs(val - round(val)) < 1e-9:
                U[idx][r] = int(round(val))
            else:
                U[idx][r] = val
        for idx, val in V_dict.items():
            if abs(val - round(val)) < 1e-9:
                V[idx][r] = int(round(val))
            else:
                V[idx][r] = val
        for idx, val in W_dict.items():
            if abs(val - round(val)) < 1e-9:
                W[idx][r] = int(round(val))
            else:
                W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
