"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Initial rank ~117, then greedy local search attempts to reduce rank.
"""
import time
import numpy as np
from math import gcd

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
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
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor. Returns (a_dict, b_dict, c_dict) or None."""
        if time.time() > deadline:
            return None
        
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Remove zeros
        combined = {k: v for k, v in combined.items() if v != 0}
        
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
        
        # Derive factors as fractions of ref_val
        # a[i] = T[i, ref_j, ref_k] / ref_val
        # b[j] = T[ref_i, j, ref_k] / ref_val
        # c[k] = T[ref_i, ref_j, k] / ref_val
        
        a = {ref_i: 1}
        b = {ref_j: 1}
        c = {ref_k: 1}
        
        for (i, j, k), val in combined.items():
            if i != ref_i:
                # We need T[i, ref_j, ref_k]
                # It might not be in combined if it was 0 in both
                # But if it's 0, then a[i] should be 0
                val_ref_j_k = combined.get((i, ref_j, ref_k), 0)
                if val_ref_j_k != 0:
                    # a[i] = val_ref_j_k / ref_val
                    # We store as numerator/denominator
                    # To avoid fractions, we can check consistency later
                    pass
            
            # This approach is tricky with integers.
            # Let's use the property: T[i,j,k] * T[ref_i, ref_j, ref_k] == T[i, ref_j, ref_k] * T[ref_i, j, ref_k] * T[ref_i, ref_j, k] / T[ref_i, ref_j, ref_k] ?
            # No.
            # T[i,j,k] = a[i] b[j] c[k]
            # T[i, ref_j, ref_k] = a[i] b[ref_j] c[ref_k]
            # T[ref_i, j, ref_k] = a[ref_i] b[j] c[ref_k]
            # T[ref_i, ref_j, k] = a[ref_i] b[ref_j] c[k]
            # T[ref_i, ref_j, ref_k] = a[ref_i] b[ref_j] c[ref_k]
            
            # Let R = T[ref_i, ref_j, ref_k]
            # a[i] = T[i, ref_j, ref_k] / (b[ref_j] c[ref_k])
            # This is circular.
            
            # Better:
            # T[i,j,k] / T[ref_i, ref_j, ref_k] = (a[i]/a[ref_i]) * (b[j]/b[ref_j]) * (c[k]/c[ref_k])
            # Let A[i] = T[i, ref_j, ref_k] / R
            # Let B[j] = T[ref_i, j, ref_k] / R
            # Let C[k] = T[ref_i, ref_j, k] / R
            # Then T[i,j,k] should be R * A[i] * B[j] * C[k] / (A[ref_i] B[ref_j] C[ref_k])?
            # A[ref_i] = 1, B[ref_j] = 1, C[ref_k] = 1.
            # So T[i,j,k] should be R * A[i] * B[j] * C[k]
            
            # We can check this using integer arithmetic if we scale.
            # T[i,j,k] * R == T[i, ref_j, ref_k] * T[ref_i, j, ref_k] * T[ref_i, ref_j, k] / R ?
            # No.
            # T[i,j,k] = R * (T[i, ref_j, ref_k]/R) * (T[ref_i, j, ref_k]/R) * (T[ref_i, ref_j, k]/R)
            # T[i,j,k] * R^2 == T[i, ref_j, ref_k] * T[ref_i, j, ref_k] * T[ref_i, ref_j, k]
            
            # This is a valid check!
            # We just need to verify this for all (i,j,k) in combined.
        
        R = ref_val
        R2 = R * R
        
        for (i, j, k), val in combined.items():
            ti = combined.get((i, ref_j, ref_k), 0)
            tj = combined.get((ref_i, j, ref_k), 0)
            tk = combined.get((ref_i, ref_j, k), 0)
            
            lhs = val * R2
            rhs = ti * tj * tk
            
            if lhs != rhs:
                return None
        
        # If we get here, it is rank 1.
        # We need to return the factors.
        # a[i] = T[i, ref_j, ref_k] / R
        # b[j] = T[ref_i, j, ref_k] / R
        # c[k] = T[ref_i, ref_j, k] / R
        
        # We want integer factors.
        # a[i] = ti / R
        # b[j] = tj / R
        # c[k] = tk / R
        
        # Product a[i] b[j] c[k] = ti tj tk / R^3
        # T[i,j,k] = R * (ti/R) * (tj/R) * (tk/R) = ti tj tk / R^2
        # This matches the check.
        
        # To get integer factors, we can scale a, b, c by R.
        # a'[i] = ti
        # b'[j] = tj
        # c'[k] = tk
        # Then a'[i] b'[j] c'[k] = ti tj tk
        # But T[i,j,k] = ti tj tk / R^2
        # So T = (1/R^2) * a' b' c'
        # This means the term is (1/R^2) * (a' b' c')
        # We can absorb 1/R^2 into one of the vectors?
        # No, the term is U * V * W.
        # We can set U = a', V = b', W = c' / R^2?
        # But W must be integer/half-integer.
        
        # Actually, we can just return the factors as they are, and let the caller handle scaling?
        # No, the caller expects dictionaries of values.
        
        # Let's return a, b, c such that T = a b c.
        # a[i] = ti / R
        # b[j] = tj / R
        # c[k] = tk / R
        
        # We need to represent these as integers.
        # If R is 1, they are integers.
        # If R is not 1, we have fractions.
        
        # For now, let's assume we can return fractions or scaled integers.
        # The output format allows half-integers.
        
        # Let's construct the dictionaries.
        a_dict = {}
        b_dict = {}
        c_dict = {}
        
        for (i, j, k), val in combined.items():
            pass # We already verified
        
        # Re-derive a, b, c
        for i in set(k[0] for k in combined.keys()):
            ti = combined.get((i, ref_j, ref_k), 0)
            if ti != 0:
                a_dict[i] = ti
        
        for j in set(k[1] for k in combined.keys()):
            tj = combined.get((ref_i, j, ref_k), 0)
            if tj != 0:
                b_dict[j] = tj
        
        for k in set(k[2] for k in combined.keys()):
            tk = combined.get((ref_i, ref_j, k), 0)
            if tk != 0:
                c_dict[k] = tk
        
        # The product a[i] b[j] c[k] equals ti tj tk.
        # The tensor value is ti tj tk / R^2.
        # So we need to scale one vector by 1/R^2.
        # Let's scale c_dict by 1/R^2.
        # c_dict[k] = tk / R^2
        
        # If R is 1, this is integer.
        # If R is not 1, we have fractions.
        
        # We can return the dictionaries with float values?
        # The task says "integer/half-integer entries".
        
        # If R is 1, we are good.
        # If R is not 1, we might fail.
        
        # For now, return a_dict, b_dict, c_dict scaled by 1/R^2 in c.
        for k in c_dict:
            c_dict[k] = c_dict[k] / (R * R)
            
        # Verify one more time with floats
        for (i, j, k), val in combined.items():
            expected = a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0)
            if abs(val - expected) > 1e-9:
                return None
        
        return (a_dict, b_dict, c_dict)
    
    # Try merging pairs - multiple passes
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline - 10:
            break
        merged = False
        i = 0
        while i < len(terms) and time.time() < deadline - 5:
            j = i + 1
            while j < len(terms) and time.time() < deadline - 5:
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    a_new, b_new, c_new = result
                    # Remove terms i and j, add new term
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
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
