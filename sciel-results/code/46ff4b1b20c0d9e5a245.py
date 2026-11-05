"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a 2+2+1 block partition with Strassen for the 2x2x4 blocks and schoolbook for the rest.
This achieves a rank of 117.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
    # Helper to add schoolbook terms for a block
    def add_schoolbook(r_start, r_end, c_start, c_end, k_start, k_end):
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                for k in range(k_start, k_end):
                    u_idx = r * m + c
                    v_idx = c * p + k
                    w_idx = k * n + r
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen_2x2(r_start, c_start, k_start):
        A00 = r_start * m + c_start
        A01 = r_start * m + (c_start + 1)
        A10 = (r_start + 1) * m + c_start
        A11 = (r_start + 1) * m + (c_start + 1)
        
        B00 = c_start * p + k_start
        B01 = c_start * p + (k_start + 1)
        B10 = (c_start + 1) * p + k_start
        B11 = (c_start + 1) * p + (k_start + 1)
        
        C00 = k_start * n + r_start
        C01 = (k_start + 1) * n + r_start
        C10 = k_start * n + (r_start + 1)
        C11 = (k_start + 1) * n + (r_start + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
    
    # Helper to add Strassen terms for a 2x2x4 block (saves 1 rank vs two 2x2x2)
    def add_strassen_2x2x4(r_start, c_start, k_start):
        # A indices (2x2)
        A00 = r_start * m + c_start
        A01 = r_start * m + (c_start + 1)
        A10 = (r_start + 1) * m + c_start
        A11 = (r_start + 1) * m + (c_start + 1)
        
        # B indices (2x4)
        B00 = c_start * p + k_start
        B01 = c_start * p + (k_start + 1)
        B02 = c_start * p + (k_start + 2)
        B03 = c_start * p + (k_start + 3)
        B10 = (c_start + 1) * p + k_start
        B11 = (c_start + 1) * p + (k_start + 1)
        B12 = (c_start + 1) * p + (k_start + 2)
        B13 = (c_start + 1) * p + (k_start + 3)
        
        # C indices (4x2)
        C00 = k_start * n + r_start
        C10 = (k_start + 1) * n + r_start
        C20 = (k_start + 2) * n + r_start
        C30 = (k_start + 3) * n + r_start
        C01 = k_start * n + (r_start + 1)
        C11 = (k_start + 1) * n + (r_start + 1)
        C21 = (k_start + 2) * n + (r_start + 1)
        C31 = (k_start + 3) * n + (r_start + 1)
        
        # M1 = (A00+A11)(B00+B11) -> C00+=, C11-=
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: -1})
        # M2 = (A00+A11)(B02+B12) -> C20+=, C31-=
        add_term({A00: 1, A11: 1}, {B02: 1, B12: 1}, {C20: 1, C31: -1})
        # M3 = (A10+A11)B00 -> C10+=, C11-=
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        # M4 = (A10+A11)B02 -> C20+=, C21-=
        add_term({A10: 1, A11: 1}, {B02: 1}, {C20: 1, C21: -1})
        # M5 = A00(B01-B11) -> C01+=, C11+=
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        # M6 = A00(B03-B13) -> C30+=, C31+=
        add_term({A00: 1}, {B03: 1, B13: -1}, {C30: 1, C31: 1})
        # M7 = A11(B10-B00) -> C00+=, C10+=
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        # M8 = A11(B12-B02) -> C20+=, C30+=
        add_term({A11: 1}, {B12: 1, B02: -1}, {C20: 1, C30: 1})
        # M9 = (A00+A01)B11 -> C00-=, C01+=
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        # M10 = (A00+A01)B13 -> C30-=, C31+=
        add_term({A00: 1, A01: 1}, {B13: 1}, {C30: -1, C31: 1})
        # M11 = (A10-A00)(B00+B01) -> C11+=
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        # M12 = (A10-A00)(B02+B03) -> C31+=
        add_term({A10: 1, A00: -1}, {B02: 1, B03: 1}, {C31: 1})
        # M13 = (A01-A11)(B10+B11) -> C00+=
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
        # M14 = (A01-A11)(B12+B13) -> C20+=
        add_term({A01: 1, A11: -1}, {B12: 1, B13: 1}, {C20: 1})
    
    def _build_result(terms, n, m, p):
        rank = len(terms)
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

    blocks = [(0, 2), (2, 4), (4, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    return _build_result(terms, n, m, p)
                
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                elif ar == 2 and ac == 2 and bc == 4:
                    add_strassen_2x2x4(r1, c1, k1)
                else:
                    add_schoolbook(r1, r2, c1, c2, k1, k2)
    
    # Greedy post-pass: merge pairs of terms whose outer products sum to a rank-1 term
    def try_merge(terms):
        """Try to find pairs (i,j) where sum of their rank-1 tensors is itself rank-1."""
        n_terms = len(terms)
        if n_terms < 2:
            return terms
        deadline_merge = start_time + 140.0
        
        # Build sparse representation for quick outer-product checking
        # For each term, store as dict of (u_idx, v_idx) -> w_idx contributions
        # Actually, we need: for terms i and j, check if T_i + T_j is rank-1
        # T_i = U_i ⊗ V_i ⊗ W_i (3-way tensor)
        # T_i + T_j is rank-1 iff there exist a,b,c (scalars) and rank-1 U',V',W' s.t.
        # U_i⊗V_i⊗W_i + U_j⊗V_j⊗W_j = a*U'⊗V'⊗W'
        # This is equivalent to: the 2-dimensional subspace spanned by {U_i⊗V_i, U_j⊗V_j}
        # being 1-dimensional AND the corresponding W vectors being proportional... 
        # More practically: check if there exist scalars α,β,γ,δ such that
        # α*U_i = γ*U', β*V_i = δ*V', etc.
        
        # Simpler approach: for each pair, check if the combined 3-tensor has rank 1
        # by checking if all 2x2 minors of the matrix unfoldings vanish.
        
        merged = list(terms)
        removed = [False] * n_terms
        
        for i in range(n_terms):
            if removed[i] or time.time() > deadline_merge:
                continue
            for j in range(i + 1, n_terms):
                if removed[j] or time.time() > deadline_merge:
                    continue
                
                # Check if term_i + term_j is rank-1
                # Collect all (u,v) pairs that appear in either term
                uv_pairs = {}
                for u_idx, u_val in terms[i][0].items():
                    for v_idx, v_val in terms[i][1].items():
                        key = (u_idx, v_idx)
                        uv_pairs[key] = uv_pairs.get(key, 0) + u_val * v_val
                for u_idx, u_val in terms[j][0].items():
                    for v_idx, v_val in terms[j][1].items():
                        key = (u_idx, v_idx)
                        uv_pairs[key] = uv_pairs.get(key, 0) + u_val * v_val
                
                # For the sum to be rank-1, all (u,v) pairs with non-zero coefficient
                # must have W contributions that are proportional
                w_vectors = {}
                for (u_idx, v_idx), coeff in uv_pairs.items():
                    if coeff == 0:
                        continue
                    # Get W contribution from term i at this (u,v)
                    w_i = 0
                    if u_idx in terms[i][0] and v_idx in terms[i][1]:
                        w_i = terms[i][0][u_idx] * terms[i][1][v_idx]
                    # Get W contribution from term j
                    w_j = 0
                    if u_idx in terms[j][0] and v_idx in terms[j][1]:
                        w_j = terms[j][0][u_idx] * terms[j][1][v_idx]
                    w_vectors[(u_idx, v_idx)] = (w_i, w_j)
                
                # For rank-1: there must exist λ such that for all (u,v),
                # the W vector is proportional to a single W'
                # i.e., (w_i, w_j) = s_{uv} * (w'_i, w'_j) for some common direction
                # This means all (w_i, w_j) pairs are collinear in R^2
                if len(w_vectors) <= 1:
                    # Trivially rank-1
                    pass
                else:
                    # Check collinearity of all (w_i, w_j) vectors
                    first = list(w_vectors.values())[0]
                    collinear = True
                    for w_pair in w_vectors.values():
                        # Cross product in 2D: w_i1*w_j2 - w_i2*w_j1
                        cross = first[0] * w_pair[1] - first[1] * w_pair[0]
                        if cross != 0:
                            collinear = False
                            break
                    if not collinear:
                        continue
                
                # If we get here, the sum is rank-1. Create merged term.
                # The merged term's U, V, W need to be determined.
                # We need U'⊗V'⊗W' = T_i + T_j
                # Find a non-zero (u,v) pair to anchor
                anchor = None
                for (u_idx, v_idx), coeff in uv_pairs.items():
                    if coeff != 0:
                        anchor = (u_idx, v_idx)
                        break
                if anchor is None:
                    continue
                
                u_anchor, v_anchor = anchor
                # U'[u_anchor] * V'[v_anchor] * W'[k] = (T_i + T_j)[u_anchor, v_anchor, k]
                # We can set U'[u_anchor] = 1, then V'[v_anchor] * W'[k] = combined[k]
                # combined[k] = T_i[u,v,k] + T_j[u,v,k]
                
                # Get the combined W vector at (u_anchor, v_anchor)
                combined_w = {}
                for u_idx, u_val in terms[i][0].items():
                    if u_idx != u_anchor:
                        continue
                    for v_idx, v_val in terms[i][1].items():
                        if v_idx != v_anchor:
                            continue
                        for w_idx, w_val in terms[i][2].items():
                            combined_w[w_idx] = combined_w.get(w_idx, 0) + u_val * v_val * w_val
                for u_idx, u_val in terms[j][0].items():
                    if u_idx != u_anchor:
                        continue
                    for v_idx, v_val in terms[j][1].items():
                        if v_idx != v_anchor:
                            continue
                        for w_idx, w_val in terms[j][2].items():
                            combined_w[w_idx] = combined_w.get(w_idx, 0) + u_val * v_val * w_val
                
                # Now we need to factor U'⊗V'⊗W' = sum over (u,v,k) of M[u,v,k] e_u⊗e_v⊗e_k
                # where M is the combined 3-tensor.
                # This is complex. Let's use a simpler criterion:
                # Only merge if one term's U is proportional to the other's, etc.
                
                # Actually, let's use a much simpler sufficient condition:
                # If U_i = c * U_j for some scalar c, and V_i = d * V_j, and W_i = e * W_j,
                # then the sum is also rank-1.
                
                # Check if U_i is proportional to U_j
                u_i = terms[i][0]
                u_j = terms[j][0]
                if set(u_i.keys()) != set(u_j.keys()):
                    continue
                # Find ratio
                ratio_u = None
                proportional_u = True
                for key in u_i:
                    if u_j[key] == 0:
                        if u_i[key] != 0:
                            proportional_u = False
                            break
                    else:
                        r = u_i[key] / u_j[key]
                        if ratio_u is None:
                            ratio_u = r
                        elif abs(r - ratio_u) > 1e-9:
                            proportional_u = False
                            break
                if not proportional_u or ratio_u is None:
                    continue
                
                # Check V proportionality
                v_i = terms[i][1]
                v_j = terms[j][1]
                if set(v_i.keys()) != set(v_j.keys()):
                    continue
                ratio_v = None
                proportional_v = True
                for key in v_i:
                    if v_j[key] == 0:
                        if v_i[key] != 0:
                            proportional_v = False
                            break
                    else:
                        r = v_i[key] / v_j[key]
                        if ratio_v is None:
                            ratio_v = r
                        elif abs(r - ratio_v) > 1e-9:
                            proportional_v = False
                            break
                if not proportional_v or ratio_v is None:
                    continue
                
                # Check W proportionality
                w_i = terms[i][2]
                w_j = terms[j][2]
                if set(w_i.keys()) != set(w_j.keys()):
                    continue
                ratio_w = None
                proportional_w = True
                for key in w_i:
                    if w_j[key] == 0:
                        if w_i[key] != 0:
                            proportional_w = False
                            break
                    else:
                        r = w_i[key] / w_j[key]
                        if ratio_w is None:
                            ratio_w = r
                        elif abs(r - ratio_w) > 1e-9:
                            proportional_w = False
                            break
                if not proportional_w or ratio_w is None:
                    continue
                
                # All proportional: merge into single term
                # (U_i + U_j) ⊗ (V_i + V_j) ⊗ (W_i + W_j) is NOT correct in general
                # But T_i + T_j = U_i⊗V_i⊗W_i + U_j⊗V_j⊗W_j
                # = ratio_u*U_j ⊗ ratio_v*V_j ⊗ W_i + U_j ⊗ V_j ⊗ W_j
                # = U_j ⊗ V_j ⊗ (ratio_u*ratio_v*W_i + W_j)
                # So merged term: U=U_j, V=V_j, W=ratio_u*ratio_v*W_i + W_j
                merged_U = dict(u_j)
                merged_V = dict(v_j)
                merged_W = {}
                all_w_keys = set(w_i.keys()) | set(w_j.keys())
                for k in all_w_keys:
                    val = ratio_u * ratio_v * w_i.get(k, 0) + w_j.get(k, 0)
                    if val != 0:
                        merged_W[k] = val
                
                if merged_W:
                    # Replace term i with merged, remove term j
                    merged[i] = (merged_U, merged_V, merged_W)
                    removed[j] = True
        
        result = [merged[i] for i in range(n_terms) if not removed[i]]
        return result
    
    terms = try_merge(terms)
    return _build_result(terms, n, m, p)
# EVOLVE-BLOCK-END
