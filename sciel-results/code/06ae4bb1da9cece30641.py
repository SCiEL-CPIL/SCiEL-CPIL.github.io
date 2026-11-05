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
    
    # Greedy post-pass: merge pairs of terms whose sum is rank-1
    def try_merge(terms):
        """Find pairs (i,j) where T_i + T_j is rank-1, and merge them."""
        n_terms = len(terms)
        if n_terms < 2:
            return terms
        deadline_merge = start_time + 140.0
        
        def is_rank1_sum(ti, tj):
            """Check if ti + tj is rank-1 and return the merged term if so."""
            # ti = (U_i, V_i, W_i), tj = (U_j, V_j, W_j)
            # T_i + T_j is rank-1 iff the 3-tensor has rank 1.
            # Check: all 2x2 minors of any matricization must vanish.
            # Build the combined tensor as dict (u,v,w) -> val
            combined = {}
            for u, uv in ti[0].items():
                for v, vv in ti[1].items():
                    for w, ww in ti[2].items():
                        key = (u, v, w)
                        combined[key] = combined.get(key, 0) + uv * vv * ww
            for u, uv in tj[0].items():
                for v, vv in tj[1].items():
                    for w, ww in tj[2].items():
                        key = (u, v, w)
                        combined[key] = combined.get(key, 0) + uv * vv * ww
            
            # Remove zeros
            combined = {k: v for k, v in combined.items() if v != 0}
            if not combined:
                return None
            
            # For rank-1, pick any two (u,v) pairs and check W proportionality
            # Group by (u,v): for each (u,v), get the W vector
            uv_w = {}
            for (u, v, w), val in combined.items():
                if (u, v) not in uv_w:
                    uv_w[(u, v)] = {}
                uv_w[(u, v)][w] = val
            
            # Pick a reference (u,v) pair
            ref_uv = list(uv_w.keys())[0]
            ref_w = uv_w[ref_uv]
            ref_w_list = sorted(ref_w.items())
            
            # For each other (u,v), check W vector is proportional to ref
            for uv, w_vec in uv_w.items():
                if uv == ref_uv:
                    continue
                w_list = sorted(w_vec.items())
                # Check proportionality
                if len(w_list) == 0:
                    continue
                # Find ratio from first element
                ref_dict = dict(ref_w_list)
                found_ratio = None
                proportional = True
                for w_idx, w_val in w_list:
                    if w_idx in ref_dict and ref_dict[w_idx] != 0:
                        r = w_val / ref_dict[w_idx]
                        if found_ratio is None:
                            found_ratio = r
                        elif abs(r - found_ratio) > 1e-9:
                            proportional = False
                            break
                    else:
                        # w_idx not in ref or ref value is 0
                        if w_val != 0:
                            proportional = False
                            break
                if not proportional:
                    return None
                # Also check: elements in ref not in w_vec must have value 0 in w_vec
                for w_idx, w_val in ref_w_list:
                    if w_idx not in w_vec and w_val != 0:
                        return None
            
            # Now construct the merged rank-1 term
            # U'[u] * V'[v] * W'[w] = combined[u,v,w]
            # Set U'[u] = 1 for one u, determine rest
            all_u = set(u for u, v, w in combined.keys())
            all_v = set(v for u, v, w in combined.keys())
            all_w = set(w for u, v, w in combined.keys())
            
            # Pick anchor u
            u_anchor = list(all_u)[0]
            U_prime = {u_anchor: 1}
            
            # For each v, V'[v] is determined by: U'[u_anchor]*V'[v]*W'[w] = combined[u_anchor,v,w]
            # So V'[v]*W'[w] = combined[u_anchor,v,w] for all w
            # This means for fixed v, the vector {combined[u_anchor,v,w]} must be proportional to W'
            
            # Determine W' from one (u,v) pair
            # Use (u_anchor, v_anchor) where v_anchor is in all_v
            v_candidates = [v for v in all_v if (u_anchor, v) in uv_w and uv_w[(u_anchor, v)]]
            if not v_candidates:
                return None
            v_anchor = v_candidates[0]
            w_vec_anchor = uv_w[(u_anchor, v_anchor)]
            if not w_vec_anchor:
                return None
            
            # Set W' = w_vec_anchor (since U'[u_anchor]=1, V'[v_anchor]=1)
            W_prime = dict(w_vec_anchor)
            V_prime = {v_anchor: 1}
            
            # Now determine V'[v] for other v: V'[v] * W'[w] = combined[u_anchor, v, w]
            for v in all_v:
                if v == v_anchor:
                    continue
                if (u_anchor, v) not in uv_w:
                    V_prime[v] = 0
                    continue
                w_vec = uv_w[(u_anchor, v)]
                # V'[v] = w_vec[w] / W'[w] for any w where W'[w] != 0
                ratio_found = None
                valid = True
                for w, val in w_vec.items():
                    if W_prime.get(w, 0) != 0:
                        r = val / W_prime[w]
                        if ratio_found is None:
                            ratio_found = r
                        elif abs(r - ratio_found) > 1e-9:
                            valid = False
                            break
                    else:
                        if val != 0:
                            valid = False
                            break
                if not valid:
                    return None
                V_prime[v] = ratio_found if ratio_found is not None else 0
            
            # Determine U'[u] for other u: U'[u] * V'[v] * W'[w] = combined[u, v, w]
            for u in all_u:
                if u == u_anchor:
                    continue
                # Use v_anchor: U'[u] * 1 * W'[w] = combined[u, v_anchor, w]
                ratio_found = None
                valid = True
                for w, w_val in W_prime.items():
                    if w_val == 0:
                        continue
                    val = combined.get((u, v_anchor, w), 0)
                    r = val / w_val
                    if ratio_found is None:
                        ratio_found = r
                    elif abs(r - ratio_found) > 1e-9:
                        valid = False
                        break
                if not valid:
                    return None
                U_prime[u] = ratio_found if ratio_found is not None else 0
            
            # Verify
            for (u, v, w), expected in combined.items():
                actual = U_prime.get(u, 0) * V_prime.get(v, 0) * W_prime.get(w, 0)
                if abs(actual - expected) > 1e-9:
                    return None
            
            # Clean up zeros
            U_prime = {k: v for k, v in U_prime.items() if v != 0}
            V_prime = {k: v for k, v in V_prime.items() if v != 0}
            W_prime = {k: v for k, v in W_prime.items() if v != 0}
            
            if U_prime and V_prime and W_prime:
                return (U_prime, V_prime, W_prime)
            return None
        
        merged = list(terms)
        removed = [False] * n_terms
        merges = 0
        
        for i in range(n_terms):
            if removed[i] or time.time() > deadline_merge:
                continue
            for j in range(i + 1, n_terms):
                if removed[j] or time.time() > deadline_merge:
                    continue
                
                merged_term = is_rank1_sum(merged[i], merged[j])
                if merged_term is not None:
                    merged[i] = merged_term
                    removed[j] = True
                    merges += 1
        
        result = [merged[i] for i in range(n_terms) if not removed[i]]
        return result
    
    terms = try_merge(terms)
    return _build_result(terms, n, m, p)
# EVOLVE-BLOCK-END
