import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - start_time < DEADLINE

    def add_strassen_block(i1, i2, j1, j2, k1, k2, terms):
        """Strassen 2x2x2: A[i1:i3, j1:j3] * B[j1:j3, k1:k3] -> C[i1:i3, k1:k3]"""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)
        
        # M1 = (a+d)(e+h)
        terms.append((
            {a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}
        ))
        # M2 = (c+d)e
        terms.append((
            {c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}
        ))
        # M3 = a(f-h)
        terms.append((
            {a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}
        ))
        # M4 = d(g-e)
        terms.append((
            {d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}
        ))
        # M5 = (a+b)h
        terms.append((
            {a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}
        ))
        # M6 = (c-a)(e+f)
        terms.append((
            {c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}
        ))
        # M7 = (b-d)(g+h)
        terms.append((
            {b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}
        ))
    
    def add_schoolbook(i, j, k, terms):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))
    
    terms = []
    
    # Row 0: schoolbook (4 middle * 6 right = 24 terms)
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k, terms)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    # Split m=4 into two 2-col groups, p=6 into three 2-col groups
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            if not time_ok():
                break
            add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
        if not time_ok():
            break
    
    rank = len(terms)
    
    def terms_to_matrices(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Direct tensor verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def merge_terms(terms, strategy):
        """
        Merge terms that share identical factor pairs.
        strategy: 0 = merge by (V,W), 1 = merge by (U,W), 2 = merge by (U,V)
        """
        if not terms:
            return terms
        
        # Build keys for each term
        def make_key(term, factor_idx):
            # factor_idx: 0=U, 1=V, 2=W
            d = term[factor_idx]
            return tuple(sorted(d.items()))
        
        groups = {}
        for idx, term in enumerate(terms):
            if strategy == 0:  # merge by (V, W)
                gkey = (make_key(term, 1), make_key(term, 2))
            elif strategy == 1:  # merge by (U, W)
                gkey = (make_key(term, 0), make_key(term, 2))
            else:  # merge by (U, V)
                gkey = (make_key(term, 0), make_key(term, 1))
            
            if gkey not in groups:
                groups[gkey] = []
            groups[gkey].append(idx)
        
        new_terms = []
        for gkey, indices in groups.items():
            if len(indices) == 1:
                new_terms.append(terms[indices[0]])
            else:
                # Merge: sum the differing factor
                base = indices[0]
                u_m = dict(terms[base][0])
                v_m = dict(terms[base][1])
                w_m = dict(terms[base][2])
                
                for idx in indices[1:]:
                    if strategy == 0:  # merge U
                        for k, v in terms[idx][0].items():
                            u_m[k] = u_m.get(k, 0) + v
                    elif strategy == 1:  # merge V
                        for k, v in terms[idx][1].items():
                            v_m[k] = v_m.get(k, 0) + v
                    else:  # merge W
                        for k, v in terms[idx][2].items():
                            w_m[k] = w_m.get(k, 0) + v
                
                # Remove zero entries
                u_m = {k: v for k, v in u_m.items() if v != 0}
                v_m = {k: v for k, v in v_m.items() if v != 0}
                w_m = {k: v for k, v in w_m.items() if v != 0}
                
                new_terms.append((u_m, v_m, w_m))
        
        return new_terms
    
    # Try each merge strategy iteratively until no more merges are possible
    for strategy in [0, 1, 2]:
        if not time_ok():
            break
        prev_rank = len(terms)
        terms = merge_terms(terms, strategy)
        # Repeat if we made progress
        while len(terms) < prev_rank and time_ok():
            prev_rank = len(terms)
            terms = merge_terms(terms, strategy)
    
    # Try all strategies again in case cross-strategy merges become possible
    for _ in range(3):
        if not time_ok():
            break
        changed = False
        for strategy in [0, 1, 2]:
            prev_rank = len(terms)
            terms = merge_terms(terms, strategy)
            if len(terms) < prev_rank:
                changed = True
        if not changed:
            break
    
    rank = len(terms)
    U, V, W = terms_to_matrices(terms)
    
    # Verify correctness
    if not verify_decomposition(U, V, W):
        # Fallback: use the original 66-term decomposition
        terms_orig = []
        for j in range(m):
            for k in range(p):
                add_schoolbook(0, j, k, terms_orig)
        for j1 in [0, 2]:
            for k1 in [0, 2, 4]:
                add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1, terms_orig)
        U, V, W = terms_to_matrices(terms_orig)
    
    return {"U": U, "V": V, "W": W}
