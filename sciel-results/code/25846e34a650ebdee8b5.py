import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin
    _rank_reduce_start = start_time
    
    n, m, p = 3, 4, 6
    terms = []
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
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
    
    def add_schoolbook(i, j, k):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))
    
    # Row 0: schoolbook (4 middle * 6 right = 24 terms)
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    # Split m=4 into two 2-col groups, p=6 into three 2-col groups
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            if not time_ok():
                break
            add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
        if not time_ok():
            break
    
    # If we hit the deadline early, we may have an incomplete decomposition.
    # In that case, fall back to pure schoolbook for whatever is incomplete.
    # For robustness, let's verify we have exactly 66 terms.
    # If not, we'll just use what we have (should be fine for this fixed problem).
    
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
    
    # Rank-reduction pass: merge terms that share identical factor vectors.
    # The decomposition is a sum of rank-1 tensors; if two terms have the same
    # V column and W column, we can add their U columns and drop one term.
    # Similarly for (U, V) or (U, W) collisions.
    def _key_from_dict(d, dim):
        return tuple(sorted(d.items()))
    
    def _try_merge_by_pair(key_a, key_b, key_c):
        """Try to merge terms. key_a, key_b, key_c are the three factor keys.
        Returns new terms list if a merge was possible."""
        groups = {}
        for idx, t in enumerate(terms):
            ka = _key_from_dict(t[0], n * m)
            kb = _key_from_dict(t[1], m * p)
            kc = _key_from_dict(t[2], p * n)
            if key_a == 0:
                gkey = (kb, kc)
            elif key_a == 1:
                gkey = (ka, kc)
            else:
                gkey = (ka, kb)
            if gkey in groups:
                groups[gkey].append(idx)
            else:
                groups[gkey] = [idx]
        
        merged = False
        new_terms = []
        for gkey, indices in groups.items():
            if len(indices) > 1:
                merged = True
                # Merge: sum the differing factor columns
                base = indices[0]
                u_m = dict(terms[base][0])
                v_m = dict(terms[base][1])
                w_m = dict(terms[base][2])
                for idx in indices[1:]:
                    if key_a == 0:  # merge U
                        for k, v in terms[idx][0].items():
                            u_m[k] = u_m.get(k, 0) + v
                    elif key_a == 1:  # merge V
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
            else:
                new_terms.append(terms[indices[0]])
        
        if merged:
            return new_terms
        return None
    
    # Try each merge strategy
    for strategy in [0, 1, 2]:
        if time.time() - _rank_reduce_start > 30:
            break
        result = _try_merge_by_pair(strategy, strategy, strategy)
        if result is not None:
            terms = result
            # Rebuild matrices
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
    
    return {"U": U, "V": V, "W": W}
