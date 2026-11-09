def synthesize(matrix, depth_limit):
    import time
    import numpy as np
    
    start_time = time.time()
    TIME_LIMIT = 80.0
    
    m, n = matrix.shape
    
    out_bits = []
    for i in range(m):
        out_bits.append(frozenset(int(j) for j in np.where(matrix[i] != 0)[0]))
    
    wires = [frozenset([i]) for i in range(n)]
    wires.append(frozenset())
    
    out_expr = []
    for i in range(m):
        out_expr.append([j for j in range(n) if j in out_bits[i]])
    
    gates = []
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)
    
    while time_left() > 5.0:
        for i in range(m):
            if len(out_expr[i]) == 0:
                out_expr[i] = [n]
        
        remaining = [i for i in range(m) if len(out_expr[i]) > 1]
        if not remaining:
            break
        
        nw = len(wires)
        
        if n <= 64:
            w_ints = []
            for k in range(nw):
                v = 0
                for b in wires[k]:
                    v |= (1 << b)
                w_ints.append(v)
            
            r_ints = []
            for i in remaining:
                v = 0
                for b in out_bits[i]:
                    v |= (1 << b)
                r_ints.append(v)
            
            best_score = -1
            best_pair = None
            best_w = None
            
            for i in range(nw):
                wi = w_ints[i]
                for j in range(i + 1, nw):
                    w = wi ^ w_ints[j]
                    if w == 0:
                        continue
                    wp = bin(w).count('1')
                    sc = 0
                    for ri in r_ints:
                        ic = bin(ri & w).count('1')
                        sc += 2 * ic - wp
                    if sc > best_score:
                        best_score = sc
                        best_pair = (i, j)
                        best_w = w
            
            if best_pair is None or best_score <= 0:
                break
            
            i, j = best_pair
            gates.append((i, j))
            new_idx = nw
            
            s = set()
            v = best_w
            for b in range(n):
                if v & (1 << b):
                    s.add(b)
            wires.append(frozenset(s))
            
            for idx, oi in enumerate(remaining):
                expr = out_expr[oi]
                if i in expr and j in expr:
                    out_expr[oi] = [x for x in expr if x != i and x != j] + [new_idx]
        else:
            break
    
    outputs = []
    for i in range(m):
        expr = out_expr[i]
        if len(expr) == 0:
            outputs.append(n)
        elif len(expr) == 1:
            outputs.append(expr[0])
        else:
            cur = expr[0]
            for w in expr[1:]:
                gates.append((cur, w))
                cur = len(wires)
                wires.append(frozenset())
            outputs.append(cur)
    
    return {'gates': gates, 'outputs': outputs}
