import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            a3 = a + 3 * d
            aps.append((a, a + d, a + 2 * d, a3))
    
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    def solve(k):
        colors = [None] * n
        # Symmetry breaking
        colors[0] = 0
        domains = [set(range(k)) for _ in range(n)]
        domains[0] = {0}
        
        def propagate(pos, color):
            for ap_idx in pos_aps[pos]:
                ap = aps[ap_idx]
                others = [p for p in ap if p != pos]
                assigned_vals = [colors[p] for p in others if colors[p] is not None]
                unassigned = [p for p in others if colors[p] is None]
                
                if len(unassigned) == 0:
                    if len(set([color] + assigned_vals)) < 3:
                        return False
                elif len(unassigned) == 1:
                    p_un = unassigned[0]
                    S = {color} | set(assigned_vals)
                    if len(S) >= 3:
                        pass
                    elif len(S) == 2:
                        domains[p_un] = domains[p_un] - S
                        if not domains[p_un]:
                            return False
                    else:
                        return False
                elif len(unassigned) == 2:
                    p1, p2 = unassigned
                    S = {color} | set(assigned_vals)
                    if len(S) <= 1:
                        for p in [p1, p2]:
                            domains[p] = domains[p] - S
                            if not domains[p]:
                                return False
                # else: 3 unassigned, no pruning
        
            return True
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            unassigned = [i for i in range(n) if colors[i] is None]
            if not unassigned:
                return True
            
            pos = min(unassigned, key=lambda i: len(domains[i]))
            
            for c in sorted(domains[pos]):
                old_domain = domains[pos].copy()
                colors[pos] = c
                domains[pos] = {c}
                
                if propagate(pos, c):
                    if backtrack():
                        return True
                
                colors[pos] = None
                domains[pos] = old_domain
            
            return False
        
        if backtrack():
            return colors
        return None
    
    result = solve(3)
    if result is not None:
        return result
    
    result = solve(4)
    if result is not None:
        return result
    
    result = solve(5)
    if result is not None:
        return result
    
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    break
                vals = {color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]}
                if len(vals) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
    return colors
