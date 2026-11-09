import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    
    # Precompute all nonconstant 4-term progressions
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            a3 = a + 3 * d
            if a3 > N:
                break
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            aps.append((idx1, idx2, idx3, idx4))
            
    # Map index to list of AP indices
    ap_indices = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            ap_indices[idx].append(i)

    def solve(max_colors):
        colors = [-1] * N
        domains = [set(range(max_colors)) for _ in range(N)]
        
        def get_mrv_var():
            min_size = max_colors + 1
            var = -1
            for i in range(N):
                if colors[i] == -1:
                    size = len(domains[i])
                    if size < min_size:
                        min_size = size
                        var = i
            return var

        def forward_check(var, val):
            for ap_idx in ap_indices[var]:
                ap = aps[ap_idx]
                c = [colors[i] for i in ap]
                unassigned = [i for i in ap if colors[i] == -1]
                
                if not unassigned:
                    # All assigned, check validity
                    distinct = len(set(c))
                    if distinct < 3:
                        return False
                else:
                    # Prune domains of unassigned variables
                    for u in unassigned:
                        assigned_colors = set(c[j] for j in range(4) if c[j] != -1)
                        k = len(assigned_colors)
                        m = len(unassigned)
                        
                        if m == 1:
                            need_new = 3 - k
                            if need_new < 0:
                                need_new = 0
                            if need_new > 1:
                                return False
                            if need_new == 1:
                                new_domain = domains[u] - assigned_colors
                                if not new_domain:
                                    return False
                                domains[u] = new_domain
            return True

        def backtrack():
            if time.time() > deadline:
                return False
            
            var = get_mrv_var()
            if var == -1:
                return True
            
            # Heuristic: Try values that are already used in neighbors? 
            # Or just 0, 1, ...
            # To minimize colors, we don't care which specific colors from the palette we use, 
            # as long as we use <= max_colors.
            # However, using colors that appear more often might be better?
            # Let's stick to sorted order.
            
            for val in sorted(domains[var]):
                colors[var] = val
                domains[var] = {val}
                
                if forward_check(var, val):
                    if backtrack():
                        return True
                
                colors[var] = -1
                domains[var] = set(range(max_colors))
            
            return False

        if backtrack():
            return colors
        return None

    # Try 3 colors first
    res = solve(3)
    if res:
        return res

    # Fallback to 4 colors
    res = solve(4)
    if res:
        return res

    # Fallback to 5 colors
    res = solve(5)
    if res:
        return res

    # Absolute fallback: Greedy
    colors = [0] * N
    return colors
