def construct():
    import time
    from math import gcd
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    deadline = time.time() + 80  # safety margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Build conflict graph: edge (i,j) if exists k distinct from i,j with lcm(i,j)=lcm(i,k)=lcm(j,k)
    # Use 0-indexed: element i is elements[i]
    
    # Precompute lcm table
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[i][j] = lcm(elements[i], elements[j])
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build adjacency: adj[i] is set of j such that (i,j) is a conflict edge
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            L = lcm_table[i][j]
            # Check if there exists k != i,j with lcm(i,k)=L and lcm(j,k)=L
            found = False
            for k in range(n):
                if k == i or k == j:
                    continue
                if lcm_table[i][k] == L and lcm_table[j][k] == L:
                    found = True
                    break
            if found:
                adj[i].add(j)
                adj[j].add(i)
    
    # Now find maximum independent set
    # Use branch and bound with greedy coloring upper bound
    
    # Convert to bitsets for efficiency
    # Use Python integers as bitsets
    
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits
    
    # Greedy coloring for upper bound (DSATUR-like)
    def greedy_color(candidates):
        # candidates is a set of vertices
        # Assign colors greedily
        color = {}
        color_classes = []
        for v in sorted(candidates, key=lambda x: -len(adj[x] & candidates)):
            used_colors = set()
            for u in adj[v]:
                if u in candidates and u in color:
                    used_colors.add(color[u])
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
        return len(color_classes)
    
    best_set = set()
    
    def branch(candidates, current):
        nonlocal best_set
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best_set):
                best_set = set(current)
            return
        
        # Upper bound: |current| + greedy_color(candidates)
        ub = len(current) + greedy_color(candidates)
        if ub <= len(best_set):
            return
        
        # Choose vertex with max degree in candidates
        v = max(candidates, key=lambda x: len(adj[x] & candidates))
        
        # Branch 1: include v
        new_current = current | {v}
        new_candidates = candidates - {v} - (adj[v] & candidates)
        branch(new_candidates, new_current)
        
        # Branch 2: exclude v
        branch(candidates - {v}, current)
    
    all_vertices = set(range(n))
    branch(all_vertices, set())
    
    # Convert back to elements
    result = [elements[i] for i in sorted(best_set)]
    
    if not result:
        result = [1]
    
    return result
