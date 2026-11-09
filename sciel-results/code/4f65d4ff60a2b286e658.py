import itertools

def construct():
    # 1. Define the Grötzsch graph (Mycielski of C5)
    edges_old = []
    # C5
    for i in range(5):
        edges_old.append((i, (i + 1) % 5))
    
    adj_old = set()
    for u, v in edges_old:
        adj_old.add((u, v))
        adj_old.add((v, u))
        
    # Shadows
    for i in range(5):
        for j in range(5):
            if i != j and (i, j) not in adj_old:
                edges_old.append((i, 5 + j))
                
    # Apex 10 connected to all shadows 5..9
    for j in range(5):
        edges_old.append((10, 5 + j))
        
    E_old = set()
    for u, v in edges_old:
        E_old.add((u, v))
        E_old.add((v, u))
        
    # 2. Generate all independent sets of G_old
    independent_sets = []
    for mask in range(1 << 11):
        S = set()
        for i in range(11):
            if mask & (1 << i):
                S.add(i)
        is_ind = True
        for u, v in E_old:
            if u in S and v in S:
                is_ind = False
                break
        if is_ind:
            independent_sets.append(S)
            
    # Sort independent sets by size descending to find large solutions faster
    independent_sets.sort(key=len, reverse=True)
    
    V_new_indices = range(5)
    new_edges_possible = []
    for i in range(5):
        for j in range(i + 1, 5):
            new_edges_possible.append((i, j))
            
    best_total_edges = len(E_old) // 2
    best_config = None
    
    # Iterate over all subsets of new edges (2^10 = 1024)
    for mask in range(1 << 10):
        G_new_edges = set()
        for k in range(10):
            if mask & (1 << k):
                u, v = new_edges_possible[k]
                G_new_edges.add((u, v))
                G_new_edges.add((v, u))
                
        # Check if G_new is triangle-free
        is_tf = True
        for u in range(5):
            for v in range(u + 1, 5):
                for w in range(v + 1, 5):
                    if (u, v) in G_new_edges and (v, w) in G_new_edges and (w, u) in G_new_edges:
                        is_tf = False
                        break
                if not is_tf:
                    break
            if not is_tf:
                break
        if not is_tf:
            continue
            
        # Backtrack to find max cross edges
        best_cross = 0
        best_assignment = None
        
        def backtrack(u, current_assignment, current_cross_count):
            nonlocal best_cross, best_assignment
            if u == 5:
                if current_cross_count > best_cross:
                    best_cross = current_cross_count
                    best_assignment = [set(s) for s in current_assignment]
                return
            
            for S in independent_sets:
                valid = True
                for prev_u in range(u):
                    prev_S = current_assignment[prev_u]
                    if S.intersection(prev_S):
                        if (u, prev_u) in G_new_edges:
                            valid = False
                            break
                if valid:
                    current_assignment.append(S)
                    backtrack(u + 1, current_assignment, current_cross_count + len(S))
                    current_assignment.pop()
                    
        backtrack(0, [], 0)
        
        total_edges = len(E_old) // 2 + len(G_new_edges) // 2 + best_cross
        if total_edges > best_total_edges:
            best_total_edges = total_edges
            best_config = (G_new_edges, best_assignment)
            
    final_edges = [list(e) for e in E_old]
    
    if best_config:
        G_new_edges, assignment = best_config
        for u, v in G_new_edges:
            if u < v:
                final_edges.append([11 + u, 11 + v])
                
        for u in range(5):
            S = assignment[u]
            for w in S:
                final_edges.append([11 + u, w])
                
    return final_edges
