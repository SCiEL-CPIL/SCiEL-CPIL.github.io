<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    # Start with K_{3,13}
    adj = [0] * n
    for u in range(3):
        for v in range(3, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def has_q3_after_add(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph using bitmask operations."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        result = False
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                def backtrack(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == len(unassigned):
                        # All assigned, verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True
                    
                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
                    
                    # Filter candidates: must be adjacent to all assigned Q3-neighbors of x
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if backtrack(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    
                    return False
                
                if backtrack(0, used_mask):
                    result = True
                    break
            if result:
                break
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return result

    # Try to add missing edges
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))
    
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 15:
            break
        if not has_q3_after_add(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
=======
def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    def make_adj(initial_edges):
        adj = [0] * n
        for u, v in initial_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edge_count(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def try_add_edges(adj, deadline):
        """Try to add missing edges greedily."""
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))
        
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if not _has_q3_after_add(adj, u, v, deadline):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj

    def _has_q3_after_add(adj, u, v, deadline):
        """Check if adding edge (u,v) creates a Q3 subgraph using bitmask operations."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        result = False
        for (qa, qb, unassigned, neighbor_lists) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                def backtrack(idx, used_mask):
                    if time.time() > deadline - 2:
                        return False
                    if idx == len(unassigned):
                        # All assigned, verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True
                    
                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
                    
                    # Filter candidates: must be adjacent to all assigned Q3-neighbors of x
                    for y in neighbor_lists[idx]:
                        hy = assignment[y]
                        candidates &= adj[hy]
                    
                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit
                        
                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if backtrack(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    
                    return False
                
                if backtrack(0, used_mask):
                    result = True
                    break
            if result:
                break
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return result

    # Try multiple initial configurations
    best_edges = []
    best_count = 0
    
    # Configuration 1: K_{3,13}
    init1 = [(u, v) for u in range(3) for v in range(3, 16)]
    adj1 = make_adj(init1)
    adj1 = try_add_edges(adj1, deadline)
    count1 = edge_count(adj1)
    if count1 > best_count:
        best_count = count1
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj1[i] & (1 << j)]
    
    if time.time() > deadline - 20:
        return best_edges
    
    # Configuration 2: K_{4,12}
    init2 = [(u, v) for u in range(4) for v in range(4, 16)]
    adj2 = make_adj(init2)
    adj2 = try_add_edges(adj2, deadline)
    count2 = edge_count(adj2)
    if count2 > best_count:
        best_count = count2
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj2[i] & (1 << j)]
    
    if time.time() > deadline - 20:
        return best_edges
    
    # Configuration 3: K_{4,4,4,4} complete 4-partite
    groups = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
    init3 = []
    for gi in range(4):
        for gj in range(gi+1, 4):
            for u in groups[gi]:
                for v in groups[gj]:
                    init3.append((u, v))
    adj3 = make_adj(init3)
    adj3 = try_add_edges(adj3, deadline)
    count3 = edge_count(adj3)
    if count3 > best_count:
        best_count = count3
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj3[i] & (1 << j)]
    
    if time.time() > deadline - 10:
        return best_edges
    
    # Configuration 4: K_{2,2,2,2,2,2,2,2} complete 8-partite (16 vertices, 2 per part)
    groups4 = [[2*i, 2*i+1] for i in range(8)]
    init4 = []
    for gi in range(8):
        for gj in range(gi+1, 8):
            for u in groups4[gi]:
                for v in groups4[gj]:
                    init4.append((u, v))
    adj4 = make_adj(init4)
    adj4 = try_add_edges(adj4, deadline)
    count4 = edge_count(adj4)
    if count4 > best_count:
        best_count = count4
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj4[i] & (1 << j)]
    
    if time.time() > deadline - 5:
        return best_edges
    
    # Configuration 5: K_{3,3,3,3,4} complete 5-partite
    groups5 = [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14,15]]
    init5 = []
    for gi in range(5):
        for gj in range(gi+1, 5):
            for u in groups5[gi]:
                for v in groups5[gj]:
                    init5.append((u, v))
    adj5 = make_adj(init5)
    adj5 = try_add_edges(adj5, deadline)
    count5 = edge_count(adj5)
    if count5 > best_count:
        best_count = count5
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj5[i] & (1 << j)]
    
    return best_edges
>>>>>>> REPLACE
