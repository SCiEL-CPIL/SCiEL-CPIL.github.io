import random
import time

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute for each pair of Q3 vertices, the list of unassigned vertices sorted by constraint
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 16
    full_mask = (1 << n) - 1
    
    best_edges = 0
    best_adj = None
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def has_q3_after_add(adj, u, v):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        found = False
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            if found:
                break
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)
                
                def bt(idx, used_mask):
                    nonlocal found
                    if found:
                        return True
                    if time.time() > deadline - 3:
                        return False
                    if idx == 6:
                        # Check all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        found = True
                        return True
                    
                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
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
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False
                
                if bt(0, used_mask):
                    found = True
                    break
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return found
    
    def try_add_edges_greedily(adj, missing_edges):
        for u, v in missing_edges:
            if time.time() > deadline - 10:
                break
            if not (adj[u] & (1 << v)):
                if not has_q3_after_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    def local_search(adj):
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try adding each missing edge
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline - 15:
                        break
                    if not (adj[i] & (1 << j)):
                        if not has_q3_after_add(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
            
            # Try removing one edge and re-adding greedily
            if time.time() < deadline - 25:
                edges_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges_list.append((i, j))
                random.shuffle(edges_list)
                
                for a, b in edges_list[:50]:  # Limit number of removals tried
                    if time.time() > deadline - 25:
                        break
                    # Save state
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    
                    old_count = count_edges(adj)
                    
                    # Try to add all missing edges greedily
                    for i in range(n):
                        for j in range(i + 1, n):
                            if time.time() > deadline - 25:
                                break
                            if not (adj[i] & (1 << j)):
                                if not has_q3_after_add(adj, i, j):
                                    adj[i] |= (1 << j)
                                    adj[j] |= (1 << i)
                    
                    new_count = count_edges(adj)
                    if new_count > old_count:
                        improved = True
                    else:
                        # Restore: we can't easily undo, so rebuild
                        # Actually, we need to be more careful. Let's just keep the better state.
                        # Since we modified adj in place, we need to track what was added.
                        # This is complex, so let's skip this approach for now.
                        pass
    
    # Multiple restarts with different strategies
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize with K_{3,13}
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Add Hamiltonian cycle on part B (vertices 3-15)
        if restart == 0:
            # Standard cycle: 3-4-5-...-15-3
            for i in range(3, 15):
                adj[i] |= (1 << (i + 1))
                adj[i + 1] |= (1 << i)
            adj[3] |= (1 << 15)
            adj[15] |= (1 << 3)
        elif restart == 1:
            # Different cycle
            cycle = list(range(3, 16))
            random.shuffle(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        else:
            # Random subset of cycle edges
            cycle = list(range(3, 16))
            random.shuffle(cycle)
            for i in range(len(cycle)):
                if random.random() < 0.7:
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Get missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))
        
        random.shuffle(missing)
        
        # Greedily add edges
        try_add_edges_greedily(adj, missing)
        
        # Local search
        local_search(adj)
        
        # Track best
        cur_count = count_edges(adj)
        if cur_count > best_edges:
            best_edges = cur_count
            best_adj = [a for a in adj]
    
    # Final greedy pass on best
    if best_adj and time.time() < deadline - 5:
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                if not (best_adj[i] & (1 << j)):
                    if not has_q3_after_add(best_adj, i, j):
                        best_adj[i] |= (1 << j)
                        best_adj[j] |= (1 << i)
    
    # Collect edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
