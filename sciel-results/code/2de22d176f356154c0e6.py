import time
import random
import itertools

def construct():
    deadline = time.time() + 115
    n = 16
    
    # Precompute all 8-subsets of vertices
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute Q3 edge masks for each permutation? No, too many.
    # Instead, for a given 8-subset, check if the induced subgraph is isomorphic to Q3.
    # Q3 has 12 edges. The induced subgraph must have exactly 12 edges.
    # Then we check if it's bipartite and has degree 3 for all vertices?
    # Actually, Q3 is the unique 3-regular bipartite graph on 8 vertices.
    # So: 
    # 1. Count edges in induced subgraph. Must be 12.
    # 2. Check if all vertices have degree 3 within the subset.
    # 3. Check if it is bipartite.
    # If all true, it is Q3.
    
    def is_q3_induced(subset, adj):
        # subset is a tuple of 8 vertices
        # Map local indices 0..7 to global vertices
        local_to_global = {i: subset[i] for i in range(8)}
        
        # Count edges and check degrees
        deg = [0] * 8
        edge_count = 0
        for i in range(8):
            u = subset[i]
            for j in range(i+1, 8):
                v = subset[j]
                if adj[u] & (1 << v):
                    deg[i] += 1
                    deg[j] += 1
                    edge_count += 1
        
        if edge_count != 12:
            return False
            
        for d in deg:
            if d != 3:
                return False
        
        # Check bipartiteness
        # BFS/DFS on the induced subgraph
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        while queue:
            u_local = queue.pop(0)
            u_global = subset[u_local]
            # Iterate neighbors of u_global within the subset
            mask = adj[u_global]
            for v_local in range(8):
                if v_local == u_local:
                    continue
                v_global = subset[v_local]
                if mask & (1 << v_global):
                    if color[v_local] == -1:
                        color[v_local] = 1 - color[u_local]
                        queue.append(v_local)
                    elif color[v_local] == color[u_local]:
                        return False
        
        # If we reached here, it's a 3-regular bipartite graph on 8 vertices, which is Q3
        return True

    def count_q3s(adj):
        count = 0
        for subset in all_8_subsets:
            if is_q3_induced(subset, adj):
                count += 1
        return count

    def find_q3(adj):
        for subset in all_8_subsets:
            if is_q3_induced(subset, adj):
                return subset
        return None

    def edge_count(adj):
        c = 0
        for u in range(n):
            c += bin(adj[u]).count('1')
        return c // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    # Strategy:
    # 1. Start with K16.
    # 2. While Q3 exists:
    #      Find a Q3.
    #      Remove an edge from it. Which one?
    #      To maximize final edges, we should remove an edge that is "shared" by many Q3s?
    #      Or just remove a random one?
    #      Let's try removing the edge that appears in the most Q3s within the current graph?
    #      That's expensive.
    #      Simpler: Remove a random edge from the found Q3.
    #      Even better: Try all 12 edges of the Q3. For each, remove it, count remaining Q3s (approximate?), pick the one that reduces Q3 count most.
    #      Counting Q3s is O(C(16,8) * check) = 12870 * check.
    #      Check is fast.
    #      Let's try: For each edge in the Q3, remove it, count Q3s in the new graph. Pick min.
    
    adj = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    best_adj = adj[:]
    best_count = edge_count(adj)
    
    # Greedy removal
    while time.time() < deadline - 20:
        q3 = find_q3(adj)
        if not q3:
            break
        
        # Get edges of this Q3
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                u = q3[i]
                v = q3[j]
                if adj[u] & (1 << v):
                    q3_edges.append((u, v))
        
        # Evaluate removing each edge
        best_edge_to_remove = None
        min_q3_count = float('inf')
        
        for u, v in q3_edges:
            trial = adj[:]
            trial[u] &= ~(1 << v)
            trial[v] &= ~(1 << u)
            
            # Count Q3s in trial
            c = 0
            for subset in all_8_subsets:
                if is_q3_induced(subset, trial):
                    c += 1
                    if c > min_q3_count: # Early break
                        break
            
            if c < min_q3_count:
                min_q3_count = c
                best_edge_to_remove = (u, v)
                if c == 0:
                    break
                    
        if best_edge_to_remove:
            u, v = best_edge_to_remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        # If no edge removed (shouldn't happen), break to avoid infinite loop
        if not q3_edges:
            break

    # Now adj is Q3-free.
    # Polish: Local search to add edges back if possible.
    # This is similar to the original but starting from a dense Q3-free graph.
    
    def creates_q3_fast(u, v, adj):
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    # Local search: try to add missing edges
    missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
    
    current_count = edge_count(adj)
    
    # Greedy add
    random.shuffle(missing_edges)
    for u, v in missing_edges:
        if time.time() > deadline - 5:
            break
        if adj[u] & (1 << v):
            continue
        if not creates_q3_fast(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_count += 1
            
    if current_count > best_count:
        best_count = current_count
        best_adj = adj[:]

    # More aggressive local search: swaps
    # If we can't add, remove an edge and try to add two?
    # Or remove one, add one (net 0) to change structure?
    
    for _ in range(200):
        if time.time() > deadline - 2:
            break
        
        # Try to add any missing edge
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        random.shuffle(missing)
        
        added = False
        for u, v in missing:
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_count += 1
                added = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                break
        
        if added:
            continue
            
        # If stuck, try a swap: remove a random edge, then try to add edges
        existing = [(u, v) for u in range(n) for v in range(u+1, n) if (adj[u] & (1 << v))]
        if not existing:
            break
            
        u_rem, v_rem = random.choice(existing)
        trial = adj[:]
        trial[u_rem] &= ~(1 << v_rem)
        trial[v_rem] &= ~(1 << u_rem)
        
        # Try to add edges to trial
        missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
        random.shuffle(missing_trial)
        
        trial_count = edge_count(trial)
        for u, v in missing_trial:
            if trial[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, trial):
                trial[u] |= (1 << v)
                trial[v] |= (1 << u)
                trial_count += 1
                if trial_count > current_count:
                    # Accept improvement
                    adj = trial[:]
                    current_count = trial_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                    break
                # If equal, we might accept to escape local optimum?
                # But we only add if strictly better or equal?
                # Let's stick to strictly better for stability, or random acceptance.
        
    return edges_from_adj(best_adj)

if __name__ == "__main__":
    # Test
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Edges: {len(edges)}, Time: {end-start:.2f}s")
    
    # Verify
    n = 16
    adj = [0]*n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Check Q3
    def is_q3_induced(subset, adj):
        local_to_global = {i: subset[i] for i in range(8)}
        deg = [0] * 8
        edge_count = 0
        for i in range(8):
            u = subset[i]
            for j in range(i+1, 8):
                v = subset[j]
                if adj[u] & (1 << v):
                    deg[i] += 1
                    deg[j] += 1
                    edge_count += 1
        if edge_count != 12:
            return False
        for d in deg:
            if d != 3:
                return False
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        while queue:
            u_local = queue.pop(0)
            u_global = subset[u_local]
            mask = adj[u_global]
            for v_local in range(8):
                if v_local == u_local:
                    continue
                v_global = subset[v_local]
                if mask & (1 << v_global):
                    if color[v_local] == -1:
                        color[v_local] = 1 - color[u_local]
                        queue.append(v_local)
                    elif color[v_local] == color[u_local]:
                        return False
        return True

    found_q3 = False
    for subset in itertools.combinations(range(n), 8):
        if is_q3_induced(subset, adj):
            found_q3 = True
            break
            
    print(f"Contains Q3: {found_q3}")
