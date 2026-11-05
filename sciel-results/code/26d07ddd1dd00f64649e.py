import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 20
    rng = random.Random(12345)
    
    # Precompute all triples for fast iteration
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    num_triples = len(triples)
    
    # For each vertex, precompute which triples it belongs to, and the other two vertices
    vertex_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        vertex_triples[i].append((idx, j, k))
        vertex_triples[j].append((idx, i, k))
        vertex_triples[k].append((idx, i, j))
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    best_edges = 0
    best_adj = None
    
    # Try multiple random starting graphs
    num_starts = 12
    per_start_time = 8.0  # seconds per start
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Create random starting graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.6:
                    add_edge(adj, i, j)
        
        # Clean: remove edges until valid
        max_clean_iters = 200
        for _ in range(max_clean_iters):
            if time.time() > deadline - 5:
                break
            # Find a violation
            violated = False
            for (i, j, k) in triples:
                common = adj[i] & adj[j] & adj[k]
                mask = (1 << i) | (1 << j) | (1 << k)
                common &= ~mask
                if bin(common).count('1') >= 3:
                    # Remove a random edge between triple and common
                    v_list = [x for x in range(n) if (common >> x) & 1]
                    u_list = [i, j, k]
                    ru = rng.choice(u_list)
                    rv = rng.choice(v_list)
                    remove_edge(adj, ru, rv)
                    violated = True
                    break
            if not violated:
                break
        
        if time.time() > deadline - 5:
            break
        
        # Verify it's valid
        is_valid_now = True
        for (i, j, k) in triples:
            common = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            common &= ~mask
            if bin(common).count('1') >= 3:
                is_valid_now = False
                break
        
        if not is_valid_now:
            continue
        
        cur_edges = count_edges(adj)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
        
        # Greedy edge addition with incremental common-neighbor tracking
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Maintain common neighbor bitmasks for each pair
        # common[i][j] = bitmask of vertices adjacent to both i and j (excluding i,j)
        common = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                cm = adj[i] & adj[j]
                cm &= ~((1 << i) | (1 << j))
                common[i][j] = common[j][i] = cm
        
        # Maintain triple common neighbor bitmasks
        # triple_common[idx] = bitmask of common neighbors of triple idx
        triple_common = [0] * num_triples
        for idx, (i, j, k) in enumerate(triples):
            cm = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            cm &= ~mask
            triple_common[idx] = cm
        
        # Count of triples with >= 3 common neighbors
        violation_count = sum(1 for tc in triple_common if bin(tc).count('1') >= 3)
        
        start_round_time = time.time()
        max_round_time = 6.0
        
        while time.time() < start_round_time + max_round_time and time.time() < deadline - 5:
            rng.shuffle(all_edges)
            improved = False
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                
                # Try adding edge (u, v)
                add_edge(adj, u, v)
                
                # Update common masks for pairs involving u or v
                # For pair (u, x): common neighbors gain v if v adjacent to x
                # For pair (v, x): common neighbors gain u if u adjacent to x
                old_common = {}
                for x in range(n):
                    if x == u or x == v:
                        continue
                    # Update common[u][x] and common[v][x]
                    if (adj[v] >> x) & 1:
                        common[u][x] |= (1 << v)
                        common[x][u] |= (1 << v)
                    if (adj[u] >> x) & 1:
                        common[v][x] |= (1 << u)
                        common[x][v] |= (1 << u)
                
                # Update triple_common for triples containing u or v
                old_tc = {}
                for idx, other1, other2 in vertex_triples[u]:
                    # Triple is (u, other1, other2) or some permutation
                    # Common neighbors of this triple: was triple_common[idx]
                    # Now with u-v edge added: v becomes a common neighbor if v adjacent to other1 and other2
                    if (adj[v] >> other1) & 1 and (adj[v] >> other2) & 1:
                        triple_common[idx] |= (1 << v)
                    old_tc[idx] = triple_common[idx]
                
                for idx, other1, other2 in vertex_triples[v]:
                    if idx in old_tc:
                        continue
                    if (adj[u] >> other1) & 1 and (adj[u] >> other2) & 1:
                        triple_common[idx] |= (1 << u)
                    old_tc[idx] = triple_common[idx]
                
                # Check if any triple now has >= 3 common neighbors
                new_violations = 0
                for idx in old_tc:
                    if bin(triple_common[idx]).count('1') >= 3:
                        new_violations += 1
                
                if new_violations > 0:
                    # Invalid, revert
                    remove_edge(adj, u, v)
                    
                    # Revert common masks
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        if (adj[v] >> x) & 1:
                            common[u][x] &= ~(1 << v)
                            common[x][u] &= ~(1 << v)
                        if (adj[u] >> x) & 1:
                            common[v][x] &= ~(1 << u)
                            common[x][v] &= ~(1 << u)
                    
                    # Revert triple_common
                    for idx in old_tc:
                        if idx in [idx2 for idx2, _, _ in vertex_triples[u]]:
                            if (adj[v] >> [o1 for i2, o1, o2 in vertex_triples[u] if i2 == idx][0]) & 1 and \
                               (adj[v] >> [o2 for i2, o1, o2 in vertex_triples[u] if i2 == idx][0]) & 1:
                                triple_common[idx] &= ~(1 << v)
                        if idx in [idx2 for idx2, _, _ in vertex_triples[v]]:
                            if (adj[u] >> [o1 for i2, o1, o2 in vertex_triples[v] if i2 == idx][0]) & 1 and \
                               (adj[u] >> [o2 for i2, o1, o2 in vertex_triples[v] if i2 == idx][0]) & 1:
                                triple_common[idx] &= ~(1 << u)
                else:
                    # Valid, keep the edge
                    cur_edges += 1
                    improved = True
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_adj = [a for a in adj]
            
            if not improved:
                break
        
        # Local search: edge swaps
        if time.time() < deadline - 10:
            swap_rounds = 0
            max_swap_rounds = 30
            
            while swap_rounds < max_swap_rounds and time.time() < deadline - 5:
                swap_rounds += 1
                rng.shuffle(all_edges)
                any_improved = False
                
                for (u1, v1) in all_edges:
                    if time.time() > deadline - 5:
                        break
                    if not ((adj[u1] >> v1) & 1):
                        continue
                    
                    remove_edge(adj, u1, v1)
                    cur_edges -= 1
                    
                    # Rebuild common and triple_common after removal
                    for i in range(n):
                        for j in range(i+1, n):
                            cm = adj[i] & adj[j]
                            cm &= ~((1 << i) | (1 << j))
                            common[i][j] = common[j][i] = cm
                    
                    for idx, (i, j, k) in enumerate(triples):
                        cm = adj[i] & adj[j] & adj[k]
                        mask = (1 << i) | (1 << j) | (1 << k)
                        cm &= ~mask
                        triple_common[idx] = cm
                    
                    for (u2, v2) in all_edges:
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        if (adj[u2] >> v2) & 1:
                            continue
                        
                        add_edge(adj, u2, v2)
                        
                        # Quick check
                        valid = True
                        for (i, j, k) in triples:
                            cm = adj[i] & adj[j] & adj[k]
                            mask = (1 << i) | (1 << j) | (1 << k)
                            cm &= ~mask
                            if bin(cm).count('1') >= 3:
                                valid = False
                                break
                        
                        if valid:
                            cur_edges += 1
                            any_improved = True
                            if cur_edges > best_edges:
                                best_edges = cur_edges
                                best_adj = [a for a in adj]
                            break
                        else:
                            remove_edge(adj, u2, v2)
                    
                    if not any_improved:
                        # Revert the removal
                        add_edge(adj, u1, v1)
                        cur_edges += 1
                        # Rebuild
                        for i in range(n):
                            for j in range(i+1, n):
                                cm = adj[i] & adj[j]
                                cm &= ~((1 << i) | (1 << j))
                                common[i][j] = common[j][i] = cm
                        for idx, (i, j, k) in enumerate(triples):
                            cm = adj[i] & adj[j] & adj[k]
                            mask = (1 << i) | (1 << j) | (1 << k)
                            cm &= ~mask
                            triple_common[idx] = cm
                    else:
                        # Rebuild after successful swap
                        for i in range(n):
                            for j in range(i+1, n):
                                cm = adj[i] & adj[j]
                                cm &= ~((1 << i) | (1 << j))
                                common[i][j] = common[j][i] = cm
                        for idx, (i, j, k) in enumerate(triples):
                            cm = adj[i] & adj[j] & adj[k]
                            mask = (1 << i) | (1 << j) | (1 << k)
                            cm &= ~mask
                            triple_common[idx] = cm
                        any_improved = False
                        break
                
                if not any_improved:
                    break
    
    # If we never found a valid graph, return empty
    if best_adj is None:
        return []
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
