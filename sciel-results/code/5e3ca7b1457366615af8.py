import time
import random
import numpy as np

def construct():
    n = 16
    total_time_limit = 145  # Safety margin for 150s
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)
    
    def build_masks(adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for i in range(n):
                if adj[u, i]:
                    m |= (1 << i)
            masks[u] = m
        return masks
    
    def get_score_full(adj, masks, deg):
        """Full score computation - O(E * avg_triangle_size)"""
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common &= common - 1
        return max_sum
    
    def get_all_triangles(adj, masks, deg):
        """Return list of (sum, u, v, w) for all triangles"""
        tris = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        if w > v:  # Only count each triangle once
                            s = deg[u] + deg[v] + deg[w]
                            tris.append((s, u, v, w))
                        common &= common - 1
        return tris
    
    def incremental_score(adj, masks, deg, removed_edge, added_edge):
        """
        Compute score after removing one edge and adding one non-edge.
        Only triangles affected are those containing removed_edge's endpoints
        or added_edge's endpoints.
        """
        ru, rv = removed_edge
        au, av = added_edge
        
        # Vertices whose degree changed
        changed = {ru, rv, au, av}
        
        # Triangles that could be affected are those containing at least one
        # of the changed vertices AND at least one of the edges that changed
        
        # Strategy: find all triangles in the new graph that contain any changed vertex
        # and check if they exist in the new graph
        
        max_sum = 0
        
        # Check all triangles that might exist in the new graph
        # A triangle (a,b,c) is affected if:
        # - It contains ru or rv (removed edge endpoints) - might be destroyed
        # - It contains au or av (added edge endpoints) - might be created
        # - It contains both removed and added edge endpoints - degree changes
        
        # Simpler: just check all triangles in the new graph
        # But we can be smarter: only triangles containing at least one changed vertex
        # can have different degree sums or existence
        
        # Actually, the cleanest approach: recompute score but only for triangles
        # that contain at least one of the 4 changed vertices
        
        for a in range(n):
            if a not in changed:
                continue
            for b in range(a + 1, n):
                if not adj[a, b]:
                    continue
                if b not in changed and a not in changed:
                    continue
                # Check if (a,b) is an edge and find common neighbors
                common = masks[a] & masks[b]
                # Filter to only consider triangles where at least one vertex is in changed
                # But we need to check all common neighbors for existence
                while common:
                    lsb = common & -common
                    c = lsb.bit_length() - 1
                    if c > b:
                        # This triangle (a,b,c) exists in new graph
                        # Check if it's affected (contains a changed vertex)
                        if a in changed or b in changed or c in changed:
                            s = deg[a] + deg[b] + deg[c]
                            if s > max_sum:
                                max_sum = s
                    common &= common - 1
        
        # But we also need to check triangles NOT containing changed vertices
        # because their degree sums didn't change, but we need the max
        # Actually, if no triangle contains a changed vertex, the score is unchanged
        # from the previous score for those triangles. But we don't track that.
        
        # For correctness, let's just do full recomputation but with early exit
        # if we can prove the bound
        
        # Actually, let me just do full recomputation - it's fast enough for n=16
        # The real speedup comes from doing more iterations
        
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = masks[u] & masks[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common &= common - 1
        return max_sum
    
    def make_initial_graphs():
        """Create diverse initial graphs"""
        graphs = []
        
        # 1. Standard K_{8,8} + (0,1)
        adj1 = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                adj1[u, v] = 1
                adj1[v, u] = 1
        adj1[0, 1] = 1
        adj1[1, 0] = 1
        graphs.append(adj1.copy())
        
        # 2. K_{8,8} + random intra-partition edge
        for seed in range(5):
            rng = random.Random(seed)
            adj2 = np.zeros((n, n), dtype=np.int8)
            for u in range(8):
                for v in range(8, 16):
                    adj2[u, v] = 1
                    adj2[v, u] = 1
            # Add random intra-partition edge
            part = rng.choice([list(range(8)), list(range(8, 16))])
            u, v = rng.sample(part, 2)
            adj2[u, v] = 1
            adj2[v, u] = 1
            graphs.append(adj2.copy())
        
        # 3. Near-complete bipartite with some modifications
        for seed in range(3):
            rng = random.Random(100 + seed)
            adj3 = np.zeros((n, n), dtype=np.int8)
            for u in range(8):
                for v in range(8, 16):
                    adj3[u, v] = 1
                    adj3[v, u] = 1
            # Remove 1 random bipartite edge, add 2 random intra edges
            bip_edges = [(u, v) for u in range(8) for v in range(8, 16)]
            u, v = rng.choice(bip_edges)
            adj3[u, v] = 0
            adj3[v, u] = 0
            
            intra1 = rng.sample(range(8), 2)
            adj3[intra1[0], intra1[1]] = 1
            adj3[intra1[1], intra1[0]] = 1
            intra2 = rng.sample(range(8, 16), 2)
            adj3[intra2[0], intra2[1]] = 1
            adj3[intra2[1], intra2[0]] = 1
            graphs.append(adj3.copy())
        
        # 4. Random 65-edge graphs
        for seed in range(3):
            rng = random.Random(200 + seed)
            adj4 = np.zeros((n, n), dtype=np.int8)
            edges_to_add = rng.sample(all_edges, 65)
            for u, v in edges_to_add:
                adj4[u, v] = 1
                adj4[v, u] = 1
            graphs.append(adj4.copy())
        
        return graphs
    
    start_time = time.time()
    
    # Evaluate all initial graphs and pick the best
    best_adj = None
    best_score = float('inf')
    best_masks = None
    best_deg = None
    
    init_graphs = make_initial_graphs()
    
    for adj in init_graphs:
        masks = build_masks(adj)
        deg = adj.sum(axis=1)
        score = get_score_full(adj, masks, deg)
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_masks = masks
            best_deg = deg.copy()
    
    # Multi-start simulated annealing with different seeds
    all_results = [(best_score, best_adj.copy(), best_masks.copy(), best_deg.copy())]
    
    # Use remaining time for multiple SA runs
    time_per_run = (total_time_limit - (time.time() - start_time)) / 5
    if time_per_run < 10:
        time_per_run = 10
    
    for run_seed in range(5):
        if time.time() - start_time > total_time_limit - 15:
            break
        
        # Use the best found so far as starting point, with some perturbation
        if run_seed == 0:
            cur_adj = best_adj.copy()
            cur_masks = best_masks.copy()
            cur_deg = best_deg.copy()
            cur_score = best_score
        else:
            # Perturb the best graph
            cur_adj = best_adj.copy()
            rng = random.Random(run_seed * 1000)
            for _ in range(rng.randint(1, 3)):
                # Random edge swap
                edges_in = [(u, v) for u in range(n) for v in range(u + 1, n) if cur_adj[u, v]]
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not cur_adj[u, v]]
                if edges_in and non_edges:
                    u, v = rng.choice(edges_in)
                    x, y = rng.choice(non_edges)
                    cur_adj[u, v] = 0
                    cur_adj[v, u] = 0
                    cur_adj[x, y] = 1
                    cur_adj[y, x] = 1
            
            cur_masks = build_masks(cur_adj)
            cur_deg = cur_adj.sum(axis=1)
            cur_score = get_score_full(cur_adj, cur_masks, cur_deg)
        
        run_start = time.time()
        run_limit = run_start + time_per_run
        
        T_start = 3.0
        T_end = 0.05
        total_iters = 100000  # Will be cut by time
        
        for it in range(total_iters):
            if time.time() > run_limit:
                break
            
            frac = it / total_iters
            T = T_start * (T_end / T_start) ** frac
            
            # Choose edge to remove
            # With probability 0.8, pick from worst triangles
            if random.random() < 0.8:
                # Find worst triangles
                tris = get_all_triangles(cur_adj, cur_masks, cur_deg)
                if tris:
                    tris.sort(key=lambda x: -x[0])
                    top_k = min(3, len(tris))
                    worst = random.choice(tris[:top_k])
                    # Pick one of its 3 edges
                    edge_idx = random.randint(0, 2)
                    if edge_idx == 0:
                        u, v = worst[1], worst[2]
                    elif edge_idx == 1:
                        u, v = worst[1], worst[3]
                    else:
                        u, v = worst[2], worst[3]
                    if u > v:
                        u, v = v, u
                else:
                    edges_in = [(a, b) for a in range(n) for b in range(a + 1, n) if cur_adj[a, b]]
                    u, v = random.choice(edges_in)
            else:
                edges_in = [(a, b) for a in range(n) for b in range(a + 1, n) if cur_adj[a, b]]
                u, v = random.choice(edges_in)
            
            # Choose non-edge to add
            # Prefer low-degree vertices
            non_edges = [(a, b) for a in range(n) for b in range(a + 1, n) if not cur_adj[a, b]]
            if not non_edges:
                break
            
            # Weight by inverse degree sum
            weights = [1.0 / (1.0 + cur_deg[a] + cur_deg[b]) for a, b in non_edges]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            x, y = non_edges[0]
            for (a, b), w in zip(non_edges, weights):
                cum += w
                if r <= cum:
                    x, y = a, b
                    break
            
            # Apply move
            cur_adj[u, v] = 0
            cur_adj[v, u] = 0
            cur_adj[x, y] = 1
            cur_adj[y, x] = 1
            
            cur_masks[u] &= ~(1 << v)
            cur_masks[v] &= ~(1 << u)
            cur_masks[x] |= (1 << y)
            cur_masks[y] |= (1 << x)
            
            cur_deg[u] -= 1
            cur_deg[v] -= 1
            cur_deg[x] += 1
            cur_deg[y] += 1
            
            new_score = get_score_full(cur_adj, cur_masks, cur_deg)
            
            # Accept/reject
            if new_score < cur_score:
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj = cur_adj.copy()
                    best_masks = cur_masks.copy()
                    best_deg = cur_deg.copy()
                    all_results.append((best_score, best_adj.copy(), best_masks.copy(), best_deg.copy()))
            else:
                delta = new_score - cur_score
                if random.random() < np.exp(-delta / T):
                    cur_score = new_score
                else:
                    # Revert
                    cur_adj[u, v] = 1
                    cur_adj[v, u] = 1
                    cur_adj[x, y] = 0
                    cur_adj[y, x] = 0
                    cur_masks[u] |= (1 << v)
                    cur_masks[v] |= (1 << u)
                    cur_masks[x] &= ~(1 << y)
                    cur_masks[y] &= ~(1 << x)
                    cur_deg[u] += 1
                    cur_deg[v] += 1
                    cur_deg[x] -= 1
                    cur_deg[y] -= 1
        
        # After each SA run, do a quick local search polish
        if time.time() - start_time < total_time_limit - 10:
            polish_adj = best_adj.copy()
            polish_masks = build_masks(polish_adj)
            polish_deg = polish_adj.sum(axis=1)
            polish_score = get_score_full(polish_adj, polish_masks, polish_deg)
            
            polish_deadline = min(time.time() + 5, start_time + total_time_limit - 5)
            improved = True
            while improved and time.time() < polish_deadline:
                improved = False
                for u, v in all_edges:
                    if time.time() > polish_deadline:
                        break
                    if not polish_adj[u, v]:
                        continue
                    for x, y in all_edges:
                        if time.time() > polish_deadline:
                            break
                        if x == u or x == v or y == u or y == v:
                            continue
                        if polish_adj[x, y]:
                            continue
                        
                        # Try swap
                        polish_adj[u, v] = 0
                        polish_adj[v, u] = 0
                        polish_adj[x, y] = 1
                        polish_adj[y, x] = 1
                        
                        polish_masks[u] &= ~(1 << v)
                        polish_masks[v] &= ~(1 << u)
                        polish_masks[x] |= (1 << y)
                        polish_masks[y] |= (1 << x)
                        
                        polish_deg[u] -= 1
                        polish_deg[v] -= 1
                        polish_deg[x] += 1
                        polish_deg[y] += 1
                        
                        new_ps = get_score_full(polish_adj, polish_masks, polish_deg)
                        
                        if new_ps < polish_score:
                            polish_score = new_ps
                            improved = True
                        else:
                            polish_adj[u, v] = 1
                            polish_adj[v, u] = 1
                            polish_adj[x, y] = 0
                            polish_adj[y, x] = 0
                            polish_masks[u] |= (1 << v)
                            polish_masks[v] |= (1 << u)
                            polish_masks[x] &= ~(1 << y)
                            polish_masks[y] &= ~(1 << x)
                            polish_deg[u] += 1
                            polish_deg[v] += 1
                            polish_deg[x] -= 1
                            polish_deg[y] -= 1
            
            if polish_score < best_score:
                best_score = polish_score
                best_adj = polish_adj
                best_masks = polish_masks
                best_deg = polish_deg
    
    # Final: convert best to edges list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
    
    return {'n': 16, 'edges': best_edges}
