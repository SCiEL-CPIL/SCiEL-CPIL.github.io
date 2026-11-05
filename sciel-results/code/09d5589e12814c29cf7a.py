import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges)
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x such that 
    # edges (u,v), (v,w), (w,x), (x,u) exist.
    # To avoid duplicates, we can fix u < v < w < x and consider the 3 ways to form a cycle?
    # No, a 4-cycle is a subgraph isomorphic to C4.
    # In K_n, any 4 distinct vertices form 3 distinct 4-cycles (as subgraphs)?
    # Vertices a,b,c,d. Cycles: (a,b,c,d), (a,b,d,c), (a,c,b,d).
    # Actually, the set of edges for a 4-cycle on {a,b,c,d} is:
    # 1. {(a,b), (b,c), (c,d), (d,a)}
    # 2. {(a,b), (b,d), (d,c), (c,a)}
    # 3. {(a,c), (c,b), (b,d), (d,a)}
    # We need to check if the edges in the cycle exist in our graph.
    
    cycles = []
    for comb in itertools.combinations(range(n), 4):
        u, v, w, x = comb
        # 3 distinct 4-cycles on these 4 vertices
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        cycles.append(c1)
        cycles.append(c2)
        cycles.append(c3)
        
    # Optimization: For each edge, store the index of cycles it participates in
    # This helps in fast validity checking if we track which cycles are "active" (all 4 edges present)
    
    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        remaining = all_edges[:]
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            # Check if adding (u,v) with some color is possible
            # A color c is forbidden if there exists a 4-cycle containing (u,v) 
            # and 3 other edges with colors that conflict with c.
            # Specifically, for any 4-cycle (u,v,a,b) [order matters for edge set], 
            # if edges (v,a), (a,b), (b,u) exist, let their colors be c1, c2, c3.
            # Then c cannot be c1, c2, or c3.
            
            forbidden = set()
            feasible = True
            
            # Iterate over common neighbors or just check all potential 4-cycles involving (u,v)?
            # Faster: iterate over pairs (a,b) such that (u,a), (a,b), (b,v) exist?
            # Or iterate over a in neighbors[u] and b in neighbors[v] such that (a,b) exists.
            
            for a in neighbors[u]:
                if a == v: continue
                ca = adj[u][a]
                for b in neighbors[v]:
                    if b == u or b == a: continue
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        # Cycle is u-a-b-v-u? No, edges are (u,a), (a,b), (b,v), (v,u).
                        # Colors: ca, cb, cv, c_new.
                        # Must be distinct.
                        if ca == cb or cb == cv or ca == cv:
                            # This 4-cycle is already invalid in the current graph?
                            # Wait, if the current graph is valid, this cannot happen.
                            # But if we are building greedily, we assume current graph is valid.
                            # If ca, cb, cv are not distinct, the current graph is invalid.
                            # We should ensure current graph is always valid.
                            # If so, ca, cb, cv are distinct.
                            feasible = False
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if not feasible:
                    break
            
            if feasible:
                candidates = [c for c in range(num_colors) if c not in forbidden]
                if candidates:
                    c = rng.choice(candidates)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    
        return edges, adj, neighbors

    def is_valid(adj):
        # Check all precomputed cycles
        for cyc in cycles:
            # Check if all 4 edges exist
            if adj[cyc[0][0]][cyc[0][1]] == -1: continue
            if adj[cyc[1][0]][cyc[1][1]] == -1: continue
            if adj[cyc[2][0]][cyc[2][1]] == -1: continue
            if adj[cyc[3][0]][cyc[3][1]] == -1: continue
            
            c1 = adj[cyc[0][0]][cyc[0][1]]
            c2 = adj[cyc[1][0]][cyc[1][1]]
            c3 = adj[cyc[2][0]][cyc[2][1]]
            c4 = adj[cyc[3][0]][cyc[3][1]]
            
            if len({c1, c2, c3, c4}) != 4:
                return False
        return True

    def try_add_edge(adj, neighbors, u, v, edges, rng):
        """Try to add edge (u,v) with a valid color. Returns True if added."""
        if adj[u][v] != -1:
            return True # Already exists
        
        forbidden = set()
        for a in neighbors[u]:
            if a == v: continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a: continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        return False # Current graph invalid? Should not happen if we maintain validity.
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
        
        candidates = [c for c in range(num_colors) if c not in forbidden]
        if candidates:
            c = rng.choice(candidates)
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
            return True
        return False

    def try_recolor(adj, neighbors, u, v, new_c, edges, edge_idx):
        """Check if recoloring (u,v) to new_c is valid. Returns True if valid."""
        old_c = adj[u][v]
        if old_c == new_c:
            return True
            
        # Check all 4-cycles containing (u,v)
        # A 4-cycle containing (u,v) must have the form (u, v, a, b) where (v,a), (a,b), (b,u) exist.
        # Or (u, a, v, b) ... wait, (u,v) is an edge.
        # The other 3 edges must form a path of length 3 from v to u not using (v,u).
        # So we look for a, b such that (v,a), (a,b), (b,u) exist.
        
        for a in neighbors[v]:
            if a == u: continue
            cva = adj[v][a]
            for b in neighbors[u]:
                if b == v or b == a: continue
                if adj[a][b] != -1:
                    cab = adj[a][b]
                    cbu = adj[b][u]
                    # Cycle: (u,v), (v,a), (a,b), (b,u)
                    # Colors: new_c, cva, cab, cbu
                    if new_c == cva or new_c == cab or new_c == cbu or cva == cab or cab == cbu or cva == cbu:
                        return False
                    # Note: cva, cab, cbu must be distinct in a valid graph.
                    # So we just need new_c not in {cva, cab, cbu}.
        
        # If valid, apply
        adj[u][v] = new_c
        adj[v][u] = new_c
        # Update edge list
        # We need to find the edge in the list. 
        # For speed, we can just return True and let the caller update, or update here.
        # Let's update here for simplicity, assuming edge_idx is valid or we search.
        # Better: pass edges and update.
        # But we don't have direct index. Let's just return True and let the main loop handle it?
        # No, we need to modify adj.
        # We will modify adj and return True. The caller must update the edge list.
        return True

    # Phase 1: Greedy Start
    best_edges = []
    best_adj = None
    best_neighbors = None
    best_count = 0
    
    for i in range(50):
        if time.time() > deadline: break
        edges, adj, neighbors = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = [row[:] for row in adj]
            best_neighbors = [set(s) for s in neighbors]
            
    # Phase 2: Local Search with Recoloring
    # We work on a copy of the best found so far
    if best_adj is None:
        return [[0, 1, 0]]
        
    adj = [row[:] for row in best_adj]
    neighbors = [set(s) for s in best_neighbors]
    edges = best_edges[:]
    
    # Create a set of current edges for fast lookup
    current_edge_set = set((min(u,v), max(u,v)) for u,v,c in edges)
    missing_edges = [e for e in all_edges if e not in current_edge_set]
    
    rng = random.Random(12345)
    
    improved = True
    stall_count = 0
    last_improvement_time = time.time()
    
    while time.time() < deadline - 5:
        if not improved:
            stall_count += 1
            if stall_count > 20:
                # Try a different seed or break
                break
            improved = False
        else:
            stall_count = 0
            last_improvement_time = time.time()
            
        # Shuffle missing edges to try different ones
        if missing_edges:
            rng.shuffle(missing_edges)
        
        added_this_round = False
        recolored_this_round = False
        
        # Try to add missing edges
        for u, v in missing_edges[:]:
            if time.time() > deadline - 5: break
            if try_add_edge(adj, neighbors, u, v, edges, rng):
                # Successfully added
                current_edge_set.add((min(u,v), max(u,v)))
                if (u,v) in missing_edges:
                    missing_edges.remove((u,v))
                elif (v,u) in missing_edges:
                    missing_edges.remove((v,u))
                added_this_round = True
                improved = True
                break # Restart loop with updated state
                
        if added_this_round:
            continue
            
        # If no edge could be added, try to recolor to unlock an addition
        # Pick a random missing edge and see why it failed
        if missing_edges:
            target_u, target_v = rng.choice(missing_edges)
            
            # Find a blocking 4-cycle
            # We need to find a 4-cycle (target_u, target_v, a, b) where 
            # colors of (v,a), (a,b), (b,u) conflict with all available colors for (u,v).
            # Actually, if try_add_edge failed, it means for ALL 4-cycles, the union of forbidden colors
            # covered all 20 colors.
            
            # Strategy: Pick a random existing edge (x,y) and try to recolor it.
            # If recoloring is valid, do it.
            # Then try to add the target edge again.
            
            # To be smarter, pick an edge that is in a 4-cycle with the target edge.
            # Find a 4-cycle involving (target_u, target_v).
            # (target_u, target_v, a, b) => edges (v,a), (a,b), (b,u) exist.
            
            blocking_cycle_edges = []
            found_blocker = False
            for a in neighbors[target_v]:
                if a == target_u: continue
                for b in neighbors[target_u]:
                    if b == target_v or b == a: continue
                    if adj[a][b] != -1:
                        # This is a 4-cycle
                        # Edges: (target_u, target_v), (target_v, a), (a, b), (b, target_u)
                        # We can try to recolor (target_v, a), (a, b), or (b, target_u)
                        blocking_cycle_edges.append((target_v, a))
                        blocking_cycle_edges.append((a, b))
                        blocking_cycle_edges.append((b, target_u))
                        found_blocker = True
                        break
                if found_blocker:
                    break
            
            if found_blocker:
                # Try to recolor one of the edges in the blocking cycle
                rng.shuffle(blocking_cycle_edges)
                for x, y in blocking_cycle_edges:
                    # Try random new colors
                    for new_c in rng.sample(range(num_colors), num_colors):
                        # Check if recoloring (x,y) to new_c is valid
                        # We need to check all 4-cycles containing (x,y)
                        valid = True
                        for p in neighbors[x]:
                            if p == y: continue
                            c_xp = adj[x][p]
                            for q in neighbors[y]:
                                if q == x or q == p: continue
                                if adj[p][q] != -1:
                                    c_pq = adj[p][q]
                                    c_qy = adj[q][y]
                                    # Cycle (x, y, q, p) ?? No, (x, p, q, y)
                                    # Edges: (x,p), (p,q), (q,y), (y,x)
                                    # Colors: c_xp, c_pq, c_qy, new_c
                                    if new_c in {c_xp, c_pq, c_qy} or len({c_xp, c_pq, c_qy}) < 3:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if valid:
                            # Apply recolor
                            old_c = adj[x][y]
                            adj[x][y] = new_c
                            adj[y][x] = new_c
                            # Update edges list
                            # Find (x,y) or (y,x) in edges
                            for i, e in enumerate(edges):
                                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                                    e[2] = new_c
                                    break
                            recolored_this_round = True
                            break
                    if recolored_this_round:
                        break
                if recolored_this_round:
                    # Try to add the target edge again
                    if try_add_edge(adj, neighbors, target_u, target_v, edges, rng):
                        current_edge_set.add((min(target_u,target_v), max(target_u,target_v)))
                        if (target_u, target_v) in missing_edges:
                            missing_edges.remove((target_u, target_v))
                        elif (target_v, target_u) in missing_edges:
                            missing_edges.remove((target_v, target_u))
                        improved = True
                        break # Restart loop

    # Final check and return
    if is_valid(adj):
        return edges
    else:
        # Fallback to best valid found
        return best_edges
