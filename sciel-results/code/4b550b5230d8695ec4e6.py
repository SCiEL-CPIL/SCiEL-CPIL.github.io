def construct():
    import random
    import time

    start_time = time.time()
    deadline = start_time + 115

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def check_4cycle_conflict(adj, neighbors, u, v, c):
        """
        Check if adding edge (u,v) with color c creates a non-rainbow 4-cycle.
        Returns True if conflict exists (invalid), False if valid.
        """
        # A 4-cycle containing (u,v) is u-v-x-y-u
        # x must be in neighbors[v] (x != u)
        # y must be in neighbors[u] (y != v, y != x)
        # (x,y) must be an edge (adj[x][y] != -1)
        # Colors: c (uv), adj[v][x] (vx), adj[x][y] (xy), adj[y][u] (yu)
        # Must be 4 distinct colors.

        # Optimization: Iterate over the smaller neighbor list first?
        # Actually, the structure is asymmetric in the check (x from v, y from u).
        # But we can just iterate.
        
        # Pre-fetch lists for speed
        nbrs_v = neighbors[v]
        nbrs_u = neighbors[u]
        
        for x in nbrs_v:
            if x == u:
                continue
            cvx = adj[v][x]
            
            # If cvx == c, then any valid cycle with this x is immediately invalid
            # because c and cvx are on the cycle.
            if cvx == c:
                # Check if there exists ANY y in nbrs_u such that y != v, y != x, and adj[x][y] != -1
                # If such y exists, we have a conflict.
                for y in nbrs_u:
                    if y == v or y == x:
                        continue
                    if adj[x][y] != -1:
                        return True
                # If no such y exists, this x doesn't create a conflict with c==cvx
                # (because no cycle is formed)
                continue

            # cvx != c. Check specific cycles.
            for y in nbrs_u:
                if y == v or y == x:
                    continue
                if adj[x][y] != -1:
                    cxy = adj[x][y]
                    cyu = adj[y][u]
                    
                    # Check if {c, cvx, cxy, cyu} has size 4
                    # Fast path: check pairwise equality
                    if c == cxy or c == cyu:
                        return True
                    if cvx == cxy or cvx == cyu:
                        return True
                    if cxy == cyu:
                        return True
        return False

    def try_random_construction(seed):
        """Try one random construction."""
        rng = random.Random(seed)
        edges_order = all_edges[:]
        rng.shuffle(edges_order)
        
        adj = [[-1]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        current_edges = []
        
        for u, v in edges_order:
            valid_color = -1
            colors = list(range(20))
            rng.shuffle(colors)
            
            for c in colors:
                if not check_4cycle_conflict(adj, neighbors, u, v, c):
                    valid_color = c
                    break
            
            if valid_color != -1:
                adj[u][v] = valid_color
                adj[v][u] = valid_color
                neighbors[u].append(v)
                neighbors[v].append(u)
                current_edges.append([u, v, valid_color])
        
        return current_edges, adj, neighbors

    def try_improve(edges, adj, neighbors, seed):
        """
        Structural improvement: 
        1. Identify a set of "expensive" edges to remove (e.g., edges that block many others).
        2. Remove them.
        3. Re-run a greedy fill on the remaining graph + removed edges to see if we get denser.
        
        A simpler local search: 
        Pick a random edge to remove. Then try to add back ALL missing edges greedily.
        If the result is better, keep it.
        """
        if time.time() > deadline - 5:
            return edges[:]

        rng = random.Random(seed)
        
        # We will try to improve by removing a random subset of edges and re-filling.
        # Let's try removing 1 random edge first.
        if not edges:
            return edges[:]
        
        idx = rng.randrange(len(edges))
        u, v, c = edges[idx]
        
        # Temporarily remove
        new_adj = [row[:] for row in adj]
        new_neighbors = [lst[:] for lst in neighbors]
        
        new_adj[u][v] = -1
        new_adj[v][u] = -1
        new_neighbors[u].remove(v)
        new_neighbors[v].remove(u)
        
        # The set of edges currently in the graph (minus the one removed)
        current_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        current_edge_set.discard((min(u, v), max(u, v)))
        
        # The set of all possible edges
        all_edge_set = set(all_edges)
        
        # Missing edges are those in all_edge_set but not in current_edge_set
        missing_edges = [e for e in all_edges if e not in current_edge_set]
        
        # Shuffle missing edges to add in random order
        rng.shuffle(missing_edges)
        
        added_count = 0
        new_edges_list = [e[:] for e in edges if e != [u, v, c]] # Copy remaining
        
        # Try to add back the removed edge and all missing edges
        # We prioritize the removed edge? Or just mix them?
        # Let's put the removed edge at the front of the list to try to re-add it, 
        # but with a potentially different color.
        
        candidates = [(u, v)] + missing_edges
        # Shuffle candidates? No, we want to try the removed one first maybe?
        # Actually, random order is better for escaping local minima.
        rng.shuffle(candidates)
        
        for a, b in candidates:
            if new_adj[a][b] != -1:
                continue # Already added (if it was in missing and added earlier? No, missing was disjoint)
            
            # Try colors
            colors = list(range(20))
            rng.shuffle(colors)
            for c_new in colors:
                if not check_4cycle_conflict(new_adj, new_neighbors, a, b, c_new):
                    new_adj[a][b] = c_new
                    new_adj[b][a] = c_new
                    new_neighbors[a].append(b)
                    new_neighbors[b].append(a)
                    new_edges_list.append([a, b, c_new])
                    added_count += 1
                    break
        
        # Compare
        if len(new_edges_list) > len(edges):
            return new_edges_list
        
        return edges[:]

    best_edges = []
    seed = 0
    
    # Phase 1: Random restarts
    while time.time() < deadline - 20:
        edges, adj, neighbors = try_random_construction(seed)
        
        if len(edges) > len(best_edges):
            best_edges = edges[:]
            best_adj = [row[:] for row in adj]
            best_neighbors = [lst[:] for lst in neighbors]
            
        seed += 1
        
        # Phase 2: Local improvement on the current best if we have time
        # We only improve the global best, not every local candidate, to save time?
        # Or improve the current candidate if it's close to best?
        if time.time() < deadline - 10:
            # Try to improve the current best_edges
            # We need adj and neighbors for best_edges. 
            # If we just found a new best, we have them. 
            # If not, we don't have adj/neighbors for best_edges easily unless we store them.
            
            # Let's store best_adj and best_neighbors globally.
            if len(edges) >= len(best_edges):
                # If this is the best, we have adj/neighbors from this run
                if time.time() < deadline - 5:
                    improved = try_improve(edges, adj, neighbors, seed * 100)
                    if len(improved) > len(best_edges):
                        best_edges = improved
                        # Note: We don't have the adj/neighbors for 'improved' directly 
                        # unless we return them. 
                        # For simplicity, let's just update best_edges. 
                        # Next iteration of improvement will need adj/neighbors.
                        # To do this properly, we should return adj/neighbors from try_improve.
                        
    # To make the improvement loop effective, try_improve should return adj/neighbors too.
    # Let's refactor slightly to ensure we can chain improvements if needed, 
    # but given the 120s limit, a few random restarts + one improvement pass on the best found
    # is likely the most robust.
    
    # Final check: if we have time, try to improve the absolute best_edges found so far.
    # We need to reconstruct adj/neighbors for best_edges or have stored them.
    # Let's store the best adj/neighbors during the main loop.
    
    # Re-doing the logic to be cleaner:
    
    # --- Revised Main Loop ---
    
    best_edges = []
    best_adj = None
    best_neighbors = None
    seed = 0
    
    while time.time() < deadline - 15:
        edges, adj, neighbors = try_random_construction(seed)
        
        if len(edges) > len(best_edges):
            best_edges = edges
            best_adj = adj
            best_neighbors = neighbors
            
        seed += 1
        
        # If we have found a good solution and have time, try to improve it
        if len(best_edges) > 0 and time.time() < deadline - 5:
            # Use a different seed for improvement
            imp_seed = seed * 7919
            improved_edges, improved_adj, improved_neighbors = try_improve_with_state(best_edges, best_adj, best_neighbors, imp_seed)
            
            if len(improved_edges) > len(best_edges):
                best_edges = improved_edges
                best_adj = improved_adj
                best_neighbors = improved_neighbors

    return best_edges

    # Wait, I defined try_improve but used try_improve_with_state. 
    # I need to implement try_improve_with_state or fix try_improve to return state.
    
    # Let's redefine the helper inside construct to return state.

def construct():
    import random
    import time

    start_time = time.time()
    deadline = start_time + 115

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def check_4cycle_conflict(adj, neighbors, u, v, c):
        """
        Check if adding edge (u,v) with color c creates a non-rainbow 4-cycle.
        Returns True if conflict exists (invalid), False if valid.
        """
        nbrs_v = neighbors[v]
        nbrs_u = neighbors[u]
        
        for x in nbrs_v:
            if x == u:
                continue
            cvx = adj[v][x]
            
            if cvx == c:
                for y in nbrs_u:
                    if y == v or y == x:
                        continue
                    if adj[x][y] != -1:
                        return True
                continue

            for y in nbrs_u:
                if y == v or y == x:
                    continue
                if adj[x][y] != -1:
                    cxy = adj[x][y]
                    cyu = adj[y][u]
                    
                    if c == cxy or c == cyu:
                        return True
                    if cvx == cxy or cvx == cyu:
                        return True
                    if cxy == cyu:
                        return True
        return False

    def try_random_construction(seed):
        """Try one random construction."""
        rng = random.Random(seed)
        edges_order = all_edges[:]
        rng.shuffle(edges_order)
        
        adj = [[-1]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        current_edges = []
        
        for u, v in edges_order:
            valid_color = -1
            colors = list(range(20))
            rng.shuffle(colors)
            
            for c in colors:
                if not check_4cycle_conflict(adj, neighbors, u, v, c):
                    valid_color = c
                    break
            
            if valid_color != -1:
                adj[u][v] = valid_color
                adj[v][u] = valid_color
                neighbors[u].append(v)
                neighbors[v].append(u)
                current_edges.append([u, v, valid_color])
        
        return current_edges, adj, neighbors

    def try_improve(edges, adj, neighbors, seed):
        """
        Attempt to improve the graph by removing one random edge and re-filling.
        Returns (new_edges, new_adj, new_neighbors).
        """
        if time.time() > deadline - 2:
            return edges, adj, neighbors

        rng = random.Random(seed)
        
        if not edges:
            return edges, adj, neighbors

        # Pick a random edge to remove
        idx = rng.randrange(len(edges))
        u, v, c = edges[idx]
        
        # Create deep copies
        new_adj = [row[:] for row in adj]
        new_neighbors = [lst[:] for lst in neighbors]
        
        # Remove edge (u,v)
        new_adj[u][v] = -1
        new_adj[v][u] = -1
        new_neighbors[u].remove(v)
        new_neighbors[v].remove(u)
        
        # Identify current edges (as set of tuples)
        current_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        current_edge_set.discard((min(u, v), max(u, v)))
        
        # Identify missing edges
        missing_edges = [e for e in all_edges if e not in current_edge_set]
        
        # Candidates: the removed edge + all missing edges
        candidates = [(u, v)] + missing_edges
        rng.shuffle(candidates)
        
        new_edges_list = [e[:] for e in edges if e != [u, v, c]]
        
        for a, b in candidates:
            if new_adj[a][b] != -1:
                continue
            
            colors = list(range(20))
            rng.shuffle(colors)
            for c_new in colors:
                if not check_4cycle_conflict(new_adj, new_neighbors, a, b, c_new):
                    new_adj[a][b] = c_new
                    new_adj[b][a] = c_new
                    new_neighbors[a].append(b)
                    new_neighbors[b].append(a)
                    new_edges_list.append([a, b, c_new])
                    break
        
        if len(new_edges_list) > len(edges):
            return new_edges_list, new_adj, new_neighbors
        
        return edges, adj, neighbors

    best_edges = []
    best_adj = None
    best_neighbors = None
    seed = 0
    
    while time.time() < deadline - 10:
        edges, adj, neighbors = try_random_construction(seed)
        
        if len(edges) > len(best_edges):
            best_edges = edges
            best_adj = adj
            best_neighbors = neighbors
            
        seed += 1
        
        # Try to improve the best found so far
        if len(best_edges) > 0 and time.time() < deadline - 2:
            # Use a derived seed for variation
            imp_seed = seed * 104729
            improved_edges, improved_adj, improved_neighbors = try_improve(best_edges, best_adj, best_neighbors, imp_seed)
            
            if len(improved_edges) > len(best_edges):
                best_edges = improved_edges
                best_adj = improved_adj
                best_neighbors = improved_neighbors

    return best_edges
