import random
import time

def construct():
    deadline = time.time() + 130
    n = 20
    num_colors = 20
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def get_forbidden_colors_and_impossible(adj, neighbors, u, v):
        """
        Returns (impossible, forbidden_colors) for adding edge (u, v).
        """
        impossible = False
        forbidden_colors = set()
        
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        return True, set()
                    forbidden_colors.add(ca)
                    forbidden_colors.add(cb)
                    forbidden_colors.add(cv)
        
        return False, forbidden_colors

    def greedy_build(seed, start_density=0.0):
        """
        Builds a graph greedily.
        If start_density > 0, it first tries to build a C4-free graph up to that density,
        then continues with standard greedy.
        """
        rng = random.Random(seed)
        edges_list = all_edges[:]
        rng.shuffle(edges_list)
        
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # Phase 1: Build a C4-free core if requested?
        # Actually, just random greedy is fine, but let's try to bias towards C4-free initially?
        # The "bipartite seeding" might imply starting with a bipartite graph.
        # A bipartite graph can have C4s.
        # Let's just use random greedy but with a twist:
        # Try to add edges. If we fail, skip.
        
        for (u, v) in edges_list:
            impossible, forbidden = get_forbidden_colors_and_impossible(adj, neighbors, u, v)
            if not impossible:
                valid_colors = [c for c in range(num_colors) if c not in forbidden]
                if valid_colors:
                    c = rng.choice(valid_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    
        return edges, adj, neighbors

    def local_search(edges, adj, neighbors, seed, max_iters=1000):
        """
        Local search with edge addition, removal, and recoloring.
        """
        rng = random.Random(seed)
        current_edges = edges[:]
        
        # Precompute present set for fast lookup
        present = set((min(u,v), max(u,v)) for u, v, c in current_edges)
        
        # Candidate edges for addition
        candidates = [(u, v) for u, v in all_edges if (u, v) not in present]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # 1. Try to add an edge
            if candidates:
                u, v = rng.choice(candidates)
                impossible, forbidden = get_forbidden_colors_and_impossible(adj, neighbors, u, v)
                if not impossible:
                    valid_colors = [c for c in range(num_colors) if c not in forbidden]
                    if valid_colors:
                        c = rng.choice(valid_colors)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        current_edges.append([u, v, c])
                        present.add((u, v))
                        candidates.remove((u, v))
                        improved = True
                        continue # Restart loop with new state
            
            # 2. If no edge added, try to recolor an existing edge or remove one
            # Recoloring might unlock a forbidden color for another edge.
            if current_edges:
                # Pick a random edge to recolor
                idx = rng.randint(0, len(current_edges) - 1)
                u, v, c_old = current_edges[idx]
                
                # Try to find a new color for this edge
                # The forbidden colors for (u,v) given the *other* edges are the ones that would create a non-rainbow C4.
                # Note: Changing the color of (u,v) affects C4s containing (u,v).
                # It does NOT affect C4s not containing (u,v).
                # So we just need to check if the new color is valid for (u,v) in the current graph.
                # Wait, if we change color of (u,v), we don't need to check if it *creates* a C4, 
                # because the C4s involving (u,v) already exist (the edges are there).
                # We just need the 4 colors on those C4s to be distinct.
                
                # So, for edge (u,v), the forbidden colors are those that match the other 3 edges in any C4 containing (u,v).
                # This is exactly what `get_forbidden_colors_and_impossible` calculates, 
                # BUT it assumes (u,v) is NOT present.
                # If (u,v) IS present, we need to calculate forbidden colors excluding (u,v) itself.
                
                # Let's write a helper to get forbidden colors for an existing edge.
                def get_forbidden_colors_existing(adj, neighbors, u, v):
                    forbidden = set()
                    for a in neighbors[u]:
                        if a == v: continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a: continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                # If ca, cb, cv are distinct, then c_uv cannot be any of them.
                                # If they are not distinct, the graph is already invalid (shouldn't happen if we maintain validity).
                                if ca == cb or cb == cv or ca == cv:
                                    return None # Invalid state
                                forbidden.add(ca)
                                forbidden.add(cb)
                                forbidden.add(cv)
                    return forbidden

                forb = get_forbidden_colors_existing(adj, neighbors, u, v)
                if forb is not None:
                    valid_colors = [c for c in range(num_colors) if c not in forb]
                    if valid_colors and c_old in forb: # If current color is actually invalid? No, it was valid when added.
                        # Maybe a new edge was added that made the old color invalid?
                        # No, adding an edge doesn't invalidate existing edges.
                        # So c_old is always valid.
                        # We want to change it to a *different* color to potentially help others?
                        # No, changing color of (u,v) only affects C4s containing (u,v).
                        # It doesn't affect C4s not containing (u,v).
                        # So recoloring (u,v) cannot help add a new edge (x,y) unless (x,y) is part of a C4 with (u,v).
                        
                        # So, let's just try to recolor to a random valid color.
                        if len(valid_colors) > 1:
                            new_c = rng.choice(valid_colors)
                            if new_c != c_old:
                                current_edges[idx][2] = new_c
                                adj[u][v] = new_c
                                adj[v][u] = new_c
                                improved = True
                                continue
            else:
                # No edges, nothing to do
                pass

            # 3. If still stuck, remove a random edge
            if current_edges:
                idx = rng.randint(0, len(current_edges) - 1)
                u, v, c = current_edges.pop(idx)
                adj[u][v] = -1
                adj[v][u] = -1
                neighbors[u].remove(v)
                neighbors[v].remove(u)
                present.discard((u, v))
                candidates.append((u, v))
                improved = True # Count removal as an improvement step to allow further search

        return current_edges

    best_edges = None
    best_count = 0
    
    # Run multiple restarts
    # The "bipartite seeding" might be better simulated by just running many random restarts
    # and using the stronger local search.
    
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initial build
        edges, adj, neighbors = greedy_build(i)
        
        # Local search
        polished = local_search(edges, adj, neighbors, i, max_iters=200)
        
        if len(polished) > best_count:
            best_count = len(polished)
            best_edges = polished[:]

    if best_edges is None:
        best_edges = [[0, v, 0] for v in range(1, 20)]
        
    return best_edges
