import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for 150s limit

    def _check_valid(adj, n):
        """
        Returns True if graph is triangle-free and alpha < 10.
        adj: list of bitmasks
        """
        # 1. Check Triangle Free
        for u in range(n):
            # For each neighbor v > u
            v_mask = adj[u] & (1 << (n)) - 1 # Not needed, adj[u] only has bits < n
            # Iterate neighbors
            # Optimization: If adj[u] & adj[v] is non-zero for any v in adj[u], triangle exists.
            # We can just check for all u, v in adj[u] with v > u:
            # Actually, simpler: for all u, check if adj[u] & adj[v] != 0 for any v in adj[u]?
            # No, that checks triangles containing u and v.
            # Let's just do: for u in range(n): for v in adj[u]: if v > u and (adj[u] & adj[v]): return False
            # This is O(E * avg_degree).
            
            # Faster triangle check:
            # A graph is triangle-free iff for every edge (u,v), adj[u] & adj[v] == 0.
            # We can iterate u, then v in adj[u].
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        
        # 2. Check Alpha < 10
        # Branch and bound to find if there is an independent set of size 10.
        # If we find one, return False. If we exhaust search, return True.
        
        # Pre-compute degrees in the whole graph? No, need induced subgraph degrees.
        
        def find_independent_set(candidates, current_size):
            # candidates: bitmask of vertices available
            if current_size == 10:
                return True
            
            if not candidates:
                return False
            
            # Pruning: max possible size from candidates is popcount(candidates)
            if current_size + candidates.bit_count() < 10:
                return False
            
            # Heuristic: Pick vertex with lowest degree in the induced subgraph defined by candidates
            # to maximize the chance of finding a large independent set.
            # Or highest degree to prune? 
            # If we want to *find* a set of size 10, picking low degree vertices in the candidate set
            # allows us to keep more candidates for the next step.
            
            # Get list of vertices in candidates
            verts = []
            temp_cand = candidates
            while temp_cand:
                lsb = temp_cand & -temp_cand
                v = lsb.bit_length() - 1
                verts.append(v)
                temp_cand ^= lsb
            
            if not verts:
                return False

            # Calculate degree of each v in the induced subgraph of candidates
            # deg(v) = popcount(adj[v] & candidates)
            # Sort by degree ascending
            verts.sort(key=lambda v: (adj[v] & candidates).bit_count())
            
            # Branch on the first vertex (lowest degree)
            v = verts[0]
            
            # Branch 1: Include v
            # New candidates: candidates without v and without its neighbors
            new_cands = candidates & ~(adj[v] | (1 << v))
            if find_independent_set(new_cands, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            new_cands2 = candidates & ~(1 << v)
            if find_independent_set(new_cands2, current_size):
                return True
            
            return False

        all_vertices = (1 << n) - 1
        if find_independent_set(all_vertices, 0):
            return False
        return True

    def _add_safe_edge(adj, n):
        """
        Finds a pair (u, v) that can be connected without creating a triangle.
        Returns (u, v) or None.
        """
        # We want to add an edge (u, v) such that adj[u] & adj[v] == 0.
        # And u, v are not already connected.
        # To break an independent set, we specifically want to pick u, v from that set.
        # But here we just pick any safe edge to increase density.
        
        # Iterate all pairs? O(n^2). n=40 is small.
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1): # Not connected
                    if (adj[u] & adj[v]) == 0: # Safe
                        return (u, v)
        return None

    def _find_independent_set_of_size_10(adj, n):
        """
        Returns a list of 10 vertices forming an independent set, or None.
        """
        def backtrack(candidates, current_set):
            if len(current_set) == 10:
                return current_set
            if not candidates:
                return None
            
            verts = []
            temp_cand = candidates
            while temp_cand:
                lsb = temp_cand & -temp_cand
                v = lsb.bit_length() - 1
                verts.append(v)
                temp_cand ^= lsb
            
            if not verts:
                return None

            # Heuristic: lowest degree in candidates
            verts.sort(key=lambda v: (adj[v] & candidates).bit_count())
            
            for v in verts:
                # Try including v
                new_cands = candidates & ~(adj[v] | (1 << v))
                result = backtrack(new_cands, current_set + [v])
                if result:
                    return result
                
                # If we haven't found it including v, try excluding v?
                # Standard backtracking:
                # Actually, if we just iterate, we need to be careful.
                # Let's stick to the previous logic which was simpler but maybe slower?
                # No, the previous logic in _check_valid was just boolean.
                # Here we need the set to modify it.
                
                # Let's refine: 
                # If we don't include v, we recurse on candidates without v.
                # But we are iterating v in a loop. This is not standard backtracking.
                # Standard: 
                # 1. Include v: recurse(new_cands, set+[v])
                # 2. Exclude v: recurse(cands - v, set)
                # We must do BOTH if the first fails.
                
                # Let's rewrite backtrack properly.
                pass
                
        # Rewrite backtrack to be correct
        def backtrack2(candidates, current_set):
            if len(current_set) == 10:
                return current_set
            if not candidates:
                return None
            
            # Pruning
            if len(current_set) + candidates.bit_count() < 10:
                return None

            verts = []
            temp_cand = candidates
            while temp_cand:
                lsb = temp_cand & -temp_cand
                v = lsb.bit_length() - 1
                verts.append(v)
                temp_cand ^= lsb
            
            if not verts:
                return None

            # Pick vertex with lowest degree in candidates
            v = min(verts, key=lambda x: (adj[x] & candidates).bit_count())
            
            # Branch 1: Include v
            new_cands = candidates & ~(adj[v] | (1 << v))
            res = backtrack2(new_cands, current_set + [v])
            if res:
                return res
            
            # Branch 2: Exclude v
            new_cands2 = candidates & ~(1 << v)
            res = backtrack2(new_cands2, current_set)
            if res:
                return res
            
            return None

        all_vertices = (1 << n) - 1
        return backtrack2(all_vertices, [])

    def try_build(n, time_limit):
        if time.time() > start_time + time_limit:
            return None
            
        # Try a few different random seeds/initializations
        for attempt in range(100):
            if time.time() > start_time + time_limit:
                return None
            
            # Initialize random triangle-free graph
            adj = [0] * n
            # Greedy addition of edges
            # To make it denser, we can try to add edges in a random order
            # and check if safe.
            
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
            random.shuffle(edges)
            
            for u, v in edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Local Search
            for _ in range(500): # Max iterations
                if time.time() > start_time + time_limit:
                    return None
                
                # Check validity
                if _check_valid(adj, n):
                    # Success
                    final_edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                final_edges.append([u, v])
                    return {'n': n, 'edges': final_edges}
                
                # Find an independent set of size 10 to break
                indep_set = _find_independent_set_of_size_10(adj, n)
                
                if not indep_set:
                    # This shouldn't happen if _check_valid returned False due to alpha
                    # But _check_valid also checks triangles. 
                    # If _check_valid is False, it's either triangle or alpha.
                    # We should check which one failed.
                    # For efficiency, let's assume if _check_valid is False, we need to fix it.
                    # Let's re-verify if it's triangle or alpha.
                    is_tf = True
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                if adj[u] & adj[v]:
                                    is_tf = False
                                    break
                        if not is_tf:
                            break
                    
                    if not is_tf:
                        # Remove a random edge from a triangle
                        # Find triangle
                        for u in range(n):
                            for v in range(u + 1, n):
                                if (adj[u] >> v) & 1 and (adj[u] & adj[v]):
                                    w = (adj[u] & adj[v]).bit_length() - 1
                                    # Remove edge (u, w) or (v, w) or (u, v)
                                    # Randomly pick one
                                    r = random.random()
                                    if r < 0.33:
                                        adj[u] &= ~(1 << w)
                                        adj[w] &= ~(1 << u)
                                    elif r < 0.66:
                                        adj[v] &= ~(1 << w)
                                        adj[w] &= ~(1 << v)
                                    else:
                                        adj[u] &= ~(1 << v)
                                        adj[v] &= ~(1 << u)
                                    break
                            if not is_tf:
                                break
                        continue

                    # If it's alpha issue, but _find_independent_set_of_size_10 returned None?
                    # This implies a bug in the checker or the finder.
                    # Let's just break and restart.
                    break
                
                # We have an independent set of size 10.
                # Try to add an edge between two vertices in this set that is safe.
                added = False
                for i in range(10):
                    for j in range(i + 1, 10):
                        u = indep_set[i]
                        v = indep_set[j]
                        if not ((adj[u] >> v) & 1): # Not connected
                            if (adj[u] & adj[v]) == 0: # Safe
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                added = True
                                break
                    if added:
                        break
                
                if not added:
                    # No safe edge in the independent set.
                    # Try to add a safe edge anywhere?
                    # Or remove a random edge to allow restructuring?
                    # Removing an edge might reduce alpha, which is good.
                    # But it also reduces density.
                    # Let's try to add a random safe edge anywhere first.
                    safe_edge = _add_safe_edge(adj, n)
                    if safe_edge:
                        u, v = safe_edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        # Graph is maximal triangle-free (with respect to edge addition).
                        # If alpha is still >= 10, we are stuck with this topology.
                        # Restart with a new random graph.
                        break

        return None

    # Try n=40
    result = try_build(40, 50)
    if result:
        return result

    # Try n=39
    if time.time() < deadline:
        result = try_build(39, 60)
        if result:
            return result

    # Try n=38
    if time.time() < deadline:
        result = try_build(38, 20)
        if result:
            return result

    # Fallback: Known small graph
    # K_{9,9} is n=18, alpha=9.
    n = 18
    edges = [[i, j] for i in range(9) for j in range(9, 18)]
    return {'n': n, 'edges': edges}
