"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_initial_paths(edges):
        """Greedy longest path decomposition to get a decent starting point."""
        if not edges:
            return []
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        while edge_set:
            # Pick a random start with highest degree
            deg = defaultdict(int)
            for u, v in edge_set:
                deg[u] += 1
                deg[v] += 1
            if not deg:
                break
            max_d = max(deg.values())
            candidates = [v for v, d in deg.items() if d == max_d]
            start = random.choice(candidates)
            
            path = [start]
            visited = {start}
            last = start
            
            while True:
                nbrs = []
                for nb in adj[last]:
                    if nb in visited:
                        continue
                    e = tuple(sorted((last, nb)))
                    if e in edge_set:
                        # Count onward moves for nb to prefer dead ends (Warnsdorff-like)
                        onward = 0
                        for nn in adj[nb]:
                            if nn in visited or nn == last:
                                continue
                            e2 = tuple(sorted((nb, nn)))
                            if e2 in edge_set:
                                onward += 1
                        nbrs.append((onward, nb))
                
                if not nbrs:
                    break
                
                # Pick neighbor with fewest onward moves
                min_onward = min(x[0] for x in nbrs)
                best_nbs = [nb for o, nb in nbrs if o == min_onward]
                nxt = random.choice(best_nbs)
                
                e = tuple(sorted((last, nxt)))
                edge_set.remove(e)
                path.append(nxt)
                visited.add(nxt)
                last = nxt
            
            if len(path) >= 2:
                paths.append(path)
            else:
                # Fallback for isolated edges if logic fails (shouldn't happen)
                for e in edge_set:
                    paths.append(list(e))
                edge_set = set()
                break
                
        return paths

    def solve_sa(edges, time_limit=10.0):
        if not edges:
            return []
        
        edge_set = set(edges)
        
        # Build adjacency for quick lookup
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Initial solution
        current_paths = get_initial_paths(edge_set)
        current_cost = len(current_paths)
        
        # SA Parameters
        t_start = time.time()
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        # We want to spend most of the time here.
        # Let's define steps based on time.
        # Estimate steps per second: ~10,000 - 50,000 depending on graph size.
        # Let's just loop until time limit.
        
        best_paths = [p[:] for p in current_paths]
        best_cost = current_cost
        
        rng = random.Random(42)
        
        # Precompute edge sets for each path for fast membership? 
        # No, paths are small. 
        # We need to check if an edge exists in the graph.
        
        def get_path_edges(path):
            return {tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)}
        
        # For efficiency, maintain a set of all edges in current solution? 
        # It's always edge_set.
        
        # To check if a merge is valid:
        # Path P1 ends at u, Path P2 starts at v.
        # If (u,v) is in edge_set, we can merge P1 + P2 (if no vertex conflict).
        # Vertex conflict: if any vertex in P1 is in P2, merge is invalid.
        
        # To check if a split is valid:
        # Just remove an edge from a path. Always valid if path length > 1.
        
        # Optimization: 
        # Maintain a dictionary: vertex -> list of path indices containing it?
        # Or just check sets.
        # Graphs are small (n<=30). Paths are short. 
        # Checking vertex intersection: O(len(P1)*len(P2)).
        
        # Let's optimize merge check:
        # If we merge P1 and P2, the new path is P1 + P2.
        # This is valid iff:
        # 1. Edge (end(P1), start(P2)) exists.
        # 2. No vertex in P1 is in P2.
        
        # To make "No vertex in P1 is in P2" fast:
        # We can store the set of vertices for each path.
        
        current_path_vertex_sets = [set(p) for p in current_paths]
        
        while time.time() - t_start < time_limit:
            T = max(T * alpha, T_min)
            
            # Choose a move
            # P(Split) = 0.5, P(Merge) = 0.5
            # Actually, we want to reduce path count, so Merge is the goal.
            # But we need Split to escape local minima.
            
            if len(current_paths) == 0:
                break
                
            if rng.random() < 0.5:
                # MERGE
                if len(current_paths) < 2:
                    continue
                
                i, j = rng.sample(range(len(current_paths)), 2)
                p1 = current_paths[i]
                p2 = current_paths[j]
                
                # Try all 4 combinations of endpoints
                # (start1, end1) and (start2, end2)
                # 1. end1 - start2
                # 2. end1 - end2 (reverse p2)
                # 3. start1 - start2 (reverse p1)
                # 4. start1 - end2 (reverse p1, reverse p2)
                
                # Helper to check merge
                def try_merge(p_a, p_b, reverse_a, reverse_b):
                    if reverse_a:
                        p_a = p_a[::-1]
                    if reverse_b:
                        p_b = p_b[::-1]
                    
                    end_a = p_a[-1]
                    start_b = p_b[0]
                    e = tuple(sorted((end_a, start_b)))
                    
                    if e not in edge_set:
                        return None
                    
                    # Check vertex overlap
                    # Using sets is faster than list intersection for small lists?
                    # set(p_a) & set(p_b)
                    # But we have current_path_vertex_sets.
                    # We need to map indices to sets.
                    # This is tricky with reversals.
                    # For simplicity, just use sets.
                    if set(p_a) & set(p_b):
                        return None
                    
                    return p_a + p_b[1:] # p_b[0] is same as end_a? No, p_a[-1] is end_a.
                                         # The edge connects end_a and start_b.
                                         # So the path is p_a + p_b[1:]? 
                                         # Wait. p_a = [v1...vk], p_b = [wk...].
                                         # Edge is (vk, wk).
                                         # New path: [v1...vk, wk...].
                                         # p_b[0] is wk.
                                         # So yes, p_a + p_b[1:] is WRONG. 
                                         # p_a is [v1...vk]. p_b is [wk...].
                                         # The vertex wk is NOT in p_a.
                                         # So the path is p_a + p_b.
                                         # BUT, the edge is (vk, wk).
                                         # So the sequence of vertices is v1, ..., vk, wk, ...
                                         # So it is p_a + p_b.
                                         # Wait, if I append p_b, I get ..., vk, wk, ...
                                         # Yes.
                                         # But I must ensure I don't duplicate the vertex at the junction.
                                         # p_a ends with vk. p_b starts with wk.
                                         # They are distinct (checked by overlap).
                                         # So p_a + p_b is correct.
                                         # Wait, earlier I said p_b[1:]. Why?
                                         # If I reverse p_b, p_b becomes [wk...].
                                         # If I don't reverse, p_b is [wk...].
                                         # So it's always p_a + p_b.
                                         # Let's re-verify.
                                         # Path A: [1, 2, 3]. Path B: [4, 5].
                                         # Edge (3,4).
                                         # Merged: [1, 2, 3, 4, 5].
                                         # p_a + p_b = [1,2,3] + [4,5] = [1,2,3,4,5]. Correct.
                                         # What if we reverse B?
                                         # Path B reversed: [5, 4].
                                         # Edge (3,5)? No, we check edge (end_a, start_b_reversed).
                                         # If we reverse B, start_b_reversed is 5.
                                         # Edge (3,5).
                                         # Merged: [1, 2, 3, 5, 4].
                                         # p_a + p_b_reversed = [1,2,3] + [5,4] = [1,2,3,5,4]. Correct.
                                         # So yes, p_a + p_b (where p_b is the reversed list if needed).
                                         # My code above `return p_a + p_b[1:]` was a mistake in thought process.
                                         # It should be `p_a + p_b`.
                                         # Wait, does p_b[0] duplicate p_a[-1]?
                                         # No, we checked `set(p_a) & set(p_b)` is empty.
                                         # So p_a[-1] != p_b[0].
                                         # So `p_a + p_b` is correct.
                                         # 
                                         # CORRECTION: 
                                         # If I use `p_a + p_b`, the edge is (p_a[-1], p_b[0]).
                                         # This is exactly what we check.
                                         # So return `p_a + p_b`.
                
                merged = None
                # 1. end1 - start2
                m = try_merge(p1, p2, False, False)
                if m: merged = m; i_rev = False; j_rev = False
                else:
                    # 2. end1 - end2 (reverse p2)
                    m = try_merge(p1, p2, False, True)
                    if m: merged = m; i_rev = False; j_rev = True
                    else:
                        # 3. start1 - start2 (reverse p1)
                        m = try_merge(p1, p2, True, False)
                        if m: merged = m; i_rev = True; j_rev = False
                        else:
                            # 4. start1 - end2 (reverse p1, reverse p2)
                            m = try_merge(p1, p2, True, True)
                            if m: merged = m; i_rev = True; j_rev = True
                
                if merged:
                    new_paths = current_paths[:]
                    # Remove i and j (careful with indices)
                    # Sort indices to remove safely
                    idxs = sorted([i, j], reverse=True)
                    new_paths.pop(idxs[0])
                    new_paths.pop(idxs[1])
                    new_paths.append(merged)
                    
                    new_cost = len(new_paths)
                    delta_cost = new_cost - current_cost # -1
                    
                    if delta_cost < 0 or rng.random() < pow(2.718, -delta_cost / T):
                        # Accept
                        current_paths = new_paths
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_paths = [p[:] for p in current_paths]
                        
                        # Update vertex sets
                        # This is O(N) to rebuild, but N is small.
                        current_path_vertex_sets = [set(p) for p in current_paths]
                        
            else:
                # SPLIT
                # Pick a path with length > 1
                splittable = [k for k, p in enumerate(current_paths) if len(p) > 1]
                if not splittable:
                    continue
                
                k = rng.choice(splittable)
                p = current_paths[k]
                # Pick a random edge to remove
                # Edge index in path: 0 to len(p)-2
                pos = rng.randint(0, len(p) - 2)
                
                p1 = p[:pos+1]
                p2 = p[pos+1:]
                
                new_paths = current_paths[:]
                new_paths.pop(k)
                new_paths.append(p1)
                new_paths.append(p2)
                
                new_cost = len(new_paths)
                delta_cost = new_cost - current_cost # +1
                
                if rng.random() < pow(2.718, -delta_cost / T):
                    # Accept
                    current_paths = new_paths
                    current_cost = new_cost
                    current_path_vertex_sets = [set(p) for p in current_paths]
        
        return best_paths

    def validate_and_fix(paths, edges):
        edge_set = set(edges)
        used_edges = set()
        valid_paths = []
        for p in paths:
            if len(p) != len(set(p)):
                # Not simple, break it
                # This shouldn't happen with SA logic, but just in case
                for i in range(len(p)-1):
                    e = tuple(sorted((p[i], p[i+1])))
                    if e in edge_set and e not in used_edges:
                        used_edges.add(e)
                        valid_paths.append([p[i], p[i+1]])
                continue
            
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set:
                    ok = False
                    break
                if e in used_edges:
                    ok = False
                    break
                used_edges.add(e)
            
            if ok:
                valid_paths.append(p)
            else:
                for i in range(len(p) - 1):
                    e = tuple(sorted((p[i], p[i+1])))
                    if e in edge_set and e not in used_edges:
                        used_edges.add(e)
                        valid_paths.append([p[i], p[i+1]])
        
        if used_edges != edge_set:
            for e in edge_set - used_edges:
                valid_paths.append(list(e))
        
        return valid_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        
        if time_left < 5:
            # Emergency fallback: greedy
            paths = get_initial_paths(g_edges)
        else:
            # Allocate time
            # 3 graphs.
            # Let's give ~40s to each if possible.
            # But if we are fast, we can give more.
            # SA converges.
            t_alloc = min(40.0, time_left / (3 - i))
            paths = solve_sa(g_edges, time_limit=t_alloc)
            paths = validate_and_fix(paths, g_edges)
            
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
