import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build graph adjacency for conflict detection
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Conflict sets: two edges conflict iff they share a vertex OR are joined by an edge
        # We store as bitmasks for speed if m <= 64, else sets
        use_mask = m <= 64
        conflict = [0] * m if use_mask else [set() for _ in range(m)]
        
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                # Check shared vertex
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    if use_mask:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
                    else:
                        conflict[i].add(j)
                        conflict[j].add(i)
                else:
                    # Check if joined by an edge
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        if use_mask:
                            conflict[i] |= (1 << j)
                            conflict[j] |= (1 << i)
                        else:
                            conflict[i].add(j)
                            conflict[j].add(i)

        # Helper to check if i and j conflict
        def is_conflict(i, j):
            if use_mask:
                return (conflict[i] >> j) & 1
            else:
                return j in conflict[i]

        # 1. Greedy DSATUR to get an initial upper bound (UB)
        # This is fast and provides a tight bound for the exact solver.
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            for _ in range(m):
                best_i, best_key = None, (-1, -1)
                for i in uncolored:
                    seen = set()
                    if use_mask:
                        mask = conflict[i]
                        # Iterate neighbors
                        # Optimization: just count unique colors among colored neighbors
                        # For DSATUR, we need saturation number (distinct colors of neighbors)
                        # and degree.
                        # Since m is small (<= 80*6/2 = 240), O(m^2) is fine.
                        for j in uncolored:
                            if j != i and is_conflict(i, j):
                                if colors[j] != -1:
                                    seen.add(colors[j])
                    else:
                        for j in conflict[i]:
                            if colors[j] != -1:
                                seen.add(colors[j])
                    deg = 0
                    if use_mask:
                        # Count uncolored neighbors
                        # mask & uncolored_mask? No, uncolored is a set.
                        # Just loop.
                        for j in uncolored:
                            if j != i and is_conflict(i, j):
                                deg += 1
                    else:
                        for j in conflict[i]:
                            if j in uncolored:
                                deg += 1
                    key = (len(seen), deg)
                    if key > best_key:
                        best_key = key
                        best_i = i
                i = best_i
                taken = set()
                if use_mask:
                    for j in uncolored:
                        if j != i and is_conflict(i, j) and colors[j] != -1:
                            taken.add(colors[j])
                else:
                    for j in conflict[i]:
                        if colors[j] != -1:
                            taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        initial_colors = greedy_dsatur()
        ub = max(initial_colors) + 1 if initial_colors else 0
        
        # 2. Exact Minimum Coloring via Max Independent Set (MaxIS)
        # Since the conflict graph is a perfect graph (complement of a strong product of graphs? 
        # Actually, the conflict graph for induced matchings is the complement of the "induced matching graph".
        # Is it perfect? 
        # The Erdos-Nešetřil conjecture bounds the strong chromatic index.
        # Regardless, we can find the chromatic number exactly by finding the maximum independent set 
        # (which corresponds to a maximum induced matching) and using the fact that for perfect graphs 
        # chi = omega. However, we don't know if the conflict graph is perfect.
        # But we can still use a branch-and-bound for coloring.
        # A standard approach: 
        #   chi(G) is the smallest k such that G can be covered by k independent sets.
        #   We can use a recursive backtracking that assigns edges to colors, ensuring each color class is independent.
        #   Pruning: if current_color > ub, stop.
        #   Ordering: process edges in a fixed order (e.g., by degree or static order).
        #   To make it faster, we can use the "DSATUR" ordering dynamically, but for exactness with bitmasks, 
        #   a simpler "branch on the most constrained uncolored edge" is good.

        # Let's use a specialized B&B for coloring:
        # State: uncolored set, current color assignment, current max color used.
        # We want to minimize max color.
        # We can do a binary search on K (number of colors) from lower_bound to ub.
        # Lower bound: clique number of conflict graph.
        
        def find_clique_number():
            # Bron-Kerbosch with pivot
            max_clique = 0
            R = set()
            P = set(range(m))
            X = set()
            
            def bronk(R, P, X):
                nonlocal max_clique
                if not P and not X:
                    if len(R) > max_clique:
                        max_clique = len(R)
                    return
                # Pivot
                if P:
                    p = next(iter(P))
                else:
                    p = next(iter(X))
                candidates = P - (set([p]) if use_mask else set(conflict[p])) # Wait, conflict[p] is neighbors. 
                # In Bron-Kerbosch, we need non-neighbors? No, we need vertices in P that are NOT adjacent to p?
                # Standard BK: choose pivot u in P ∪ X. Iterate v in P \ N(u).
                # Here conflict[i] is the set of neighbors of i in the conflict graph.
                # So N(p) = conflict[p].
                if use_mask:
                    # Convert mask to set for BK? Or use bit ops.
                    # For m <= 64, bit ops are fast.
                    # Let's stick to sets for BK for simplicity, m is small.
                    neigh_p = set()
                    mask = conflict[p]
                    for k in range(m):
                        if (mask >> k) & 1:
                            neigh_p.add(k)
                else:
                    neigh_p = conflict[p]
                
                for v in list(P):
                    if v not in neigh_p:
                        R.add(v)
                        P_new = P & (set([v]) if use_mask else set(conflict[v])) # Intersection with non-neighbors? No, with neighbors of v?
                        # Wait. In the conflict graph, we want a clique. 
                        # So new P = P ∩ N(v).
                        # N(v) is conflict[v].
                        if use_mask:
                            # P is a set. conflict[v] is a mask.
                            # We need P ∩ {x | (conflict[v] >> x) & 1}
                            P_new = set()
                            cv_mask = conflict[v]
                            for x in P:
                                if x != v and (cv_mask >> x) & 1:
                                    P_new.add(x)
                        else:
                            P_new = P & conflict[v]
                        
                        X_new = X & (set([v]) if use_mask else set(conflict[v]))
                        if use_mask:
                            cv_mask = conflict[v]
                            X_new = set()
                            for x in X:
                                if x != v and (cv_mask >> x) & 1:
                                    X_new.add(x)
                        else:
                            X_new = X & conflict[v]
                        
                        bronk(R, P_new, X_new)
                        P.remove(v)
                        X.add(v)
            
            bronk(R, P, X)
            return max_clique

        # Clique number is a lower bound on chromatic number.
        # However, BK on conflict graph might be slow if graph is dense.
        # For m <= 240, it's usually fine.
        
        # Alternative: Just use the greedy UB and try to prove it's optimal or find a better one.
        # Since the graphs are small, we can try to color with K = ub, ub-1, etc.
        
        def can_color_with_k(k):
            # Backtracking to see if we can color with k colors.
            # Order edges by degree (descending) to constrain early.
            order = sorted(range(m), key=lambda i: (len(conflict[i]) if not use_mask else bin(conflict[i]).count('1')), reverse=True)
            colors = [-1] * m
            # available_colors[i] = bitmask of colors available for edge i
            # Initially, all k colors available.
            # When we assign color c to edge i, for all neighbors j of i, remove c from available[j].
            # If available[j] becomes empty, fail.
            
            # To speed up, we can use a "most constrained variable" heuristic.
            # But for small m, simple backtracking with good ordering works.
            
            # Let's use a recursive function with memoization? No, state is too large.
            # Simple B&B.
            
            def backtrack(idx, used_colors):
                if time.time() > time_limit:
                    return None
                if idx == m:
                    return colors[:]
                
                i = order[idx]
                
                # Try each available color
                # We need to know which colors are still available for i.
                # We can compute this on the fly: 
                # A color c is available if no neighbor j of i has color c.
                
                # To make it faster, we can track for each color c, the set of edges with that color.
                # Or, simpler: just check neighbors.
                
                # Heuristic: try colors in order 0..k-1.
                # Optimization: If we haven't used color c yet, and we have used < k colors, 
                # we can restrict to only using color 0..used_colors (canonical labeling).
                # This reduces symmetry.
                
                for c in range(min(used_colors + 1, k)):
                    # Check if c is valid for i
                    valid = True
                    if use_mask:
                        mask = conflict[i]
                        # Check if any neighbor j has color c
                        # We can precompute for each color c, a bitmask of edges with color c.
                        # Let's maintain color_masks[c]
                        # But we are in a recursion, so we need to pass state or use global with undo.
                        # Let's use global arrays with undo.
                        pass
                    else:
                        for j in conflict[i]:
                            if colors[j] == c:
                                valid = False
                                break
                    
                    if valid:
                        colors[i] = c
                        new_used = max(used_colors, c + 1)
                        res = backtrack(idx + 1, new_used)
                        if res:
                            return res
                        colors[i] = -1
                return None
            
            # The above simple backtracking might be too slow without forward checking.
            # Let's implement a slightly better one with forward checking (arc consistency lite).
            # Actually, for m=240, simple B&B with symmetry breaking is often sufficient if UB is tight.
            
            # Let's try a different approach: 
            # We know the graph is likely close to perfect.
            # We can use the "greedy" result as a starting point and try to reduce colors.
            
            return None # Placeholder, will implement better below

        # Better Exact Solver: 
        # Use the fact that we have a valid coloring with 'ub' colors.
        # We try to remove color 'ub-1' (the highest one).
        # We take all edges with color 'ub-1' and try to re-color them with colors 0..ub-2.
        # This is a smaller problem.
        # If successful, we now have ub-1 colors.
        # We can repeat this process.
        
        current_colors = initial_colors[:]
        current_k = ub
        
        # Optimization: Precompute conflict neighbors as lists for faster iteration in re-coloring
        neighbors = []
        for i in range(m):
            if use_mask:
                nb = []
                mask = conflict[i]
                for j in range(m):
                    if (mask >> j) & 1:
                        nb.append(j)
                neighbors.append(nb)
            else:
                neighbors.append(list(conflict[i]))

        # Try to reduce colors one by one
        while current_k > 1 and time.time() < time_limit:
            # Find all edges with the highest color (current_k - 1)
            high_edges = [i for i in range(m) if current_colors[i] == current_k - 1]
            if not high_edges:
                break
            
            # Try to re-color these edges using colors 0..current_k-2
            # This is a subproblem: color the subgraph induced by high_edges?
            # No, they can conflict with edges of other colors.
            # So we just need to assign a color in [0, current_k-2] to each edge in high_edges
            # such that no two adjacent edges (in conflict graph) share a color.
            # Note: Edges in high_edges might conflict with each other.
            
            # Use backtracking on this small set
            # Order high_edges by degree in the conflict graph (restricted to all edges? or just high_edges?)
            # Restricting to all edges is safer.
            
            h_edges = sorted(high_edges, key=lambda i: len(neighbors[i]), reverse=True)
            
            def try_recolor(idx):
                if time.time() > time_limit:
                    return False
                if idx == len(h_edges):
                    return True
                i = h_edges[idx]
                # Find available colors in [0, current_k-2]
                used = set()
                for j in neighbors[i]:
                    if current_colors[j] != -1 and current_colors[j] < current_k - 1:
                        used.add(current_colors[j])
                    # If neighbor is in high_edges and already assigned in this recursion, check that too
                    # But we are modifying current_colors in place? 
                    # Better to use a local assignment map for high_edges.
                
                # Let's use a local dict for the new colors of high_edges
                # This is getting complicated. Let's use a simple recursive B&B with undo.
                
                # Try each color c in 0..current_k-2
                for c in range(current_k - 1):
                    valid = True
                    for j in neighbors[i]:
                        # If j is in high_edges and assigned a color in this round, check it
                        # If j is not in high_edges, check its original color
                        if j in high_edges_set:
                            if new_colors.get(j, -1) == c:
                                valid = False
                                break
                        else:
                            if current_colors[j] == c:
                                valid = False
                                break
                    
                    if valid:
                        new_colors[i] = c
                        if try_recolor(idx + 1):
                            return True
                        del new_colors[i]
                return False

            high_edges_set = set(high_edges)
            new_colors = {}
            
            # Apply symmetry breaking: 
            # The first edge in h_edges can be assigned any color.
            # Subsequent edges can be assigned colors <= max color assigned so far + 1?
            # Not exactly, because we are constrained by existing colors.
            
            if try_recolor(0):
                # Update current_colors
                for i in high_edges:
                    current_colors[i] = new_colors[i]
                current_k -= 1
            else:
                # Cannot reduce further
                break

        # 3. Final Polish: Local Search to ensure validity and maybe squeeze more
        # The re-coloring above is exact for the reduction step.
        # But we might have missed opportunities if the order of reduction was bad.
        # Let's do a quick LNS to see if we can reduce further.
        
        # Actually, the "remove highest color" strategy is quite effective.
        # Let's verify the result is valid.
        
        # Verification
        ok = True
        for c in range(current_k):
            members = [i for i in range(m) if current_colors[i] == c]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    if is_conflict(members[a], members[b]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return current_colors
        else:
            # Fallback to initial greedy if something went wrong
            return initial_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0.5:
            # Emergency greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cols = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near |= adj[x]
                taken = set()
                for f, c in cols.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                cols[e] = c
            results.append([cols[e] for e in edges])
            continue
        tl = max(1.0, remaining - 1.0)
        results.append(solve_instance(n, d, seed, tl))
    return results
