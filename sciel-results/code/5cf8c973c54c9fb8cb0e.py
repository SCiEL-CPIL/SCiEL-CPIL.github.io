import time
import random
import sys
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

    def build_conflict_graph(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            conflicts[i].discard(i)
        return [list(c) for c in conflicts]

    def hopcroft_karp(adj, n_left, n_right):
        # adj: list of lists, adj[i] contains neighbors in right side
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = []
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u

    def max_induced_matching_heuristic(edges, adj, conflicts, remaining, time_budget=5.0):
        """
        Find a large induced matching among 'remaining' edges.
        Uses a greedy + local search (2-swap) approach to maximize size.
        """
        if not remaining:
            return []
        
        # Initial greedy: pick edges with lowest degree in conflict graph
        # to allow more neighbors to be picked later? Or highest?
        # Actually, for matching, we want to pick edges that don't conflict.
        # Let's start with a greedy set.
        # Sort by degree in conflict graph (within remaining)
        
        rem_set = set(remaining)
        
        # Calculate degrees within remaining
        deg = {i: 0 for i in remaining}
        for i in remaining:
            for j in conflicts[i]:
                if j in rem_set:
                    deg[i] += 1
        
        # Greedy: pick edge with lowest degree, remove it and its neighbors
        candidates = sorted(remaining, key=lambda x: deg[x])
        
        selected = []
        available = set(remaining)
        
        # Simple greedy
        for i in candidates:
            if i not in available:
                continue
            selected.append(i)
            # Remove i and its neighbors from available
            available.remove(i)
            for j in conflicts[i]:
                if j in available:
                    available.remove(j)
        
        # Local search to improve size
        # Move: replace an edge e in selected with an edge f in available
        # that conflicts with e but not with any other edge in selected (except e)
        # This increases size by 1 if f doesn't conflict with selected \ {e}
        
        t_start = time.time()
        t_limit = t_start + time_budget
        
        # Build set for fast lookup
        selected_set = set(selected)
        
        # Precompute conflicts for available edges?
        # For each e in selected, and each f in available:
        # If f conflicts with e, check if f conflicts with any other e' in selected.
        # If not, swap e -> f is valid and increases size by 1 (if we just replace)
        # Wait, replacing e with f keeps size same. We need to ADD f.
        # But f must not conflict with any other selected edge.
        # So if f conflicts with e, and e is removed, f can be added IF f doesn't conflict with others.
        # This effectively moves the "blockage".
        
        # Better move: 2-swap. Remove e1, e2. Add f1, f2.
        # This is complex. Let's stick to single swap first.
        
        # Actually, a standard heuristic for Max Independent Set (which this is) is:
        # Try to add an edge from available if it doesn't conflict with selected.
        # If stuck, try to remove one selected edge to free up space for multiple available edges.
        
        # Let's try a simple hill climbing on size.
        # State: selected list.
        # Move: 
        # 1. If there is an edge in available that doesn't conflict with selected, add it.
        # 2. If not, try removing one edge e from selected, and see if we can add >1 edges from the newly available set.
        
        # Re-implementation of local search for size maximization
        current_selected = selected[:]
        current_size = len(current_selected)
        
        # Precompute conflict sets for speed
        # conflicts[i] is list of all conflicting edges
        
        # Map edge index to position in current_selected? No, just use set.
        
        while time.time() < t_limit:
            improved = False
            
            # Phase 1: Add any non-conflicting edge from available
            # available = all edges not in current_selected and not conflicting with current_selected?
            # No, "available" in the context of independent set usually means "not in set".
            # But we must ensure validity.
            
            # Let's maintain a set of "blocked" edges (conflicting with selected).
            # This is expensive to update.
            
            # Alternative: Just check all edges not in selected.
            selected_set = set(current_selected)
            
            # Find all edges not in selected
            # For each such edge, check if it conflicts with any in selected.
            # If not, add it.
            
            # To speed up, we can iterate through remaining edges.
            # If m is small (< 200), this is fine.
            
            new_selected = current_selected[:]
            new_set = set(new_selected)
            
            for e in remaining:
                if e in new_set:
                    continue
                # Check conflict
                conflict = False
                for f in conflicts[e]:
                    if f in new_set:
                        conflict = True
                        break
                if not conflict:
                    new_selected.append(e)
                    new_set.add(e)
                    improved = True
            
            if len(new_selected) > current_size:
                current_selected = new_selected
                current_size = len(current_selected)
                improved = True
            
            if not improved:
                # Phase 2: Try 1-swap
                # Remove one edge e from selected.
                # Then see if we can add any edges.
                # If we can add > 0 edges, and the net gain is positive, do it.
                # Actually, if we remove e, we might be able to add f1, f2...
                
                best_swap_gain = 0
                best_swap_new_set = None
                
                # Sample some edges to remove to save time
                # If m is small, try all.
                for e in current_selected:
                    # Tentatively remove e
                    temp_set = set(current_selected)
                    temp_set.remove(e)
                    
                    # Try to add edges
                    added = []
                    # Iterate through remaining edges not in temp_set
                    for f in remaining:
                        if f in temp_set:
                            continue
                        # Check conflict with temp_set
                        conflict = False
                        for g in conflicts[f]:
                            if g in temp_set:
                                conflict = True
                                break
                        if not conflict:
                            added.append(f)
                            temp_set.add(f)
                    
                    gain = len(added) - 1 # Removed 1, added len(added)
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_new_set = temp_set
                        best_swap_added = added # Not strictly needed if we have set
                
                if best_swap_gain > 0:
                    current_selected = list(best_swap_new_set)
                    current_size = len(current_selected)
                    improved = True
            
            if not improved:
                break
            
        return current_selected

    def color_graph_max_matching(edges, adj, time_budget=45.0):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflict_graph(edges, adj)
        
        colors = [-1] * m
        remaining = list(range(m))
        color_idx = 0
        
        t_start = time.time()
        
        while remaining:
            if time.time() > t_start + time_budget:
                break
            
            # Find max induced matching in remaining
            # Use heuristic
            selected = max_induced_matching_heuristic(edges, adj, conflicts, remaining, time_budget=2.0)
            
            if not selected:
                # Should not happen if remaining is not empty
                # Fallback: pick first edge
                selected = [remaining[0]]
            
            # Assign color
            for i in selected:
                colors[i] = color_idx
            
            # Remove selected and their neighbors from remaining
            remove_set = set(selected)
            for i in selected:
                for j in conflicts[i]:
                    remove_set.add(j)
            
            remaining = [i for i in remaining if i not in remove_set]
            
            color_idx += 1
        
        # If we ran out of time, color the rest greedily
        if remaining:
            # Simple greedy for the rest
            # Order by degree
            rem_set = set(remaining)
            deg = {i: 0 for i in remaining}
            for i in remaining:
                for j in conflicts[i]:
                    if j in rem_set:
                        deg[i] += 1
            
            candidates = sorted(remaining, key=lambda x: deg[x], reverse=True)
            
            for i in candidates:
                if colors[i] != -1:
                    continue
                # Find smallest available color
                used = set()
                for j in conflicts[i]:
                    if colors[j] != -1:
                        used.add(colors[j])
                c = 0
                while c in used:
                    c += 1
                colors[i] = c
            
            # Update color_idx if needed (though we just need the array)
        
        # Compact colors to 0..k-1
        if not colors:
            return colors, 0
            
        max_c = max(colors) + 1
        # Check if any color is missing?
        # The heuristic assigns sequential colors.
        # Greedy fallback might use higher colors but not necessarily fill gaps?
        # Actually, greedy uses smallest available, so it might fill gaps if we were careful,
        # but since we are doing it in a batch, it's safer to remap.
        
        # Remap
        mapping = {}
        new_colors = [0] * m
        next_c = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_c
                next_c += 1
            new_colors[colors[i]] = mapping[colors[i]] # Bug: should be new_colors[i]
            
        # Correct remap
        new_colors = [0] * m
        mapping = {}
        next_c = 0
        for i in range(m):
            c = colors[i]
            if c not in mapping:
                mapping[c] = next_c
                next_c += 1
            new_colors[i] = mapping[c]
            
        return new_colors, next_c

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    best_results = [None] * len(specs)
    
    # Allocate time per graph
    time_per_graph = 45.0
    
    for idx, spec in enumerate(specs):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        t_start = time.time()
        t_limit = t_start + time_per_graph
        
        colors, num_colors = color_graph_max_matching(edges, adj, time_budget=time_per_graph)
        
        # Verify validity
        # Build conflicts again for verification (or reuse? We didn't return it)
        # Let's just trust the algorithm for now, or do a quick check.
        # Quick check:
        m = len(edges)
        if m > 0:
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            valid = True
            for i in range(m):
                for j in conflicts[i]: # Wait, conflicts is local to color_graph...
                    pass
            
            # We need to re-build or return conflicts.
            # For safety, let's re-build and check.
            conflicts_check = build_conflict_graph(edges, adj)
            for i in range(m):
                ci = colors[i]
                for j in conflicts_check[i]:
                    if j > i and colors[j] == ci:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback to greedy if invalid
                order = list(range(m))
                order.sort(key=lambda i: len(conflicts_check[i]), reverse=True)
                colors = [-1] * m
                for i in order:
                    forbidden = set()
                    for j in conflicts_check[i]:
                        if colors[j] != -1:
                            forbidden.add(colors[j])
                    c = 0
                    while c in forbidden:
                        c += 1
                    colors[i] = c
                # Remap
                mapping = {}
                new_colors = [0] * m
                next_c = 0
                for i in range(m):
                    c = colors[i]
                    if c not in mapping:
                        mapping[c] = next_c
                        next_c += 1
                    new_colors[i] = mapping[c]
                colors = new_colors
        
        best_results[idx] = colors
        
    return best_results

