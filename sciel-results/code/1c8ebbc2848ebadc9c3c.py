import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    base_edge_set = set()
    for e in P['edges']:
        base_edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edge_set:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Generate all candidate edges (non-base, triangle-free with base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            # Check triangle-free: no common neighbor in base
            if base_adj[u] & base_adj[v]:
                continue
            candidates.append((u, v))
    
    # For each candidate edge, precompute which pairs it "covers"
    # A pair (a,b) is covered if adding the edge makes dist(a,b) <= 4
    # In the base path, dist(a,b) = |a-b|
    # After adding edge (u,v), the distance can be reduced via the shortcut
    
    # Precompute: for each candidate, which pairs (a,b) with |a-b| > 4 become <= 4
    # We need to compute the new distance after adding the edge to the base path
    # In a path, adding edge (u,v) creates a cycle. The new distance between a and b is:
    # min(|a-b|, dist_path(a,u) + 1 + dist_path(v,b), dist_path(a,v) + 1 + dist_path(u,b))
    # where dist_path is the path distance
    
    def path_dist(a, b):
        return abs(a - b)
    
    # Universe: all pairs (a,b) with a < b and |a-b| > 4 (need to be brought to <= 4)
    # Actually we need ALL pairs to be <= 4, but base already has some <= 4
    # Pairs with |a-b| <= 4 are already satisfied
    # Pairs with |a-b| > 4 need to be covered
    
    # For a set of added edges S, the graph is base + S
    # We need to check that all pairs have distance <= 4 in this graph
    # And that the graph is triangle-free
    
    # For the set cover formulation:
    # Universe = pairs (a,b) with a < b and path_dist(a,b) > 4
    # Each candidate edge e "covers" pair (a,b) if in base + {e}, dist(a,b) <= 4
    # But this is an approximation - multiple edges together can cover pairs that no single edge covers alone
    # However, for a good lower bound and initial solution, this works
    
    # Let's use a different approach: directly search for minimum edge sets
    # using branch and bound on the actual diameter constraint
    
    # First, let's find a good initial solution using greedy
    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            m = max(dist)
            if m > max_dist:
                max_dist = m
        return max_dist
    
    def is_triangle_free_with_current(adj, u, v):
        return len(adj[u] & adj[v]) == 0
    
    # Greedy to get initial upper bound
    adj = [set(s) for s in base_adj]
    greedy_added = []
    current_diam = compute_diameter(adj)
    
    while current_diam > 4:
        if time.time() > deadline - 5:
            break
        best_edge = None
        best_diam = current_diam
        best_sum = float('inf')
        
        for (u, v) in candidates:
            if not is_triangle_free_with_current(adj, u, v):
                continue
            adj[u].add(v)
            adj[v].add(u)
            d = compute_diameter(adj)
            # Sum of distances for tie-breaking
            s = 0
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    uu = q.popleft()
                    for vv in adj[uu]:
                        if dist[vv] == -1:
                            dist[vv] = dist[uu] + 1
                            q.append(vv)
                s += sum(dist)
            adj[u].remove(v)
            adj[v].remove(u)
            
            if d < best_diam or (d == best_diam and s < best_sum):
                best_diam = d
                best_sum = s
                best_edge = (u, v)
        
        if best_edge is None:
            break
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        greedy_added.append((u, v))
        current_diam = best_diam
    
    upper_bound = len(greedy_added) if current_diam <= 4 else 100
    
    # Now branch and bound to find minimum
    # State: set of added edges, current adjacency
    # We search for a set of size < upper_bound that achieves diameter <= 4
    
    best_solution = greedy_added[:] if current_diam <= 4 else None
    
    # Sort candidates by "coverage" - how many pairs they help
    # Precompute for each candidate, the pairs it covers in the base graph
    def pairs_covered_by_single_edge(u, v):
        """In base path + edge (u,v), which pairs (a,b) with |a-b|>4 become <=4?"""
        covered = set()
        for a in range(n):
            for b in range(a + 1, n):
                if abs(a - b) <= 4:
                    continue
                # New distance: min(|a-b|, |a-u|+1+|v-b|, |a-v|+1+|u-b|)
                d1 = abs(a - b)
                d2 = abs(a - u) + 1 + abs(v - b)
                d3 = abs(a - v) + 1 + abs(u - b)
                if min(d1, d2, d3) <= 4:
                    covered.add((a, b))
        return covered
    
    # For sorting: prefer edges that cover more pairs
    edge_coverage = {}
    for (u, v) in candidates:
        edge_coverage[(u, v)] = len(pairs_covered_by_single_edge(u, v))
    
    # Sort candidates by coverage descending
    sorted_candidates = sorted(candidates, key=lambda e: -edge_coverage[e])
    
    # Branch and bound
    # We'll use a recursive search with pruning
    # To keep it tractable, limit the search depth and use good ordering
    
    def check_diameter_and_triangle_free(added_edges):
        """Check if the set of added edges gives diameter <= 4 and is triangle-free."""
        adj_check = [set(s) for s in base_adj]
        for (u, v) in added_edges:
            if u in adj_check[v] or v in adj_check[u]:
                return False, False  # not triangle-free (would create triangle)
            # Actually check: adding (u,v) creates triangle if u and v share a neighbor
            # We need to check ALL pairs of added edges for triangles too
            adj_check[u].add(v)
            adj_check[v].add(u)
        
        # Check triangle-free
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj_check[i]:
                    if adj_check[i] & adj_check[j]:
                        return False, False
        
        # Check diameter
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj_check[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            m = max(dist)
            if m > max_dist:
                max_dist = m
        
        return max_dist <= 4, True
    
    # Branch and bound search
    # Use iterative deepening: try size 1, 2, 3, ... up to upper_bound - 1
    # For each size, use backtracking with pruning
    
    def can_complete(current_set, remaining_slots, adj_state):
        """Check if we can complete to a valid solution with remaining_slots more edges."""
        if time.time() > deadline - 2:
            return None
        if not remaining_slots:
            return check_diameter_and_triangle_free(current_set)
        
        # Lower bound: current diameter
        diam, tf = check_diameter_and_triangle_free(current_set)
        if not tf:
            return None
        if diam <= 4:
            return current_set[:]
        
        # Try adding edges
        for (u, v) in sorted_candidates:
            if (u, v) in current_set:
                continue
            if not is_triangle_free_with_current(adj_state, u, v):
                continue
            
            new_set = current_set + [(u, v)]
            new_adj = [set(s) for s in adj_state]
            new_adj[u].add(v)
            new_adj[v].add(u)
            
            result = can_complete(new_set, remaining_slots - 1, new_adj)
            if result is not None:
                return result
        
        return None
    
    # Try from size 1 up to upper_bound - 1
    found = None
    for target_size in range(1, min(upper_bound, 10)):  # limit to reasonable sizes
        if time.time() > deadline - 3:
            break
        result = can_complete([], target_size, [set(s) for s in base_adj])
        if result is not None:
            found = result
            break
    
    if found is not None:
        best_solution = found
    
    # Also try to improve: if we have a solution, try to remove edges
    if best_solution is not None:
        # Try removing each edge and checking if still valid
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(best_solution)):
                if time.time() > deadline - 2:
                    break
                candidate_remove = best_solution[i]
                new_solution = best_solution[:i] + best_solution[i+1:]
                diam, tf = check_diameter_and_triangle_free(new_solution)
                if tf and diam <= 4:
                    best_solution = new_solution
                    improved = True
                    break
    
    # Return result
    result = []
    for edge in (best_solution if best_solution else []):
        result.append([int(edge[0]), int(edge[1])])
    return result
