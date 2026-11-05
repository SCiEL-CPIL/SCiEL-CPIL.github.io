import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for safety

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

    def solve_graph(edges, time_budget):
        if not edges:
            return []
        
        # Build conflict graph H
        # Vertices of H are indices 0..M-1 corresponding to edges in 'edges'
        M = len(edges)
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Adjacency in G
        adj_g = {}
        for u, v in edges:
            adj_g.setdefault(u, set()).add(v)
            adj_g.setdefault(v, set()).add(u)
            
        # Build H adjacency
        # H[i] is set of indices j such that edges[i] and edges[j] are at dist <= 2 in G
        # Two edges (u,v) and (x,y) are at dist 1 if they share a vertex.
        # Dist 2 if there is an edge (v,x) or (v,y) or (u,x) or (u,y) connecting them.
        # More simply: dist <= 2 in G means there is a path of length 1 or 2 between any vertex of e1 and any vertex of e2?
        # No, the definition of strong edge coloring: edges are "adjacent" if they share a vertex or if there is an edge connecting a vertex of one to a vertex of the other.
        # So, e1=(u,v), e2=(x,y).
        # Conflict if:
        # 1. {u,v} & {x,y} != empty (share vertex)
        # 2. (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E
        
        # Precompute neighbor edges for each edge
        # For each edge e=(u,v), find all edges f=(x,y) such that e and f are in conflict.
        conflict = [set() for _ in range(M)]
        
        # Efficient construction:
        # For each edge e=(u,v):
        #   - All edges incident to u or v are at dist 1.
        #   - All edges incident to any neighbor of u or v (except u,v themselves) are at dist 2.
        
        # Let's build a list of edges incident to each vertex
        incident_edges = {v: [] for v in range(40+60+80)} # Max n is 80, but we handle per graph
        # Actually, we just need to do this per graph.
        
        # Re-initialize for current graph size
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        incident_edges = {v: set() for v in range(n)}
        for i, (u, v) in enumerate(edges):
            incident_edges[u].add(i)
            incident_edges[v].add(i)
            
        for i, (u, v) in enumerate(edges):
            # Dist 1: edges sharing u or v
            neighbors_1 = incident_edges[u] | incident_edges[v]
            # Remove self
            neighbors_1.discard(i)
            
            # Dist 2: edges connected to neighbors of u or v
            # Neighbors of u in G
            for w in adj_g.get(u, []):
                for j in incident_edges.get(w, []):
                    if j != i:
                        # Check if j is already in neighbors_1? If yes, it's dist 1.
                        # If not, it's dist 2 (unless it shares vertex with v? handled by v's neighbors)
                        # Actually, if j is incident to w, and w is neighbor of u, then dist(e, j) <= 2.
                        # We just add all such j.
                        conflict[i].add(j)
                        
            for w in adj_g.get(v, []):
                for j in incident_edges.get(w, []):
                    if j != i:
                        conflict[i].add(j)
                        
            # Ensure symmetry (though the loop above adds both ways if we iterate all i)
            # Wait, the loop adds j to conflict[i]. When we process j, it will add i to conflict[j].
            # So we don't need to explicitly symmetrize if we process all i.
            
        # Remove self if present (shouldn't be, but just in case)
        for i in range(M):
            conflict[i].discard(i)

        # DSATUR Initial Coloring
        def dsatur():
            # Colors assigned to each vertex in H (index 0..M-1)
            colors = [-1] * M
            # Saturation: number of distinct colors in neighbors
            sat = [set() for _ in range(M)]
            # Degree in H
            deg = [len(conflict[i]) for i in range(M)]
            
            # Order vertices by degree initially? Or just pick max sat/deg dynamically.
            uncolored = set(range(M))
            
            for _ in range(M):
                if not uncolored:
                    break
                # Select vertex with max saturation, then max degree
                best_v = None
                best_sat = -1
                best_deg = -1
                # Optimization: If M is large, this O(M^2) scan is slow.
                # But M is small (<= 500). 500^2 = 250k. 500 iterations * 250k = 125M ops.
                # Might be slow in Python.
                # Let's use a list and remove.
                
                # Better: Maintain a priority queue? Or just simple scan since M is small.
                # 500 * 500 = 250,000 comparisons per graph. 3 graphs. 750,000. Very fast.
                
                for v in uncolored:
                    s = len(sat[v])
                    d = deg[v]
                    if s > best_sat or (s == best_sat and d > best_deg):
                        best_sat = s
                        best_deg = d
                        best_v = v
                        
                # Assign smallest available color
                available = set(range(1, M + 1))
                for u in conflict[best_v]:
                    if colors[u] != -1:
                        available.discard(colors[u])
                        
                if not available:
                    # Should not happen with M colors
                    c = max(colors) + 1
                else:
                    c = min(available)
                    
                colors[best_v] = c
                uncolored.remove(best_v)
                
                # Update saturation of neighbors
                for u in conflict[best_v]:
                    if colors[u] == -1:
                        sat[u].add(c)
                        
            return colors, max(colors)

        colors, k = dsatur()
        
        # Color Reduction Loop
        # Try to reduce k by removing colors one by one.
        # We maintain the current colors.
        
        def try_remove_color(c_remove, current_colors, k_current):
            # Get edges with color c_remove
            to_recolor = [i for i in range(M) if current_colors[i] == c_remove]
            if not to_recolor:
                return True, current_colors, k_current - 1
            
            # Domain for these vertices: {1, ..., k_current} \ {c_remove}
            # Constraints:
            # 1. Among themselves (conflict in H)
            # 2. Against neighbors not in to_recolor (fixed colors)
            
            # We use backtracking to assign new colors to to_recolor.
            # Order: MRV (Minimum Remaining Values)
            
            # Precompute fixed neighbors' colors for each vertex in to_recolor
            # And conflicts within to_recolor
            
            # Map vertex index to position in to_recolor list for fast access?
            # Or just use the global index.
            
            # We will use a recursive solver.
            # To avoid deep recursion limits, M is small.
            
            # Sort to_recolor by number of available colors (MRV)
            # Available colors for v: {1..k_current} \ {c_remove} \ {colors[u] for u in conflict[v] if u not in to_recolor}
            
            # Note: Colors of vertices in to_recolor are unknown yet.
            
            # Let's implement a simple backtracking.
            
            # Heuristic: Sort to_recolor by degree in the subgraph induced by to_recolor + constraints from outside.
            
            def get_domain(v):
                dom = set(range(1, k_current + 1))
                dom.discard(c_remove)
                for u in conflict[v]:
                    if current_colors[u] != c_remove:
                        dom.discard(current_colors[u])
                return dom
            
            # Initial domains
            domains = {v: get_domain(v) for v in to_recolor}
            
            # If any domain is empty, fail
            for v in to_recolor:
                if not domains[v]:
                    return False, current_colors, k_current
            
            # Order variables
            # Simple: sort by len(domains[v])
            order = sorted(to_recolor, key=lambda v: len(domains[v]))
            
            # Backtracking
            # new_colors is a dict for the vertices in to_recolor
            new_colors = {}
            
            def backtrack(idx):
                if idx == len(order):
                    return True
                
                v = order[idx]
                # Try each color in domain
                # Shuffle for randomness?
                for c in domains[v]:
                    # Check if c is valid with already assigned neighbors in to_recolor
                    valid = True
                    for u in conflict[v]:
                        if u in new_colors:
                            if new_colors[u] == c:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Assign
                    new_colors[v] = c
                    
                    # Update domains of subsequent variables?
                    # For simplicity, we just check against new_colors in the loop.
                    # This is O(N) per check. N is small.
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Unassign
                    del new_colors[v]
                
                return False
            
            if backtrack(0):
                # Apply new colors
                for v, c in new_colors.items():
                    current_colors[v] = c
                return True, current_colors, k_current - 1
            else:
                return False, current_colors, k_current

        # Iterative reduction
        # To avoid getting stuck, we can restart with different DSATUR seeds if we fail to reduce below a threshold.
        # But first, let's just try to reduce as much as possible.
        
        improved = True
        while improved:
            improved = False
            # Try removing colors in random order
            colors_to_try = list(range(1, k + 1))
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Check if color c is actually used
                if not any(x == c for x in colors):
                    continue
                    
                success, new_colors, new_k = try_remove_color(c, colors, k)
                if success:
                    colors = new_colors
                    k = new_k
                    improved = True
                    # Restart the loop to try removing other colors with the new k
                    break
        
        return colors

    results = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() - start_time > time_limit - 10:
            # Not enough time, return best effort (empty or greedy?)
            # Better to return a valid coloring if we have one, or generate a quick greedy one.
            # For safety, if we run out of time, we should return a valid coloring.
            # The DSATUR coloring is valid. If we didn't finish reduction, we return the DSATUR result.
            # But we need to handle the case where we haven't even started DSATUR?
            # The loop structure above handles this.
            pass
            
        edges = instance(*spec)
        # Allocate time budget per graph
        # 110s / 3 = 36s per graph.
        # But DSATUR is fast. Reduction is the bottleneck.
        # Let's give 30s to each.
        budget = 30.0
        col_list = solve_graph(edges, budget)
        results.append(col_list)
        
    return results

# EVOLVE-BLOCK-END
