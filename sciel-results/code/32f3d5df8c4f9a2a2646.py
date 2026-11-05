"""Strong edge colourings via DSATUR Binary Search on Conflict Graph."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate random graph with max degree d
        # The seed program uses a specific generation method, we must replicate it exactly.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        """
        Build a conflict graph where nodes are edge indices.
        Two nodes are connected if the corresponding edges in the original graph
        share a vertex or are joined by an edge (i.e., they cannot have the same color).
        """
        m = len(edges)
        if m == 0:
            return []
        
        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        # Build vertex adjacency for original graph
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        # Build adjacency list for conflict graph
        conflict_adj = [set() for _ in range(m)]
        
        # 1. Edges sharing a vertex conflict
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                e_i = inc_edges[i]
                for j in range(i + 1, len(inc_edges)):
                    e_j = inc_edges[j]
                    conflict_adj[e_i].add(e_j)
                    conflict_adj[e_j].add(e_i)
        
        # 2. Edges joined by an edge conflict
        # For edge (u, v), any edge incident to a neighbor of u (other than v) 
        # or a neighbor of v (other than u) conflicts.
        for i, (u, v) in enumerate(edges):
            # Neighbors of u (excluding v)
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                # Any edge incident to nu conflicts with edge i
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
            # Neighbors of v (excluding u)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                        
        # Convert sets to lists for faster iteration if needed, but sets are good for membership
        return conflict_adj

    def dsatur_color(conflict_adj, k, max_time):
        """
        Attempt to color the graph with k colors using DSATUR heuristic.
        Returns a list of colors if successful, None if failed or timed out.
        """
        m = len(conflict_adj)
        if m == 0:
            return []
        
        # Convert conflict_adj to sorted lists for consistent ordering in tie-breaking
        adj_lists = [sorted(lst) for lst in conflict_adj]
        degrees = [len(lst) for lst in adj_lists]
        
        colors = [-1] * m
        # To speed up, we can pre-compute some things, but DSATUR is dynamic.
        
        # Heuristic: Start with the vertex with highest degree
        # We will use a recursive backtracking search.
        
        # Optimization: Use a set of available colors for each vertex?
        # Or just check neighbors.
        
        # To avoid deep recursion limits, we might need to be careful, 
        # but m is small (~100).
        
        start_time = time.time()
        
        # Order of vertices to try? DSATUR dynamically selects the next vertex.
        # We maintain a list of uncolored vertices.
        
        # For efficiency, we can pre-calculate the initial saturation degrees (all 0)
        # and just use degree as tie-breaker initially.
        
        # Let's implement a simple recursive solver with forward checking.
        
        def get_saturation(v, colors):
            sat = set()
            for nb in adj_lists[v]:
                if colors[nb] != -1:
                    sat.add(colors[nb])
            return len(sat)
        
        def is_valid(v, c, colors):
            for nb in adj_lists[v]:
                if colors[nb] == c:
                    return False
            return True

        # To make it faster, we can use a bitset for colors if k is small, 
        # but k is up to ~45. Python integers can act as bitsets.
        # color_mask[v] is a bitmask of colors used by neighbors of v.
        
        color_masks = [0] * m
        
        def update_masks(v, c, add):
            mask_val = 1 << c
            for nb in adj_lists[v]:
                if add:
                    color_masks[nb] |= mask_val
                else:
                    color_masks[nb] &= ~mask_val

        def backtrack(uncolored):
            if time.time() - start_time > max_time:
                return False
            
            if not uncolored:
                return True
                
            # Select vertex with max saturation, then max degree
            # Saturation is count of set bits in color_masks[v]
            # We can approximate or calculate. 
            # Calculating popcount for ~100 vertices is fast.
            
            best_v = -1
            best_sat = -1
            best_deg = -1
            
            # Optimization: Instead of checking all uncolored every time, 
            # we could maintain a priority queue, but m is small.
            for v in uncolored:
                sat = bin(color_masks[v]).count('1')
                deg = degrees[v]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
            
            v = best_v
            uncolored.remove(v)
            
            # Try colors 0 to k-1
            # Only try colors not in color_masks[v]
            available_mask = ((1 << k) - 1) & ~color_masks[v]
            
            if available_mask == 0:
                return False
            
            # Iterate through available colors
            # We can iterate bits or just 0..k-1
            # 0..k-1 is simpler and k is small
            for c in range(k):
                if available_mask & (1 << c):
                    colors[v] = c
                    update_masks(v, c, True)
                    
                    if backtrack(uncolored):
                        return True
                    
                    colors[v] = -1
                    update_masks(v, c, False)
            
            uncolored.append(v)
            return False

        uncolored = list(range(m))
        # Initial ordering: sort by degree descending to help DSATUR?
        # DSATUR picks dynamically, but initial order might matter for tie-breaking in first steps.
        # Let's just start with all uncolored.
        
        if backtrack(uncolored):
            return colors
        else:
            return None

    def greedy_color(conflict_adj, rng):
        """Fallback greedy coloring."""
        m = len(conflict_adj)
        if m == 0:
            return []
        
        # Order by degree
        order = sorted(range(m), key=lambda i: -len(conflict_adj[i]))
        colours = [0] * m
        for idx in order:
            blocked = set()
            for nb in conflict_adj[idx]:
                if colours[nb] != -1:
                    blocked.add(colours[nb])
            c = 0
            while c in blocked:
                c += 1
            colours[idx] = c
        return colours

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        # Lower bound: max degree + 1
        max_deg = max(len(adj) for adj in conflict_adj) if m > 0 else 0
        lower_bound = max_deg + 1
        
        # Upper bound: Greedy coloring result
        rng = random.Random(seed)
        greedy_sol = greedy_color(conflict_adj, rng)
        upper_bound = max(greedy_sol) + 1 if m > 0 else 0
        
        # Binary search for minimum K
        lo = lower_bound
        hi = upper_bound
        
        best_colours = greedy_sol
        best_k = upper_bound
        
        # If time is tight, just return greedy
        if time_limit < 1.0:
            return greedy_sol
            
        # Allocate time for binary search steps
        # Each step calls DSATUR. DSATUR can be exponential in worst case, 
        # but usually fast for these sizes.
        
        # We will try to find the exact minimum.
        # If DSATUR fails (times out), we assume K is too small and increase K.
        
        while lo <= hi:
            mid = (lo + hi) // 2
            
            # Check if mid is feasible
            # Give DSATUR a portion of the time
            dsatur_time = max(0.1, time_limit / (hi - lo + 1))
            
            sol = dsatur_color(conflict_adj, mid, dsatur_time)
            
            if sol is not None:
                best_colours = sol
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1
            
            if time.time() > start_time + time_limit:
                break
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Allocate time per instance
        # Total 140s. 3 instances. ~45s each.
        inst_start = time.time()
        inst_limit = inst_start + 45.0
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        # Check global deadline
        if time.time() > deadline:
            # If we run out of time, we just return what we have.
            pass

    return results
# EVOLVE-BLOCK-END
