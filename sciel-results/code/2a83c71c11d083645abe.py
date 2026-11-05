import time
import random
import math

def construct():
    start_time = time.time()
    limit = 150.0
    margin = 5.0
    
    # We will try to find a valid coloring for n=26, then 25, etc.
    # We use Simulated Annealing.
    
    def get_edges(n):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def count_c4s(n, colors, edges):
        # colors is a list of length M, values 0..4
        # Build adjacency lists for each color
        adj = [set() for _ in range(5)]
        for (u, v), c in zip(edges, colors):
            adj[c].add((u, v))
            adj[c].add((v, u))
            
        # Count C4s
        # A C4 is u-v-w-x-u.
        # It is determined by two disjoint edges? No.
        # It is determined by a pair of vertices (u, v) and two common neighbors (w, x).
        # Number of C4s = sum over all colors c of (sum over all pairs (u, v) of C(common_neighbors_c(u, v), 2))
        # Note: Each C4 is counted 4 times?
        # Let's verify.
        # C4: 0-1-2-3-0.
        # Pairs with 2 common neighbors:
        # (0, 2) have common neighbors 1, 3.
        # (1, 3) have common neighbors 0, 2.
        # So each C4 contributes 1 to the count for pair (0,2) and 1 for pair (1,3).
        # Total sum = 2 * (# of C4s).
        # So # C4s = 0.5 * sum.
        
        total = 0
        for c in range(5):
            # For each color c, find all pairs (u, v) and their common neighbors
            # To speed up, we can just iterate all pairs (u, v) in the graph?
            # No, we only care about pairs that are NOT connected by color c?
            # Actually, common neighbors are vertices w such that (u, w) is color c and (v, w) is color c.
            # We can compute common neighbors for all pairs (u, v) in the graph.
            # There are n*(n-1)/2 pairs.
            # For each pair, intersect neighbor sets.
            # This is O(n^3) roughly, which is fine for n=26.
            
            # Optimization: Precompute neighbor sets for this color
            neighbors = [set() for _ in range(n)]
            for (u, v) in edges:
                if colors[edges.index((u,v))] == c: # This is slow. 
                    pass
            
            # Better: Build neighbors once per color from the current state
            # But we are updating incrementally. 
            # For the initial count and for checking validity, we can do a full recount if needed, 
            # but for SA we use incremental.
            pass
            
        return 0 # Placeholder

    # Let's implement a direct SA loop with incremental updates.
    
    def solve(n):
        edges = get_edges(n)
        M = len(edges)
        # Initialize with random colors
        colors = [random.randint(0, 4) for _ in range(M)]
        
        # Precompute edge index for fast access
        edge_idx = {}
        for idx, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx
            
        # Build initial neighbor sets for each color
        # neighbors[c][u] = set of v such that (u, v) is color c
        neighbors = [ [set() for _ in range(n)] for _ in range(5) ]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            neighbors[c][u].add(v)
            neighbors[c][v].add(u)
            
        # Calculate initial C4 count
        def calc_c4_count():
            total = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        # Count common neighbors of u and v in color c
                        common = neighbors[c][u].intersection(neighbors[c][v])
                        k = len(common)
                        if k >= 2:
                            total += k * (k - 1) // 2
            return total // 2 # Each C4 counted twice (once for each diagonal pair)
            
        current_cost = calc_c4_count()
        
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > limit - margin:
                break
                
            # Pick a random edge
            idx = random.randint(0, M - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            
            # Try a new color different from old_c
            new_c = random.choice([c for c in range(5) if c != old_c])
            
            # Calculate delta cost
            # Remove (u, v) from old_c
            # The number of C4s involving (u, v) in old_c is the number of common neighbors of u and v in old_c.
            # Let k_old = len(neighbors[old_c][u].intersection(neighbors[old_c][v]))
            # Removing (u, v) removes all C4s that use (u, v).
            # Wait, a C4 is a cycle of length 4. 
            # If we remove edge (u, v), we break all C4s containing (u, v).
            # How many C4s contain (u, v)?
            # A C4 containing (u, v) must have the form u-v-x-y-u.
            # x and y must be common neighbors of u and v in the same color?
            # No, the edges (v, x), (x, y), (y, u) must also be in the same color?
            # YES, monochromatic C4.
            # So x and y must be common neighbors of u and v in color old_c.
            # AND the edge (x, y) must also be in color old_c.
            # My previous formula `k*(k-1)/2` counts pairs of common neighbors.
            # It assumes that for ANY pair of common neighbors x, y, the edge (x, y) is in the same color.
            # THIS IS FALSE.
            # The edge (x, y) has some color. It is only part of a monochromatic C4 if it is ALSO old_c.
            
            # So the incremental update is complex.
            # We need to count:
            # 1. C4s broken: Pairs (x, y) common neighbors of u, v in old_c, such that (x, y) is old_c.
            # 2. C4s created: Pairs (x, y) common neighbors of u, v in new_c, such that (x, y) is new_c.
            
            # This requires checking the color of (x, y).
            # This is O(k) per move, where k is number of common neighbors.
            # k is small (degree is ~ n/5 = 5). So this is fast.
            
            def get_c4s_involving_edge(u, v, c):
                common = neighbors[c][u].intersection(neighbors[c][v])
                count = 0
                for x in common:
                    for y in common:
                        if x < y:
                            # Check if edge (x, y) is color c
                            if colors[edge_idx[(x, y)]] == c:
                                count += 1
                return count

            cost_removed = get_c4s_involving_edge(u, v, old_c)
            
            # Temporarily update neighbors to see new cost? 
            # Or just calculate potential cost_added based on current neighbors (which still have u-v in old_c, but we are moving it).
            # The common neighbors of u and v in new_c are those x such that (u, x) is new_c and (v, x) is new_c.
            # Note: (u, v) is NOT in new_c yet.
            common_new = neighbors[new_c][u].intersection(neighbors[new_c][v])
            cost_added = 0
            for x in common_new:
                for y in common_new:
                    if x < y:
                        if colors[edge_idx[(x, y)]] == new_c:
                            cost_added += 1
            
            delta = cost_added - cost_removed
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept move
                # Update neighbors
                neighbors[old_c][u].remove(v)
                neighbors[old_c][v].remove(u)
                neighbors[new_c][u].add(v)
                neighbors[new_c][v].add(u)
                colors[idx] = new_c
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors
            T *= cooling
            
        return best_colors if best_cost == 0 else None

    # Try n=26
    for n in [26, 25, 24, 23]:
        # Run multiple restarts if time permits
        best_sol = None
        while time.time() - start_time < limit - margin:
            sol = solve(n)
            if sol is not None:
                best_sol = sol
                break # Found a valid one, return it
        
        if best_sol is not None:
            return {'n': n, 'colors': best_sol}
            
    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
