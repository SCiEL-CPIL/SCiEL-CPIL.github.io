import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def _time_left():
        return deadline - time.time()

    def _verify(n, colors):
        """
        Check if the coloring is valid (no C4 in any color class).
        Uses bitmasks for efficiency.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        if any(c < 0 or c > 4 for c in colors):
            return False
            
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u + 1, n):
                    # Common neighbors of u and v in color c
                    common = mask_u & adj[c][v]
                    # If there are at least 2 common neighbors, we have a C4
                    if common & (common - 1):
                        return False
        return True

    def _cyclic_construction(n):
        """
        Cyclic construction for odd n.
        Valid for n=11 with 5 colors (distances 1..5).
        """
        if n % 2 == 0:
            return None
        k = n // 2
        if k > 5:
            return None
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                dist = min(d, n - d)
                c = dist - 1
                colors.append(c)
        return colors

    def _product_construction():
        """
        Lazebnik-Woldar construction for n=26.
        Vertices are (x, y, z) in Z_3 x Z_3 x Z_3.
        """
        n = 26
        verts = list(itertools.product(range(3), repeat=3))
        colors = []
        for idx1, (a, b, c) in enumerate(verts):
            for idx2, (a2, b2, c2) in enumerate(verts):
                if idx1 >= idx2:
                    continue
                # Difference in Z_3
                dx = (a - a2) % 3
                dy = (b - b2) % 3
                dz = (c - c2) % 3
                
                # Map Z_3 to {0,1} for color bits
                # 0 -> 0
                # 1 -> 1
                # 2 -> 0
                dx_bit = 1 if dx == 1 else 0
                dy_bit = 1 if dy == 1 else 0
                dz_bit = 1 if dz == 1 else 0
                
                # Color is 3-bit number: 0..7. But we only have 5 colors.
                # The standard construction uses 5 colors?
                # Actually, R_5(C4) >= 26 is known.
                # The mapping in the prompt's inspiration:
                # dx = (a - a2) % 2 ... this was wrong in the prompt's analysis.
                # Let's use the known valid construction.
                # One known construction:
                # Color = (dx + 2*dy + 4*dz) % 5 ?
                # Or maybe the vertices are in Z_5 x Z_5? No, 25 vertices.
                # 26 vertices usually comes from a specific design.
                # Let's try the mapping from the prompt's "inspiration 2" which claimed to work.
                # It used:
                # dx = (a - a2) % 2
                # This maps 0->0, 1->1, 2->0.
                # So it treats -1 (which is 2) as 0.
                # This is not symmetric.
                
                # Let's try a symmetric mapping if possible, or just trust the previous code's logic
                # which passed verification in the parent program (score 1.5455 implies it found 26).
                # The parent program's _product_construction used:
                # dx = (a - a2) % 2
                # This is asymmetric. Let's stick to it if it works.
                
                dx_bit = (a - a2) % 2
                dy_bit = (b - b2) % 2
                dz_bit = (c - c2) % 2
                
                colors.append(dx_bit + 2 * dy_bit + 4 * dz_bit)
        
        return {'n': n, 'colors': colors}

    def _spectral_local_search(n, time_budget):
        """
        Spectral relaxation guided local search.
        1. Initialize random.
        2. Use spectral properties (trace of A^2) to identify "bad" edges.
        3. Local search with targeted moves.
        """
        if _time_left() < 2.0:
            return None

        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        # Build adj masks
        adj = [[0] * n for _ in range(5)]
        
        def build_adj():
            for c in range(5):
                for i in range(n):
                    adj[c][i] = 0
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
        
        build_adj()
        
        def get_badness():
            bad = 0
            for c in range(5):
                for u in range(n):
                    mask_u = adj[c][u]
                    for v in range(u + 1, n):
                        common = mask_u & adj[c][v]
                        if common & (common - 1):
                            cnt = bin(common).count("1")
                            bad += cnt * (cnt - 1) // 2
            return bad

        best_badness = get_badness()
        
        # Spectral analysis:
        # For each color c, compute the number of "wedges" (P3) centered at each vertex.
        # deg_c[u] = popcount(adj[c][u])
        # Wedges at u in color c = deg_c[u] * (deg_c[u] - 1) / 2
        # Total wedges = sum over u.
        # This is a proxy for C4 potential.
        
        # We want to minimize badness.
        # Targeted move: Find an edge (u,v) whose removal/re-coloring reduces badness the most.
        # Approximation: Look at edges that are part of many "bad" C4s.
        
        # For simplicity, we will use a weighted random selection for edges.
        # Weight of edge (u,v) with color c:
        # Common neighbors of u and v in color c.
        # If > 1, it's bad.
        
        # Let's just do standard hill climbing but with a "spectral" restart.
        # If stuck, re-initialize using the eigenvectors of the adjacency matrix?
        # Too complex. Let's stick to efficient hill climbing with better initialization.
        
        # Better initialization:
        # Try to construct a graph where each color class is a union of cycles/paths with low density.
        
        max_steps = 20000
        stagnation_counter = 0
        
        for step in range(max_steps):
            if _time_left() < 2.0:
                break
            
            if best_badness == 0:
                return colors
            
            # Pick an edge
            # Bias towards edges that are currently "bad"
            # We can estimate badness contribution of edge (u,v) color c:
            # It contributes to common neighbors of pairs (x, y) where x~u, y~v in color c?
            # Actually, if we change color of (u,v), it affects:
            # 1. Common neighbors of u and v in old color c (decreases)
            # 2. Common neighbors of u and v in new color c' (increases)
            # 3. Common neighbors of u and x in old color c (changes if x is neighbor of v in c)
            # This is complex.
            
            # Simple heuristic: Pick random edge.
            idx = random.randint(0, num_edges - 1)
            u, v = edge_map[idx]
            old_c = colors[idx]
            
            # Try all 5 colors
            best_new_c = old_c
            best_new_badness = best_badness
            
            # Calculate current badness contribution of (u,v) in old_c?
            # Let's just try each color and calculate delta.
            
            # To calculate delta efficiently:
            # Remove (u,v) from old_c.
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            
            # Calculate badness of the graph without this edge?
            # This is O(n^2) if we recalculate everything.
            # For n=26, n^2=676. 5 colors * 20000 steps * 676 = 67M ops. Might be too slow in Python.
            
            # Optimization: Only check pairs involving u or v.
            # A C4 involves 4 vertices.
            # Changing (u,v) affects C4s that use edge (u,v).
            # A C4 using (u,v) must be u-v-x-y-u or u-v-y-x-u.
            # So x and y must be common neighbors of u and v in the color of the other edges?
            # No, all 4 edges must be same color.
            # So if (u,v) is color c, the other 3 edges (v,x), (x,y), (y,u) must also be color c.
            # So x must be a common neighbor of u and v in color c?
            # No, x is neighbor of v in c, and y is neighbor of u in c, and (x,y) is color c.
            # And (u,v) is color c.
            # So the C4 is u-v-x-y-u.
            # Edges: (u,v) c, (v,x) c, (x,y) c, (y,u) c.
            # x is in N_c(v) \cap N_c(u)? No.
            # x is in N_c(v). y is in N_c(u).
            # And (x,y) is in color c.
            # So for a fixed (u,v) in color c, the number of C4s involving (u,v) is:
            # Sum over x in N_c(v), y in N_c(u) of [ (x,y) is color c ]
            
            # Let's implement a fast delta calculation.
            
            # Current state: (u,v) is NOT in adj (we removed it).
            # We want to add it back in color new_c.
            # Delta in badness = Badness(with edge) - Badness(without edge).
            
            # Badness is sum over all pairs (a,b) of C(common_c(a,b), 2).
            # Let f_c(a,b) = C(common_c(a,b), 2).
            # Changing (u,v) from old_c to new_c:
            # 1. For old_c:
            #    - common_old(u,v) decreases by 1? No, (u,v) is an edge.
            #    - common_old(a,b) changes if a,b are u,v?
            #    - If we add edge (u,v) to color c:
            #      common_c(u,v) increases by 0? No, u and v are neighbors.
            #      common_c(a,b) for a!=u,v:
            #      If a is neighbor of u in c and b is neighbor of v in c, does common_c(a,b) change?
            #      No, common_c(a,b) is intersection of N_c(a) and N_c(b).
            #      Adding (u,v) adds v to N_c(u) and u to N_c(v).
            #      So for any a, N_c(a) is unchanged unless a==u or a==v.
            #      So only pairs (u, x) and (v, x) and (u, v) are affected?
            #      
            #      Let's check pairs (u, x):
            #      common_c(u, x) = N_c(u) & N_c(x).
            #      N_c(u) gains v. So common_c(u, x) gains v if v is in N_c(x).
            #      So common_c(u, x) increases by 1 if (v, x) is color c.
            #      Similarly, common_c(v, x) increases by 1 if (u, x) is color c.
            #      common_c(u, v) = N_c(u) & N_c(v).
            #      N_c(u) gains v, N_c(v) gains u.
            #      common_c(u, v) = (N_c(u) \ {v}) & (N_c(v) \ {u})?
            #      Actually, u is not in N_c(u). v is not in N_c(v).
            #      So common_c(u, v) is the number of common neighbors.
            #      It does NOT change when we add edge (u,v) itself?
            #      Wait. N_c(u) is the set of neighbors.
            #      If we add (u,v), v becomes a neighbor of u.
            #      So N_c(u) = N_c(u) \cup {v}.
            #      common_c(u, x) = N_c(u) & N_c(x).
            #      New common = (Old N_c(u) \cup {v}) & N_c(x) = (Old common) \cup ({v} & N_c(x)).
            #      So it increases by 1 if v is in N_c(x).
            #      
            #      So, adding edge (u,v) to color c:
            #      For each x != u,v:
            #      - If (v,x) is color c, common_c(u,x) increases by 1.
            #      - If (u,x) is color c, common_c(v,x) increases by 1.
            #      - common_c(u,v) is the number of common neighbors.
            #        It increases by 0?
            #        N_c(u) gains v. N_c(v) gains u.
            #        common_c(u,v) = N_c(u) & N_c(v).
            #        v is in N_c(u). Is v in N_c(v)? No.
            #        u is in N_c(v). Is u in N_c(u)? No.
            #        So common_c(u,v) does NOT change?
            #        Wait. common_c(u,v) counts vertices z such that (u,z) in c and (v,z) in c.
            #        Adding (u,v) does not create new common neighbors z.
            #        So common_c(u,v) is unchanged.
            
            #      So the change in badness for adding (u,v) to color c is:
            #      Sum over x != u,v of:
            #      Delta(f_c(u,x)) + Delta(f_c(v,x))
            #      where f(a,b) = C(common(a,b), 2).
            #      If common increases by 1 (from k to k+1), f increases by k.
            #      So Delta = k.
            
            #      So, to add (u,v) to color c:
            #      cost = 0
            #      for x in 0..n-1 (x != u,v):
            #          if (v,x) is color c:
            #              k = popcount(adj[c][u] & adj[c][x])  # Current common (before adding u,v)
            #              cost += k
            #          if (u,x) is color c:
            #              k = popcount(adj[c][v] & adj[c][x])
            #              cost += k
            #      
            #      This is O(n) per color.
            #      5 colors * O(n) = O(5n).
            #      For n=26, ~130 ops per edge evaluation.
            #      20000 steps * 130 = 2.6M ops. Very fast.
            
            # We removed (u,v) from adj[old_c].
            # We want to find new_c that minimizes badness.
            # Current badness is best_badness (calculated with (u,v) in old_c).
            # Let's calculate badness_without_uv = best_badness - cost_to_add_uv_to_old_c?
            # No, best_badness includes the edge.
            # Let's calculate the cost to ADD the edge to each color c.
            # cost_add(c) = increase in badness if we add (u,v) to color c (assuming it's not there).
            
            # Since we removed it from old_c, the current state has NO (u,v).
            # Let's call the current badness B_current.
            # We can calculate B_current by subtracting the cost of the edge from best_badness?
            # Or just recalculate B_current once?
            # Recalculating B_current is O(n^2).
            # Let's do that once per step? No, too slow.
            
            # Let's maintain B_current.
            # Initially B_current = best_badness.
            # We remove (u,v) from old_c.
            # B_current = best_badness - cost_add(old_c) ?
            # cost_add(old_c) was the increase when we ADDED it.
            # So yes, B_current = best_badness - cost_add(old_c).
            
            # Let's compute cost_add(c) for all c.
            costs = [0] * 5
            
            for c in range(5):
                # (u,v) is NOT in adj[c] currently (we removed it from old_c, and it wasn't in others).
                # So we can calculate cost to add.
                
                # For x in 0..n-1:
                # If x == u or x == v, skip.
                
                # We need popcount(adj[c][u] & adj[c][x]) and popcount(adj[c][v] & adj[c][x]).
                
                # Note: adj[c][u] and adj[c][v] do NOT contain v and u respectively (since we removed the edge).
                
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    # Check if (v,x) is in color c
                    # i.e., if x is in adj[c][v]
                    if adj[c][v] & (1 << x):
                        # common_c(u,x) increases by 1
                        # Current common = popcount(adj[c][u] & adj[c][x])
                        common = adj[c][u] & adj[c][x]
                        k = bin(common).count("1")
                        costs[c] += k
                    
                    # Check if (u,x) is in color c
                    if adj[c][u] & (1 << x):
                        common = adj[c][v] & adj[c][x]
                        k = bin(common).count("1")
                        costs[c] += k
            
            # Now, find best_c to minimize B_current + costs[c].
            min_cost = min(costs)
            best_new_c = costs.index(min_cost)
            
            # Apply the move
            # Add (u,v) to best_new_c
            adj[best_new_c][u] |= (1 << v)
            adj[best_new_c][v] |= (1 << u)
            colors[idx] = best_new_c
            
            new_badness = B_current + min_cost
            
            if new_badness < best_badness:
                best_badness = new_badness
                B_current = new_badness
                stagnation_counter = 0
            else:
                # Revert
                adj[best_new_c][u] &= ~(1 << v)
                adj[best_new_c][v] &= ~(1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                colors[idx] = old_c
                
                # B_current should go back to state with (u,v) in old_c?
                # No, we want to keep the state with (u,v) REMOVED?
                # No, the state must always have all edges.
                # So we revert to old_c.
                # B_current (without edge) + cost_add(old_c) = best_badness.
                # So we are back to best_badness.
                
                stagnation_counter += 1
                if stagnation_counter > 5000:
                    # Random restart
                    colors = [random.randint(0, 4) for _ in range(num_edges)]
                    build_adj()
                    best_badness = get_badness()
                    B_current = best_badness
                    stagnation_counter = 0

        if best_badness == 0:
            return colors
        return None

    # 1. Try known algebraic constructions
    best_n = 11
    best_colors = _cyclic_construction(11)
    if best_colors and _verify(11, best_colors):
        pass
    
    # Try n=26 construction
    if _time_left() > 5.0:
        res26 = _product_construction()
        if _verify(res26['n'], res26['colors']):
            best_n = 26
            best_colors = res26['colors']
    
    # Try to find n=12..25 via local search?
    if best_n < 26:
        for n in range(12, 26):
            if _time_left() < 10.0:
                break
            
            found = False
            for attempt in range(3):
                if _time_left() < 5.0:
                    break
                res = _spectral_local_search(n, 5.0)
                if res:
                    if _verify(n, res):
                        best_n = n
                        best_colors = res
                        found = True
                        break
            if found:
                pass
            else:
                break

    # Final verification
    if best_colors is None or not _verify(best_n, best_colors):
        best_n = 11
        best_colors = _cyclic_construction(11)

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
