import itertools
import time

def construct():
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Build conflict graph
    # u and v are adjacent if they share a K(2,2,2)
    # A K(2,2,2) is defined by 6 points and a partition into 3 pairs.
    # There are 8 transversals for each such structure.
    # We mark all pairs of transversals as conflicting.
    
    # adj[i] is a bit mask of vertices conflicting with i
    adj = [0] * num_triples
    
    # Iterate over all sets of 6 points
    # To speed up, we can just iterate over all pairs of triples and check if they share a K(2,2,2)
    # But precomputing via structures is better.
    
    # 1. Generate all 6-point subsets
    # 2. For each, generate all 15 partitions into 3 pairs
    # 3. For each partition, generate 8 transversals
    # 4. For each pair of transversals, set bits in adj
    
    # Optimization: 
    # Instead of iterating 6-subsets, we can just iterate over all pairs of triples (u, v)
    # and check if they can be part of a K(2,2,2).
    # Condition: u and v must be subsets of some 6-point set V, and there must be a partition of V 
    # such that u and v are transversals.
    
    # This check is expensive if done naively for 313k pairs.
    # Let's use the 6-subset approach.
    # C(16,6) = 8008.
    # For each 6-subset, 15 partitions.
    # For each partition, 8 transversals.
    # Total structures: 8008 * 15 = 120120.
    # For each, 28 pairs.
    # Total operations: 120120 * 28 = 3.3M. This is fine.
    
    for points_6 in itertools.combinations(points, 6):
        # Generate all partitions of these 6 points into 3 pairs
        # There are 15 ways.
        # p0, p1, p2, p3, p4, p5
        # Pairs: (p0,p1),(p2,p3),(p4,p5)
        # We can generate them by fixing p0's partner, etc.
        
        # Helper to generate partitions
        def get_partitions(pts):
            if len(pts) == 0:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                partner = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for sub_part in get_partitions(new_rest):
                    yield [(first, partner)] + sub_part
        
        for partition in get_partitions(list(points_6)):
            # partition is a list of 3 pairs: [(x1,y1), (x2,y2), (x3,y3)]
            p1, p2, p3 = partition
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            
            # Generate 8 transversals
            # Each transversal picks one from each pair
            transversals = []
            for a in [x1, y1]:
                for b in [x2, y2]:
                    for c in [x3, y3]:
                        t = tuple(sorted((a, b, c)))
                        transversals.append(t)
                        
            # Mark conflicts between all pairs of these 8 transversals
            idxs = [triple_to_idx[t] for t in transversals]
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    u = idxs[i]
                    v = idxs[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    # Now solve Maximum Independent Set on the graph defined by adj
    # We use a branch and bound algorithm with greedy coloring.
    
    # Convert adj to a list of neighbor masks for faster access?
    # adj is already a list of ints.
    
    # Greedy coloring for bounding
    # We order vertices by degree (descending) to improve coloring?
    # Or just use natural order.
    
    # To speed up, we can use a simple recursive MIS.
    # Since the graph is likely dense, the MIS is small.
    
    best_size = 0
    best_set_mask = 0
    
    # Use a stack or recursion
    # To avoid recursion depth issues and overhead, we can use an iterative approach 
    # or just trust Python's recursion limit (set higher).
    
    import sys
    sys.setrecursionlimit(10000)
    
    def greedy_color(candidates_mask):
        # Returns a list of colors for vertices in candidates_mask
        # and the max color count
        # We want to bound the size of the MIS.
        # We color the graph induced by candidates_mask.
        # If we use k colors, the MIS size is at most k.
        
        # Extract vertices
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        if not verts:
            return 0, []
            
        colors = {}
        color_count = 0
        
        # Greedy coloring: assign smallest possible color
        # Order: maybe by degree in the induced subgraph?
        # For simplicity, use natural order.
        for v in verts:
            # Get neighbors of v in candidates
            neigh_mask = adj[v] & candidates_mask
            used_colors = set()
            nn = neigh_mask
            while nn:
                lsb = nn & -nn
                u = lsb.bit_length() - 1
                if u in colors:
                    used_colors.add(colors[u])
                nn ^= lsb
                
            # Find smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
                
        return color_count, colors

    def solve(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask
        
        if not candidates_mask:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Bound
        bound, _ = greedy_color(candidates_mask)
        if current_size + bound <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: pick vertex with max degree in candidates
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        # Sort by degree in candidates (descending)
        def deg(v):
            return bin(adj[v] & candidates_mask).count('1')
        
        verts.sort(key=deg, reverse=True)
        v = verts[0]
        
        # Branch 1: Include v
        # New candidates: candidates that are not neighbors of v
        new_candidates = candidates_mask & ~(adj[v] | (1 << v))
        solve(new_candidates, current_size + 1, current_mask | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates_excl = candidates_mask & ~(1 << v)
        solve(new_candidates_excl, current_size, current_mask)
        
    # Initial call
    # We can start with all vertices
    all_mask = (1 << num_triples) - 1
    solve(all_mask, 0, 0)
    
    # Extract best set
    best_triples = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        best_triples.append(all_triples[idx])
        m ^= lsb
        
    return [list(t) for t in best_triples]
