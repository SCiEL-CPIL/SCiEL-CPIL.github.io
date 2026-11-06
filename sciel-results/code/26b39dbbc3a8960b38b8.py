"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }
    
    def build_adj(n, edges):
        """Build adjacency bitmasks from edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_c5(adj, n):
        """Check if graph contains a 5-cycle using bitsets. Returns True if found."""
        # A 5-cycle v0-v1-v2-v3-v4-v0 exists iff there exist distinct vertices
        # with those edges. We fix edge (v0, v1) and look for path v1-v2-v3-v4-v0
        # of length 4 not using edge (v0,v1).
        # Equivalent: for each edge (u,v), check if there's a path of length 4 from u to v.
        # Path u-a-b-c-v means: a in N(u)\{v}, b in N(a)\{u,v}, c in N(b)\{u,a,v}, c in N(v).
        # 
        # More efficient: for each pair (u,v) that is an edge,
        # compute all vertices reachable from u in exactly 3 steps avoiding v,
        # and check if any of those are in N(v).
        # 
        # Even better for small n: iterate over edges, and for each edge (u,v),
        # compute N2 = (N(u) & ~bit(v))  -- vertices adjacent to u except v
        # N3 = union of N(x) for x in N2, excluding u, v
        # Then check if N3 & N(v) & ~(bit(u)) is non-empty.
        
        for u in range(n):
            nu = adj[u]
            # For each neighbor v of u
            v_mask = nu
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                
                # Path: u - a - b - c - v
                # a in N(u) \ {v}
                a_mask = nu & ~(1 << v)
                if a_mask == 0:
                    continue
                
                # b in N(a) \ {u, v} for some a in a_mask
                # c in N(b) \ {u, a, v} and c in N(v)
                # 
                # Let's compute: for each a in a_mask, get N(a), intersect with N(v), 
                # exclude u and v, that gives candidate c values (but we need b between a and c)
                # Actually: a-b-c-v means b is common neighbor of a and... no.
                # a~b and b~c and c~v. So b is in N(a) ∩ N(c) for some c in N(v).
                # Equivalently: for some c in N(v)\{u,v}, b in N(a) ∩ N(c)\{u,v}.
                # 
                # Let me think differently. We need:
                # a ~ b, b ~ c, c ~ v, a ~ u, and a,b,c distinct from u,v and each other.
                # 
                # For fixed u, v:
                #   a_mask = N(u) \ {v}
                #   For each a in a_mask:
                #     b_mask = N(a) \ {u, v}
                #     For each b in b_mask:
                #       c_mask = N(b) \ {u, a, v}
                #       if c_mask & N(v): found C5
                # 
                # This is still potentially expensive. Let's use a different formulation.
                # 
                # Alternative: C5 exists iff there exist 3 vertices x, y, z (not all in same K4)
                # such that... this is getting complicated.
                # 
                # Let me just do the direct search but with early termination and bitsets.
                
                a_mask = nu & ~(1 << v)
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    b_mask = adj[a] & ~(1 << u) & ~(1 << v)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit
                        
                        c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                        if c_mask & adj[v]:
                            return True
        return False
    
    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9 using bitset backtracking."""
        # An independent set: no two vertices are adjacent.
        # We'll use a recursive approach with bitsets.
        
        def max_independent_size(adj, n, target):
            """Return True if there's an independent set of size >= target."""
            # Greedy upper bound: use Bron-Kerbosch style or simple recursion
            # State: set of available vertices (as bitmask), current set size
            # We want to find if we can pick 'target' non-adjacent vertices from available.
            
            def can_find(available, count):
                if count == 0:
                    return True
                if available == 0:
                    return False
                # Upper bound: if popcount(available) < count, impossible
                if bin(available).count('1') < count:
                    return False
                
                # Pick a vertex from available
                # Heuristic: pick vertex with minimum degree within available
                # (i.e., it excludes the fewest other candidates)
                best_v = -1
                best_excluded = 0
                avail = available
                while avail:
                    v_bit = avail & (-avail)
                    v = v_bit.bit_length() - 1
                    avail ^= v_bit
                    # How many vertices in available are NOT adjacent to v (excluding v itself)?
                    non_adj_in_avail = available & ~adj[v] & ~(1 << v)
                    # We want to pick v that maximizes non_adj_in_avail (keeps most options)
                    # Or minimizes adj_in_avail (excludes fewest)
                    adj_in_avail = available & adj[v]
                    excluded = bin(adj_in_avail).count('1')
                    if best_v == -1 or excluded < best_excluded:
                        best_excluded = excluded
                        best_v = v
                    if excluded == 0:
                        break
                
                if best_v == -1:
                    return False
                
                v = best_v
                v_bit = 1 << v
                
                # Option 1: include v -> remove v and all its neighbors from available
                new_avail = available & ~adj[v] & ~v_bit
                if can_find(new_avail, count - 1):
                    return True
                
                # Option 2: exclude v
                if can_find(available & ~v_bit, count):
                    return True
                
                return False
            
            return can_find((1 << n) - 1, target)
        
        return max_independent_size(adj, n, 9)
    
    def is_valid(n, adj):
        """Check if graph is C5-free and has no independent set of 9."""
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True
    
    def adj_to_edges(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            j = i + 1
            mask = adj[i]
            while j < n:
                if mask & (1 << j):
                    edges.append([i, j])
                j += 1
        return edges
    
    best_result = known_32
    
    # Strategy 1: Try to extend 8xK4 to n=33,34,35 by adding vertices
    # and adjusting edges to maintain C5-freeness and alpha < 9
    
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Start with 8xK4 and add (target_n - 32) extra vertices
        extra = target_n - 32
        
        # For each attempt, start with 8xK4 + isolated extra vertices
        # Then add edges to the extra vertices to break independent sets
        # while maintaining C5-freeness
        
        num_attempts = 500
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Initialize: 8xK4 structure
            adj = [0] * target_n
            for b in range(8):
                for i in range(4):
                    for j in range(i + 1, 4):
                        u = 4 * b + i
                        v = 4 * b + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Add edges from extra vertices (32..target_n-1) to existing vertices
            # To avoid C5: extra vertices should connect in a way that doesn't create 5-cycles
            # A safe strategy: connect each extra vertex to a subset of one K4 block
            # or to vertices from at most 2 blocks, carefully chosen
            
            # Random strategy: for each extra vertex, connect to random subset
            # of vertices, biased toward making connections that don't create C5s
            
            for ex in range(32, target_n):
                # Connect to some vertices
                # To keep C5-free: if we connect ex to vertices in different blocks,
                # we need to be careful. Safest: connect to a clique-like subset.
                # 
                # Let's try: connect ex to 2-4 vertices from a single K4 block
                # This keeps the structure mostly C5-free since within a K4,
                # adding a vertex connected to some of them creates triangles but not C5s
                # (a C5 would need 5 distinct vertices in a cycle, and within one block
                # of 4+1=5 vertices, we'd need a 5-cycle which is impossible in a
                # near-clique structure)
                
                # Pick a random block
                block = random.randint(0, 7)
                base = 4 * block
                
                # Connect to 1-4 vertices in this block
                num_conn = random.randint(1, 4)
                targets = random.sample(range(4), num_conn)
                for t in targets:
                    v = base + t
                    adj[ex] |= (1 << v)
                    adj[v] |= (1 << ex)
                
                # Optionally connect to another block (risky for C5)
                if random.random() < 0.3 and extra > 1:
                    block2 = random.randint(0, 7)
                    if block2 != block:
                        base2 = 4 * block2
                        # Connect to 1-2 vertices in second block
                        num_conn2 = random.randint(1, 2)
                        targets2 = random.sample(range(4), num_conn2)
                        for t in targets2:
                            v = base2 + t
                            adj[ex] |= (1 << v)
                            adj[v] |= (1 << ex)
            
            # Check validity
            if has_c5(adj, target_n):
                continue
            if has_independent_set_9(adj, target_n):
                continue
            
            # Valid!
            edges = adj_to_edges(adj, target_n)
            best_result = {'n': target_n, 'edges': edges}
            break
    
    # Strategy 2: If we didn't find anything better, try local search on 8xK4
    # by randomly modifying edges and checking
    
    if best_result['n'] == 32 and time.time() < deadline:
        # Try to find n=33 by more aggressive search
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(1000):
                if time.time() > deadline:
                    break
                
                # Random graph with moderate density
                # For C5-free: density should be low, around 0.1-0.3
                p = random.uniform(0.05, 0.25)
                
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue
                
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            
            if found:
                break
    
    return best_result

# EVOLVE-BLOCK-END
