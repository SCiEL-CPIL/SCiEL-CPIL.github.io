"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    total_possible = len(all_pairs)  # 190
    
    # Precompute 4-cycles for each edge
    # For edge (u,v), a 4-cycle is u-v-x-y-u where x~v, y~u, x~y, all distinct from u,v
    # We'll store: for each edge (u,v), the list of (x, y, color_needed) constraints
    # Actually, let's precompute the structure of 4-cycles per edge
    
    # For each pair (u,v), find all 4-cycles containing it
    # A 4-cycle containing (u,v) has form: u - v - x - y - u
    # where x != u,v, y != u,v,x, and edges (v,x), (x,y), (y,u) exist
    # The colors on the cycle are: c_uv, c_vx, c_xy, c_yu - must all be distinct
    
    # Precompute for each edge (u,v) the list of potential 4-cycles
    # Each cycle is represented by (x, y) meaning the other two vertices
    # The constraint is: color(u,v), color(v,x), color(x,y), color(y,u) all distinct
    
    edge_4cycles = defaultdict(list)  # edge -> list of (x, y)
    
    for u in range(n):
        for v in range(u+1, n):
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    # Check if (v,x), (x,y), (y,u) form valid edges in complete graph
                    # In complete graph, all edges exist, so just check distinctness
                    # But we need to avoid double counting: each 4-cycle u-v-x-y-u
                    # is the same as u-y-x-v-u (reversed)
                    # To avoid double counting, we can impose an ordering
                    # Let's just collect all and deduplicate later, or use a canonical form
                    edge_4cycles[(u,v)].append((x, y))
    
    # For efficient checking, let's also build reverse: for each edge, which other edges are in 4-cycles with it
    # This is the conflict graph structure
    
    def count_bad_cycles_fast(adj):
        """Count non-rainbow 4-cycles. adj is a dict mapping (u,v) -> color, or -1 if absent."""
        bad = 0
        for (u, v), c_uv in adj.items():
            if c_uv == -1:
                continue
            for (x, y) in edge_4cycles[(u,v)]:
                # Check if edges (v,x), (x,y), (y,u) exist
                c_vx = adj.get((min(v,x), max(v,x)), -1)
                c_xy = adj.get((min(x,y), max(x,y)), -1)
                c_yu = adj.get((min(y,u), max(y,u)), -1)
                if c_vx != -1 and c_xy != -1 and c_yu != -1:
                    if len({c_uv, c_vx, c_xy, c_yu}) < 4:
                        bad += 1
        return bad
    
    def make_edges(adj):
        edges = []
        for (u, v), c in adj.items():
            if c != -1:
                edges.append([u, v, c])
        return edges
    
    best_edges = [[0, v, 0] for v in range(1, n)]
    best_count = len(best_edges)
    
    def try_refine(adj, max_time_left):
        """Refine a coloring by recoloring and deleting to eliminate bad cycles."""
        current_bad = count_bad_cycles_fast(adj)
        
        while current_bad > 0 and time.time() < deadline - 5:
            # Find edges involved in bad cycles
            # Try to recolor edges to reduce bad cycles
            improved = False
            
            # Get all edges with their colors
            edge_list = [(e, c) for e, c in adj.items() if c != -1]
            random.shuffle(edge_list)
            
            for (u, v), old_c in edge_list:
                if time.time() > deadline - 5:
                    break
                
                # Try each new color
                for new_c in range(n):
                    if new_c == old_c:
                        continue
                    adj[(u, v)] = new_c
                    new_bad = count_bad_cycles_fast(adj)
                    if new_bad < current_bad:
                        current_bad = new_bad
                        improved = True
                        break
                    adj[(u, v)] = old_c
                
                if improved:
                    break
            
            if not improved:
                # Try deleting an edge
                # Find the edge in the most bad cycles
                edge_bad_counts = defaultdict(int)
                for (u, v), c_uv in adj.items():
                    if c_uv == -1:
                        continue
                    for (x, y) in edge_4cycles[(u,v)]:
                        c_vx = adj.get((min(v,x), max(v,x)), -1)
                        c_xy = adj.get((min(x,y), max(x,y)), -1)
                        c_yu = adj.get((min(y,u), max(y,u)), -1)
                        if c_vx != -1 and c_xy != -1 and c_yu != -1:
                            if len({c_uv, c_vx, c_xy, c_yu}) < 4:
                                edge_bad_counts[(u,v)] += 1
                                edge_bad_counts[(min(v,x), max(v,x))] += 1
                                edge_bad_counts[(min(x,y), max(x,y))] += 1
                                edge_bad_counts[(min(y,u), max(y,u))] += 1
                
                if edge_bad_counts:
                    worst_edge = max(edge_bad_counts, key=edge_bad_counts.get)
                    adj[worst_edge] = -1
                    current_bad = count_bad_cycles_fast(adj)
                else:
                    break
        
        return adj, current_bad
    
    # Strategy 1: Start with complete graph, assign colors, then refine
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
        
        # Assign initial colors
        strategy = attempt % 5
        
        adj = {}
        for u in range(n):
            for v in range(u+1, n):
                if strategy == 0:
                    c = random.randint(0, n-1)
                elif strategy == 1:
                    c = (u + v) % n
                elif strategy == 2:
                    c = (u * v) % n
                elif strategy == 3:
                    # Vertex-based: each vertex has an offset
                    offsets = list(range(n))
                    random.shuffle(offsets)
                    c = (offsets[u] + offsets[v]) % n
                else:
                    # Random permutation-based
                    perm = list(range(n))
                    random.shuffle(perm)
                    c = perm[(u * n + v) % n]
                
                adj[(u, v)] = c
        
        # Refine
        adj, bad = try_refine(adj, deadline - time.time())
        
        if bad == 0:
            ec = sum(1 for c in adj.values() if c != -1)
            if ec > best_count:
                best_count = ec
                best_edges = make_edges(adj)
    
    # Strategy 2: Greedy construction - add edges one by one with best color
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        adj = {}
        edge_order = all_pairs[:]
        random.shuffle(edge_order)
        
        for u, v in edge_order:
            if time.time() > deadline - 5:
                break
            
            # Try each color, pick the one that minimizes new bad cycles
            best_c = -1
            best_bad = float('inf')
            
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                adj[(u, v)] = c
                bad = count_bad_cycles_fast(adj)
                if bad < best_bad:
                    best_bad = bad
                    best_c = c
                if bad == 0:
                    break
                del adj[(u, v)]
            
            if best_c != -1 and best_bad <= 2:  # Only add if it doesn't create too many bad cycles
                adj[(u, v)] = best_c
        
        # Refine
        adj, bad = try_refine(adj, deadline - time.time())
        
        if bad == 0:
            ec = sum(1 for c in adj.values() if c != -1)
            if ec > best_count:
                best_count = ec
                best_edges = make_edges(adj)
    
    # Strategy 3: Start from a known good structure and modify
    # Use a proper edge coloring of K_20 (requires 19 colors for 1-factorization)
    # Actually K_20 has degree 19, so it needs 19 colors for proper edge coloring
    # But we have 20 colors available
    
    # 1-factorization of K_20: vertices 0..19, 19 perfect matchings
    # Standard construction: for each i in 0..18, matching M_i
    # This gives a proper edge coloring with 19 colors
    # Then we can try to use the 20th color to break rainbow constraints
    
    if time.time() < deadline - 10:
        # Construct 1-factorization of K_20
        # Using the standard "circle method" for even n
        # Fix vertex 0, rotate the rest
        adj = {}
        
        # 1-factorization: for round r in 0..18
        # Pair vertex 0 with vertex (r+1) mod 19 + 1... 
        # Actually let me use a simpler approach
        # For K_{2m}, 1-factorization:
        # Place vertices 1..2m-1 in a circle, fix vertex 0
        # Round r: pair 0 with (r+1) mod (2m-1) + 1, and pair (r+i) mod (2m-1) + 1 with (r-i) mod (2m-1) + 1 for i=1..m-1
        
        m = 10  # n = 2m = 20
        for r in range(2*m - 1):  # 19 rounds
            # Vertex 0 pairs with vertex (r % (2*m-1)) + 1
            v0_partner = (r % (2*m - 1)) + 1
            u, v = 0, v0_partner
            adj[(min(u,v), max(u,v))] = r  # color r
            
            for i in range(1, m):
                u1 = ((r + i) % (2*m - 1)) + 1
                v1 = ((r - i) % (2*m - 1)) + 1
                u, v = min(u1, v1), max(u1, v1)
                adj[(u, v)] = r  # color r
        
        # This is a proper edge coloring (no two adjacent edges share a color)
        # But it may not satisfy the rainbow 4-cycle constraint
        # Let's check and refine
        adj, bad = try_refine(adj, deadline - time.time())
        
        if bad == 0:
            ec = sum(1 for c in adj.values() if c != -1)
            if ec > best_count:
                best_count = ec
                best_edges = make_edges(adj)
        
        # Also try with random perturbations of the 1-factorization
        for perturb in range(20):
            if time.time() > deadline - 5:
                break
            
            adj2 = dict(adj)
            # Randomly recolor some edges
            for _ in range(10):
                u, v = random.choice(all_pairs)
                adj2[(u, v)] = random.randint(0, n-1)
            
            adj2, bad = try_refine(adj2, deadline - time.time())
            
            if bad == 0:
                ec = sum(1 for c in adj2.values() if c != -1)
                if ec > best_count:
                    best_count = ec
                    best_edges = make_edges(adj2)
    
    return best_edges
# EVOLVE-BLOCK-END
