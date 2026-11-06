import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145
    n = 20
    K = 20
    
    # Precompute all 4-cycles in K_n for fast validation
    # A 4-cycle is a set of 4 edges {ab, bc, cd, da} where a,b,c,d are distinct
    # We represent each 4-cycle as a frozenset of 4 edge-keys
    # Edge key: (min(u,v), max(u,v))
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
    
    # Generate all 4-cycles: choose 4 distinct vertices, then 3 ways to form a cycle
    all_4cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Three distinct 4-cycles on these 4 vertices:
        # 1: a-b-c-d-a
        # 2: a-b-d-c-a
        # 3: a-c-b-d-a
        cycles = [
            frozenset([edge_key(a,b), edge_key(b,c), edge_key(c,d), edge_key(d,a)]),
            frozenset([edge_key(a,b), edge_key(b,d), edge_key(d,c), edge_key(c,a)]),
            frozenset([edge_key(a,c), edge_key(c,b), edge_key(b,d), edge_key(d,a)]),
        ]
        all_4cycles.extend(cycles)
    
    # Remove duplicates
    all_4cycles = list(set(all_4cycles))
    
    def validate_coloring(colors_dict):
        """Check if all 4-cycles in the graph defined by colors_dict are rainbow."""
        for cycle in all_4cycles:
            # Check if all 4 edges of this cycle exist in the graph
            if all(e in colors_dict for e in cycle):
                if len(set(colors_dict[e] for e in cycle)) < 4:
                    return False
        return True
    
    def find_valid_color_fast(adj, colors, u, v):
        """Find a color for edge (u,v) such that all 4-cycles containing it remain rainbow.
        Uses the precomputed 4-cycles for speed."""
        u, v = edge_key(u, v)
        # Find all 4-cycles that include edge (u,v)
        # For each such cycle, the other 3 edges must exist and have colors
        # The new color must differ from all 3 existing colors
        
        # For efficiency, precompute which 4-cycles contain each edge
        # But that's too large. Instead, iterate over potential cycles.
        # A 4-cycle containing (u,v): u-v-x-y-u or u-v-y-x-u
        # So x and y are common neighbors of u and v (excluding each other)
        
        # Actually, let's use the precomputed cycles but filter
        # This might be slow. Let's use a smarter approach.
        
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        if not common:
            # No 4-cycle can contain (u,v), so any color works
            return 0
        
        # For each pair (x, y) of common neighbors, x != y:
        # Cycle: u - v - x - y - u
        # Edges: (u,v)=c, (v,x), (x,y), (y,u)
        # Need c != colors[(v,x)], c != colors[(x,y)], c != colors[(y,u)]
        
        forbidden = set()
        for x in common:
            for y in common:
                if x == y:
                    continue
                # Check if edges (v,x), (x,y), (y,u) all exist
                e_vx = edge_key(v, x)
                e_xy = edge_key(x, y)
                e_yu = edge_key(y, u)
                if e_vx not in colors or e_xy not in colors or e_yu not in colors:
                    continue
                forbidden.add(colors[e_vx])
                forbidden.add(colors[e_xy])
                forbidden.add(colors[e_yu])
        
        # Find a color not in forbidden
        for c in range(K):
            if c not in forbidden:
                return c
        return None
    
    # Strategy: Start with a random sparse graph, then greedily add edges
    # Use multiple random restarts
    
    best_edges = None
    best_count = 0
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random initial seed: a random matching or small random graph
        adj = {i: set() for i in range(n)}
        colors = {}
        
        # Start with a random sparse graph
        all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_possible)
        
        # Add a random subset initially
        initial_size = random.randint(10, 30)
        for u, v in all_possible[:initial_size]:
            c = random.randint(0, K - 1)
            adj[u].add(v)
            adj[v].add(u)
            colors[edge_key(u, v)] = c
        
        # Check if initial graph is valid
        if not validate_coloring(colors):
            # Remove edges until valid (or restart)
            # Just restart
            continue
        
        current_count = len(colors)
        
        # Greedy addition phase
        remaining = [e for e in all_possible if e not in colors]
        random.shuffle(remaining)
        
        for u, v in remaining:
            if time.time() > deadline - 5:
                break
            if v in adj[u]:
                continue
            
            c = find_valid_color_fast(adj, colors, u, v)
            if c is not None:
                adj[u].add(v)
                adj[v].add(u)
                colors[edge_key(u, v)] = c
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v, colors[edge_key(u, v)]] for u, v in colors]
    
    # Local search: try to recolor edges to enable more additions
    if best_edges is not None and best_count > 0:
        adj = {i: set() for i in range(n)}
        colors = {}
        for u, v, c in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            colors[edge_key(u, v)] = c
        
        # Try to add more edges with different color assignments
        remaining = [(i, j) for i in range(n) for j in range(i + 1, n) if j not in adj[i]]
        random.shuffle(remaining)
        
        for u, v in remaining:
            if time.time() > deadline - 3:
                break
            c = find_valid_color_fast(adj, colors, u, v)
            if c is not None:
                adj[u].add(v)
                adj[v].add(u)
                colors[edge_key(u, v)] = c
                best_count += 1
                best_edges.append([u, v, c])
    
    # Final verification
    if best_edges:
        final_colors = {}
        for u, v, c in best_edges:
            final_colors[edge_key(u, v)] = c
        if not validate_coloring(final_colors):
            # Fallback: use a simple valid construction
            # A star graph is always valid (no 4-cycles)
            best_edges = [[0, v, v % K] for v in range(1, n)]
            best_count = n - 1
    
    if best_edges is None:
        best_edges = [[0, v, v % K] for v in range(1, n)]
        best_count = n - 1
    
    return best_edges
