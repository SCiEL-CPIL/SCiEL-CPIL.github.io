import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cyc in _CYCLES:
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(cyc)

def is_valid_cycle(colors, cyc):
    """Check if a 4-cycle has at most 3 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    if c0 == c1 or c0 == c2 or c0 == c3:
        return True
    if c1 == c2 or c1 == c3:
        return True
    if c2 == c3:
        return True
    return False

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def build_c4_free_subgraph(rng, deadline):
    """
    Greedily build a large C4-free subgraph of K16.
    Returns a list of edge indices in the subgraph.
    """
    best_edges = []
    best_size = 0
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        edges = set()
        # Random order of edges
        edge_list = list(range(NUM_EDGES))
        rng.shuffle(edge_list)
        
        # Track adjacency to quickly check for C4
        # adj[u] is a set of neighbors
        adj = [set() for _ in range(N)]
        # Also track edge existence
        edge_set = set()
        
        current_size = 0
        for e in edge_list:
            # Decode edge
            # Find i, j from e
            # e = i*N - i*(i+1)//2 + (j-i-1)
            # This is tedious to invert, so precompute
            pass
            
        # Precompute edge endpoints
        # We need a fast way to check if adding edge (u,v) creates a C4.
        # Adding (u,v) creates a C4 if there is a common neighbor w of u and v? No.
        # A C4 is u-v-w-x-u.
        # Adding edge (u,v) creates a C4 if there exists a path of length 3 from u to v?
        # Or if there exists w, x such that u-w, w-x, x-v are in the graph?
        # Actually, adding (u,v) creates a C4 if there is a vertex w such that 
        # the number of common neighbors of u and v is ... no.
        # A C4 involving (u,v) is u-v-w-x-u.
        # This requires edges (v,w), (w,x), (x,u).
        # So we need to check if there is a path of length 3 between u and v in the current graph.
        # Or more simply: 
        # Adding (u,v) creates a C4 if there exists w, x such that 
        # (v,w) in E, (w,x) in E, (x,u) in E.
        
        # Efficient check:
        # For each w in adj[u], for each x in adj[w], if (x,v) in E, then C4 found.
        # This is O(deg(u) * avg_deg).
        
        # Let's precompute endpoints
        endpoints = []
        for i in range(N):
            for j in range(i+1, N):
                endpoints.append((i, j))
        
        for e in edge_list:
            u, v = endpoints[e]
            
            # Check if adding (u,v) creates a C4
            conflict = False
            # Path u-w-x-v
            for w in adj[u]:
                for x in adj[w]:
                    if x == v:
                        continue
                    if (x, v) in edge_set or (v, x) in edge_set:
                        # Check if x is connected to v
                        if v in adj[x]:
                            conflict = True
                            break
                if conflict:
                    break
            if conflict:
                continue
            
            # Add edge
            edges.add(e)
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].add(v)
            adj[v].add(u)
            current_size += 1
            
        if current_size > best_size:
            best_size = current_size
            best_edges = list(edges)
            
    return best_edges, best_size

def construct():
    start = time.time()
    deadline = start + 50 # Safety margin
    
    rng = random.Random(42)
    
    # Step 1: Find a large C4-free subgraph
    c4_free_edges, size = build_c4_free_subgraph(rng, deadline - 5)
    
    if size == 0:
        # Fallback to all 0s
        return [0] * NUM_EDGES
    
    # Step 2: Assign distinct colors to C4-free edges
    colors = [0] * NUM_EDGES
    used_colors = set()
    color_map = {} # edge_index -> color
    
    # Use colors 0..size-1 for the C4-free edges
    for idx, e in enumerate(c4_free_edges):
        colors[e] = idx
        used_colors.add(idx)
        color_map[e] = idx
        
    # Step 3: Assign colors to remaining edges
    # The remaining edges must take colors from used_colors
    # Constraint: For any 4-cycle with 1 edge in H and 3 in G', the color of H edge
    # must be one of the colors of the 3 G' edges.
    
    h_edges = [e for e in range(NUM_EDGES) if e not in set(c4_free_edges)]
    
    # For each h_edge, find the set of valid colors
    # valid_colors[e] = set of colors
    
    valid_colors = {}
    for e in h_edges:
        # Find all 4-cycles containing e
        # For each such cycle, if the other 3 edges are in G', 
        # then the valid colors for e are the colors of those 3 edges.
        # If there are multiple such cycles, the valid colors are the intersection.
        
        # If there are no such cycles, any color is valid?
        # But if a cycle has 2 or more edges in H, the constraint is different.
        # We only guarantee correctness for cycles with 1 edge in H if we satisfy the above.
        # Cycles with 2 edges in H: colors are from used_colors. 
        # If we pick colors arbitrarily, might be rainbow?
        # If a cycle has 2 edges in H and 2 in G', and we pick colors from used_colors,
        # it's possible to have 4 distinct colors.
        # To be safe, we should ensure that for ALL 4-cycles, the condition holds.
        
        # This CSP is complex. 
        # Simpler heuristic: 
        # Assign a random color from used_colors to each h_edge.
        # Then run a local search to fix violations.
        
        valid_colors[e] = list(used_colors)
        
    # Initialize H edges with random colors from used_colors
    for e in h_edges:
        colors[e] = rng.choice(list(used_colors))
        
    # Local search to fix violations
    # We want to minimize the number of rainbow cycles.
    # If we find a rainbow cycle, change the color of one of its edges (preferably in H) to match another.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Find a violation
        violation = None
        for cyc in _CYCLES:
            if not is_valid_cycle(colors, cyc):
                violation = cyc
                break
                
        if violation is None:
            break # No violations, we are done
            
        # Try to fix the violation by changing the color of one edge in the cycle
        # Prefer edges in H
        edges_in_cyc = list(violation)
        rng.shuffle(edges_in_cyc)
        
        fixed = False
        for e in edges_in_cyc:
            if e in set(c4_free_edges):
                continue # Don't change colors of G' edges if possible
            
            old_c = colors[e]
            # Try to change to a color that appears elsewhere in the cycle
            # Or any color that makes the cycle valid
            
            # Get colors of other edges in the cycle
            other_colors = set()
            for e2 in violation:
                if e2 != e:
                    other_colors.add(colors[e2])
                    
            # Try colors from other_colors first
            candidates = list(other_colors)
            # Also try random used colors
            candidates += rng.sample(list(used_colors), min(5, len(used_colors)))
            
            rng.shuffle(candidates)
            
            for new_c in candidates:
                if new_c == old_c:
                    continue
                colors[e] = new_c
                if is_valid_cycle(colors, violation):
                    fixed = True
                    break
                else:
                    colors[e] = old_c
                    
            if fixed:
                break
                
        if not fixed:
            # Try changing a G' edge?
            # This might break other things, but let's try
            for e in edges_in_cyc:
                old_c = colors[e]
                other_colors = set()
                for e2 in violation:
                    if e2 != e:
                        other_colors.add(colors[e2])
                candidates = list(other_colors)
                rng.shuffle(candidates)
                for new_c in candidates:
                    if new_c == old_c:
                        continue
                    colors[e] = new_c
                    if is_valid_cycle(colors, violation):
                        fixed = True
                        break
                    else:
                        colors[e] = old_c
                if fixed:
                    break
            
    # Final validation
    if not validate(colors):
        # Fallback: use only 1 color
        return [0] * NUM_EDGES
        
    return colors

construct()
