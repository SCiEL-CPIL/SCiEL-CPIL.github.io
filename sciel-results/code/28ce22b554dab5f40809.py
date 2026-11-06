"""Combined program: deterministic C4-free construction + fast verification."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _independent_set_size_bitmask(n, adj, target=11):
    """Check if there exists an independent set of size >= target using branching with bitmasks."""
    
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        # Upper bound: if number of vertices left < remaining, return count
        count = 0
        temp = vertices_mask
        while temp:
            temp &= temp - 1
            count += 1
        if count < remaining:
            return count
        
        # Pick a vertex to branch on - use first set bit
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _try_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if safe to add."""
    # A C4 is created if there exist x, y such that:
    # x is adjacent to u, y is adjacent to v, x is adjacent to y, and x != v, y != u
    # Equivalently: common neighbors of u and v (excluding each other) has at least 2 vertices
    # that are adjacent to each other... no wait.
    # 
    # Adding edge (u,v) creates a C4 if there exist x, y such that:
    # x ~ u, y ~ v, x ~ y, and x != v, y != u, and x != y
    # This forms cycle u-x-y-v-u
    # 
    # So we need: exists x in N(u)\{v} and y in N(v)\{u} such that x~y
    # Equivalently: (N(u) & N(v)) has some pair that are adjacent? No.
    # 
    # Actually: C4 is u-x-y-v-u where x~u, x~y, y~v, y~u (new edge)
    # So x must be in N(u), y must be in N(v), and x~y
    # 
    # So: for any x in N(u) and y in N(v) with x != v, y != u, if x~y then C4 exists.
    # 
    # More efficiently: let S = N(u) & N(v). If |S| >= 2, then any two vertices in S 
    # are both adjacent to u and v, so u-a-v-b-u is a C4 if a~b. But we don't need a~b.
    # Wait no: u-a-v-b-u requires a~v (yes, a in N(v)), b~u (yes, b in N(u)), 
    # and we need a~b? No! The cycle is u-a, a-v, v-b, b-u. 
    # a-v is an edge (a in N(v)), v-b is an edge (b in N(v))... wait b is in N(u) not N(v).
    # 
    # Let me redo: cycle u-a-v-b-u:
    # u-a: edge (a in N(u))
    # a-v: edge (a in N(v))  
    # v-b: edge (b in N(v))
    # b-u: edge (b in N(u))
    # So we need a in N(u)∩N(v) and b in N(u)∩N(v), a != b.
    # Then u-a-v-b-u is a 4-cycle if a != b and all four edges exist.
    # Yes! So adding (u,v) creates a C4 iff |N(u) ∩ N(v)| >= 2.
    
    common = adj[u] & adj[v]
    return not (common & (common - 1))  # safe if fewer than 2 common neighbors

def _construct_39_vertex_graph():
    """
    Construct a C4-free graph on 39 vertices with independence number < 11.
    Uses a systematic construction based on finite geometry.
    """
    n = 39
    deadline = time.time() + 110  # Leave margin for verification
    
    # Strategy: Build graph in layers to maintain C4-freeness
    # Use a construction inspired by polarity graphs
    
    best_result = None
    
    for seed in range(500):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        adj = [0] * n
        edges = []
        
        # Phase 1: Create a base structure with good properties
        # Use a random regular-like graph as starting point
        # Target degree ~6-8 for 39 vertices
        
        # Start with a random graph of moderate density
        for i in range(n):
            # Connect to 4-6 random other vertices
            candidates = [j for j in range(n) if j != i and not (adj[i] >> j) & 1]
            random.shuffle(candidates)
            num_connect = random.randint(4, 6)
            for j in candidates[:num_connect]:
                if _try_add_edge(adj, n, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        # Phase 2: Try to add more edges to reduce independence number
        # while maintaining C4-freeness
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Pick a random non-edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j or (adj[i] >> j) & 1:
                continue
            
            if _try_add_edge(adj, n, i, j):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
        
        # Check if this graph is valid
        if _has_c4_bitmask(n, adj):
            continue
        
        if _independent_set_size_bitmask(n, adj, 11) < 11:
            best_result = {'n': n, 'edges': edges.copy()}
            break
    
    if best_result:
        return best_result
    
    # Fallback: try a more structured approach
    # Use 13 triangles with carefully added inter-triangle edges
    adj = [0] * n
    edges = []
    for t in range(13):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges.append([a, b])
        edges.append([a, c])
        edges.append([b, c])
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Add inter-triangle edges using a systematic pattern
    # Connect triangle i to triangle (i+1) mod 13 with one edge
    for i in range(13):
        j = (i + 1) % 13
        u = 3*i
        v = 3*j
        if _try_add_edge(adj, n, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    # Add more edges: connect triangle i to triangle (i+2) mod 13
    for i in range(13):
        j = (i + 2) % 13
        u = 3*i + 1
        v = 3*j + 1
        if _try_add_edge(adj, n, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    # Add more edges: connect triangle i to triangle (i+3) mod 13
    for i in range(13):
        j = (i + 3) % 13
        u = 3*i + 2
        v = 3*j + 2
        if _try_add_edge(adj, n, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    if not _has_c4_bitmask(n, adj) and _independent_set_size_bitmask(n, adj, 11) < 11:
        return {'n': n, 'edges': edges}
    
    # Final fallback: 10 disjoint triangles
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to construct a 39-vertex graph
    result = _construct_39_vertex_graph()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
