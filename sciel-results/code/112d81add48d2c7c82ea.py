import time
import sys
from collections import deque

DEADLINE_MARGIN = 10.0  # seconds before hard limit to start returning
HARD_LIMIT = 150.0      # seconds

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and (w in adj[u]):
                    return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks."""
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # For each vertex in order, precompute which colors are forbidden by which earlier neighbors
    # We'll use a recursive backtracking with the ordering
    
    colors = [None] * n  # 0,1,2 or None
    # available_mask[v] = bitmask of colors still available at v given current assignments
    # We recompute as we go for simplicity
    
    def backtrack(idx):
        if time.time() > DEADLINE - 2:  # safety
            return None
        if idx == n:
            return True
        
        v = order[idx]
        # Determine which colors are forbidden
        forbidden = 0
        for nb in adj[v]:
            c = colors[nb]
            if c is not None:
                forbidden |= (1 << c)
        
        # Try available colors
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = None
        
        return False
    
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with the Mycielski of C5 (Grötzsch graph) extended to 16 vertices
    # Vertices 0-4: C5, 5-9: second layer, 10: third layer
    # 11-15: isolated initially
    
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    # C5
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    # Layer 2
    for i in range(5):
        add_edge(i, 5 + i)
        add_edge(i, 5 + (i - 1) % 5)
        add_edge(i, 5 + (i + 1) % 5)
    # Wait, that's not right. Mycielski construction:
    # Original C5: 0-1-2-3-4-0
    # New vertices 5-9, where 5+i connects to i's neighbors in C5
    # Actually standard Mycielski: M(G) has V(G) ∪ {v' : v in V(G)} ∪ {w}
    # v' connects to N(v)
    # w connects to all v'
    
    # Let me redo this properly
    adj = [set() for _ in range(n)]
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    
    # Mycielski layer: 5-9 where 5+i connects to neighbors of i in C5
    for i in range(5):
        neighbors = [(i + 1) % 5, (i - 1) % 5]
        for nb in neighbors:
            add_edge(5 + i, nb)
    
    # Universal vertex 10 connects to all 5-9
    for i in range(5):
        add_edge(10, 5 + i)
    
    # Vertices 11-15 start isolated
    
    # Current edge count
    edge_count = sum(len(adj[u]) for u in range(n)) // 2
    
    # Now try to add more edges to vertices 11-15 and between them
    # while maintaining triangle-free and not-3-colorable
    
    # Strategy: try adding edges greedily, checking constraints
    # We want to maximize edges, so try to connect 11-15 as much as possible
    
    # First, let's think about what edges can be added
    # For triangle-free: if we add edge (u,v), then no common neighbor of u and v
    
    def can_add_edge(u, v):
        if u in adj[v]:
            return False
        # Check triangle-free
        common = adj[u] & adj[v]
        if common:
            return False
        return True
    
    # We want to make the graph not 3-colorable
    # The base graph (0-10) is already not 3-colorable (it's the Grötzsch graph)
    # Adding isolated vertices 11-15 keeps it not 3-colorable
    # But we want to ADD edges to 11-15 to increase edge count
    
    # Key insight: if we connect 11-15 to each other and to 0-10, we need to be careful
    # to not create triangles and to maintain non-3-colorability
    
    # Actually, since the base is not 3-colorable, adding any edges (that don't create triangles)
    # will still result in a not-3-colorable graph, because if the whole graph were 3-colorable,
    # the subgraph on 0-10 would also be 3-colorable.
    
    # So we just need to maximize edges while maintaining triangle-free!
    
    # This is now: find maximum number of edges in a triangle-free graph on 16 vertices
    # that contains the Grötzsch graph as a subgraph.
    
    # By Turán's theorem, max edges in triangle-free graph on 16 vertices is floor(16^2/4) = 64
    # (complete bipartite K_{8,8})
    
    # But we must contain the Grötzsch graph, which is not bipartite (it's odd cycle based)
    # So we can't just take K_{8,8}.
    
    # Let's try a different approach: start with the Grötzsch graph and greedily add edges
    
    # Actually, let me reconsider. The Grötzsch graph has 11 vertices and 20 edges.
    # We have 5 isolated vertices to work with.
    
    # Let's try to add as many edges as possible involving vertices 11-15.
    
    # First, try connecting 11-15 to existing vertices
    for new_v in range(11, 16):
        # Try to connect new_v to as many existing vertices as possible
        # while maintaining triangle-free
        for existing in range(new_v):
            if can_add_edge(new_v, existing):
                add_edge(new_v, existing)
    
    # Then try to add edges between 11-15
    for i in range(11, 16):
        for j in range(i + 1, 16):
            if can_add_edge(i, j):
                add_edge(i, j)
    
    # Now try to add more edges among 0-10 if possible
    for u in range(11):
        for v in range(u + 1, 11):
            if can_add_edge(u, v):
                add_edge(u, v)
    
    # Repeat the above to see if we can add more (in case adding edges between 11-15
    # opens up possibilities... actually no, it would only close them)
    
    # Let's do multiple passes to ensure maximum
    for _ in range(10):
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    add_edge(u, v)
                    changed = True
        if not changed:
            break
    
    # Count edges
    final_edges = sum(len(adj[u]) for u in range(n)) // 2
    
    # Verify it's still not 3-colorable
    # (It should be, since the Grötzsch subgraph is still there)
    # But let's verify to be safe
    # Actually, let's just trust the structure since we only added edges to a non-3-colorable base
    
    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
