import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    # Strategy: Tree Gluing
    # Build two 3-regular trees of depth 7 (radius 7 from center edge)
    # Center edge (0, 1)
    # Tree 0: Root 0, expand to depth 7
    # Tree 1: Root 1, expand to depth 7
    
    # We will generate a random matching between the leaves of Tree 0 and Tree 1
    # and check if the resulting graph has girth >= 16.
    
    # To increase the chance of success, we can try multiple random matchings.
    
    max_attempts = 2000
    
    for attempt in range(max_attempts):
        n, edges, leaves_u, leaves_v, tree_edges = _build_trees_and_leaves()
        
        # leaves_u: list of (vertex_id, stub_index) for Tree 0 leaves
        # leaves_v: list of (vertex_id, stub_index) for Tree 1 leaves
        # Each leaf has 2 stubs.
        
        # Create a list of all stubs for U and V
        stubs_u = []
        for v, _ in leaves_u:
            stubs_u.append(v)
            stubs_u.append(v)
            
        stubs_v = []
        for v, _ in leaves_v:
            stubs_v.append(v)
            stubs_v.append(v)
            
        # Randomly shuffle V stubs and pair with U stubs
        random.shuffle(stubs_v)
        
        # Add the matching edges
        matching_edges = []
        for u_stub, v_stub in zip(stubs_u, stubs_v):
            # Ensure u_stub is from Tree 0 and v_stub is from Tree 1
            # In our construction, Tree 0 vertices are 0..254, Tree 1 are 255..509
            if u_stub > v_stub:
                u_stub, v_stub = v_stub, u_stub
            matching_edges.append((u_stub, v_stub))
            
        # Combine edges
        all_edges = tree_edges + matching_edges
        
        # Check for multi-edges (should not happen if we are careful, but tree edges are unique)
        # Tree edges are internal to trees. Matching edges are between trees.
        # So no multi-edges unless we connect a leaf to itself? No, U and V are disjoint.
        
        # Check girth
        if _check_girth(all_edges, n, g):
            return n, all_edges
            
    # Fallback to LUW
    return _luw_construct(k, g)

def _build_trees_and_leaves():
    """
    Builds two 3-regular trees of depth 7 rooted at 0 and 1.
    Returns:
    n: total vertices
    edges: list of tree edges
    leaves_u: list of leaf vertex ids in Tree 0
    leaves_v: list of leaf vertex ids in Tree 1
    """
    # Tree 0: Root 0
    # Tree 1: Root 1
    # Center edge (0, 1)
    
    n = 510 # 255 + 255
    edges = [(0, 1)]
    
    leaves_u = []
    leaves_v = []
    
    # Build Tree 0
    # Queue: (vertex, parent, depth)
    # Root 0 has parent -1, depth 0
    # But 0 is connected to 1. So 0 has 2 other neighbors.
    # We treat 0 as having 2 "free" stubs for tree expansion.
    
    queue_u = [(0, -1, 0)]
    # We need to assign IDs.
    # Tree 0 uses 0..254
    # Tree 1 uses 255..509
    
    # Let's map local IDs to global IDs.
    # Tree 0: local 0 is global 0.
    # Tree 1: local 0 is global 255.
    
    # Build Tree 0 structure
    # We will generate the tree structure and then map to IDs.
    
    def build_tree(root_local, offset, is_u):
        # Returns list of edges (local), list of leaves (local)
        local_edges = []
        local_leaves = []
        # Queue: (node, parent, depth)
        # Root has 2 children (since 1 edge is used for center)
        # Others have 2 children.
        
        # We need to assign unique local IDs.
        # Let's just use a list of nodes.
        nodes = [root_local]
        # We need to know which edges are "used" for parent.
        
        # Simpler: BFS
        # Node 0 is root.
        # Children of 0: 2 nodes.
        
        # Let's assign IDs sequentially.
        # 0 is root.
        # Next available ID is 1.
        
        # Queue: (node_id, parent_id, depth)
        q = deque([(root_local, -1, 0)])
        
        # We need to track which edges are taken.
        # For root, 1 edge is taken by center. 2 free.
        # For others, 1 edge taken by parent. 2 free.
        
        while q:
            node, parent, depth = q.popleft()
            
            if depth == 7:
                local_leaves.append(node)
                continue
                
            # Generate 2 children
            for _ in range(2):
                new_id = len(nodes)
                nodes.append(new_id)
                local_edges.append((node, new_id))
                q.append((new_id, node, depth + 1))
                
        return local_edges, local_leaves, len(nodes)
        
    # Build Tree 0
    edges_u_local, leaves_u_local, count_u = build_tree(0, 0, True)
    # Build Tree 1
    edges_v_local, leaves_v_local, count_v = build_tree(0, 0, False)
    
    # Map to global IDs
    # Tree 0: global_id = local_id
    # Tree 1: global_id = local_id + 255
    
    global_edges = []
    for u, v in edges_u_local:
        global_edges.append((u, v))
        
    for u, v in edges_v_local:
        global_edges.append((u + 255, v + 255))
        
    global_leaves_u = [l for l in leaves_u_local]
    global_leaves_v = [l + 255 for l in leaves_v_local]
    
    # Add center edge
    global_edges.append((0, 255))
    
    return n, global_edges, global_leaves_u, global_leaves_v

def _check_girth(edges, n, g):
    """Check if graph has girth at least g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length < g
    # Since g=16, we check for 4, 6, ..., 14.
    # BFS from each vertex up to depth 7.
    
    for start in range(n):
        # dist[v] = (dist, parent)
        dist = {start: (0, -1)}
        queue = deque([(start, 0, -1)])
        
        while queue:
            node, depth, parent = queue.popleft()
            if depth >= 7: # Max depth to find cycle of length 14 is 7 (7+7)
                # Actually, cycle length L = d1 + d2 + 1?
                # If we meet at depth d1 and d2, cycle is d1 + d2 + 1?
                # No, if we meet neighbor, cycle is depth + dist[neighbor] + 1.
                # We want cycle < 16.
                # So depth + dist[neighbor] + 1 < 16
                # depth + dist[neighbor] < 15
                # Max depth 7, max dist 7 -> 14 + 1 = 15 < 16.
                # So we need to check up to depth 7.
                continue
                
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in dist:
                    cycle_len = dist[neighbor][0] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[neighbor] = (depth + 1, node)
                    queue.append((neighbor, depth + 1, node))
                    
    return True

def _luw_construct(k, g):
    """Fallback to LUW construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
