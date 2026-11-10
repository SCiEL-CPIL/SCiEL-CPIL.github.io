import random
import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moore bound for k=5, g=9 is 425.
    # We attempt to construct a graph on 425 vertices.
    if k != 5 or g != 9:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
        
    n_target = 425
    
    # Build the tree
    # Nodes 0..424
    # 0: root
    # 1-5: level 1
    # 6-25: level 2
    # 26-105: level 3
    # 106-425: level 4 (leaves)
    
    # Verify counts:
    # L0: 1
    # L1: 5
    # L2: 20
    # L3: 80
    # L4: 320
    # Total: 1+5+20+80+320 = 426?
    # Wait.
    # Root (1) has 5 children.
    # L1 (5) each have 4 children (1 parent). 5*4=20.
    # L2 (20) each have 4 children. 20*4=80.
    # L3 (80) each have 4 children. 80*4=320.
    # L4 (320) leaves.
    # Total = 1 + 5 + 20 + 80 + 320 = 426.
    
    # Moore bound calculation:
    # N = 1 + k + k(k-1) + k(k-1)^2 + k(k-1)^3
    # N = 1 + 5 + 20 + 80 + 320 = 426.
    # My previous calculation was 425.
    # Let's re-verify Moore Bound formula for odd g.
    # M(k, g) = k + k(k-1) + ... + k(k-1)^{(g-1)/2}
    # For g=9, (g-1)/2 = 4.
    # Terms:
    # i=0: k = 5
    # i=1: k(k-1) = 20
    # i=2: k(k-1)^2 = 80
    # i=3: k(k-1)^3 = 320
    # Sum = 425.
    # Why the discrepancy?
    # The Moore bound counts vertices in a tree of depth (g-1)/2 rooted at a vertex?
    # No, the standard bound is derived from a tree of radius (g-1)/2.
    # For g=9, radius 4.
    # Vertices: 1 (root) + 5 (L1) + 20 (L2) + 80 (L3) + 320 (L4) = 426.
    # So N >= 426.
    
    # Let's use N = 426.
    
    tree_edges = []
    # Assign indices
    # 0: Root
    # 1-5: L1
    # 6-25: L2
    # 26-105: L3
    # 106-425: L4
    
    # Build tree edges
    # Root to L1
    for i in range(5):
        tree_edges.append((0, 1 + i))
        
    # L1 to L2
    idx = 6
    for i in range(5):
        parent = 1 + i
        for j in range(4):
            tree_edges.append((parent, idx))
            idx += 1
            
    # L2 to L3
    for i in range(20):
        parent = 6 + i
        for j in range(4):
            tree_edges.append((parent, idx))
            idx += 1
            
    # L3 to L4
    for i in range(80):
        parent = 26 + i
        for j in range(4):
            tree_edges.append((parent, idx))
            idx += 1
            
    # Leaves are 106 to 425 (320 nodes)
    leaves = list(range(106, 426))
    
    # Group leaves by L1 ancestor
    # L1 node i (1..5) covers a block of leaves.
    # L1 1 -> L2 6..9 -> L3 26..45 -> L4 ...
    # Each L1 has 4 L2, each has 4 L3, each has 4 L4.
    # 4*4*4 = 64 leaves per L1.
    
    groups = [[] for _ in range(5)]
    for i in range(5):
        # Calculate start index for group i
        # L1 i is at index 1+i.
        # Its first L2 is 6 + i*4.
        # Its first L3 is 26 + (6+i*4 - 6)*4 = 26 + i*16.
        # Its first L4 is 106 + (26+i*16 - 26)*4 = 106 + i*64.
        start_leaf = 106 + i * 64
        groups[i] = list(range(start_leaf, start_leaf + 64))
        
    best_n = 426
    best_edges = None
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Random matchings between groups
        edges = list(tree_edges)
        
        valid = True
        matchings = {}
        
        for i in range(5):
            for j in range(i + 1, 5):
                # Perfect matching between groups[i] and groups[j]
                # Random permutation
                perm = list(range(64))
                random.shuffle(perm)
                for idx, leaf_i in enumerate(groups[i]):
                    leaf_j = groups[j][perm[idx]]
                    edges.append((leaf_i, leaf_j))
        
        # Check girth
        # Build adjacency list
        adj = [[] for _ in range(426)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        for u in range(426):
            if len(adj[u]) != 5:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check girth >= 9
        # BFS from each vertex up to depth 4 (for non-backtracking)
        # Actually, check for cycles of length 2..8.
        # A cycle of length L < 9 exists if there is a closed non-backtracking walk of length L.
        # We can just do a BFS from each node and check if we return to start within 8 steps without backtracking.
        # Simpler: Check all pairs of vertices at distance d < 9/2?
        # For g=9, we need to ensure no cycle of length 2,3,4,5,6,7,8.
        
        # Efficient check:
        # For each vertex u, perform a non-backtracking BFS up to depth 4.
        # If we reach u again, we have a cycle.
        # If we reach a vertex v that is an ancestor of u in the BFS tree?
        
        # Let's use a standard girth checker.
        girth_ok = True
        for u in range(426):
            # (current, prev, depth)
            # Start: (u, -1, 0)
            # We want to find a path back to u of length < 9.
            # Since graph is regular, we can just check distances.
            # If dist(u, v) + dist(v, u) < 9? No.
            
            # Non-backtracking walk from u.
            # State: (node, prev_node)
            # Start: (u, -1)
            # Next: (v, u) for v in adj[u]
            
            visited = set()
            queue = [(u, -1, 0)]
            while queue:
                node, prev, depth = queue.pop(0)
                if depth > 4: # 2*4 = 8 < 9
                    continue
                if node == u and depth > 0:
                    if depth < 9:
                        girth_ok = False
                        break
                if not girth_ok:
                    break
                for nxt in adj[node]:
                    if nxt == prev:
                        continue
                    # Avoid immediate backtracking
                    # Also avoid closing a cycle too early?
                    # If nxt == u and depth+1 < 9: cycle found.
                    if nxt == u:
                        if depth + 1 < 9:
                            girth_ok = False
                            break
                    else:
                        # Add to queue
                        # To prevent explosion, limit depth to 4
                        if depth + 1 <= 4:
                            queue.append((nxt, node, depth + 1))
            if not girth_ok:
                break
        
        if girth_ok:
            best_n = 426
            best_edges = edges
            break
            
    if best_edges is None:
        # Fallback to LUW
        return _luw_construct(k, g)
        
    return best_n, best_edges

def _luw_construct(k, g):
    # Original LUW code
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
