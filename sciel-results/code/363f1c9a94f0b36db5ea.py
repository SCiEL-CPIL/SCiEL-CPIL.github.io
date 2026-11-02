import time
import sys
from collections import deque
import random

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
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

def _construct_moores_tree_cage(k, g):
    if k != 4 or g != 10:
        return None
        
    depth = 5
    n = 1
    for i in range(1, depth + 1):
        n += k * (k - 1) ** i
        
    # Build tree
    edges = []
    layers = [[0]]
    current_id = 1
    
    for d in range(1, depth + 1):
        layer_nodes = []
        for u in layers[d-1]:
            num_children = k if d == 1 else k - 1
            for _ in range(num_children):
                v = current_id
                current_id += 1
                layer_nodes.append(v)
                edges.append((u, v))
        layers.append(layer_nodes)
        
    n = current_id
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    leaves = layers[depth]
    L = len(leaves)
    
    # Determine branch for each leaf
    # Branch is determined by the child of the root (layer 1) that is an ancestor.
    # Layer 1 nodes are 1, 2, 3, 4.
    # We can trace back from each leaf to its layer 1 ancestor.
    parent = [-1] * n
    for u, v in edges:
        # In the tree, u is parent of v if u is in a lower layer
        # We built it top-down, so u is parent.
        if layers.index([u][0]) < layers.index([v][0]) if u in layers[0] or v in layers[0] else False:
            pass # This logic is messy.
            
    # Better: store parent during construction
    parent = [-1] * n
    # Re-do construction to store parent
    edges = []
    layers = [[0]]
    current_id = 1
    parent = [-1] * n
    for d in range(1, depth + 1):
        layer_nodes = []
        for u in layers[d-1]:
            num_children = k if d == 1 else k - 1
            for _ in range(num_children):
                v = current_id
                current_id += 1
                layer_nodes.append(v)
                parent[v] = u
                edges.append((u, v))
        layers.append(layer_nodes)
        
    # Map leaf to branch (layer 1 node)
    leaf_branch = {}
    for leaf in leaves:
        cur = leaf
        while parent[cur] != 0:
            cur = parent[cur]
        # cur is the layer 1 node
        leaf_branch[leaf] = cur
        
    # Branches are 1, 2, 3, 4
    branch_ids = [1, 2, 3, 4]
    
    # Construct 3-regular graph on leaves with no intra-branch edges
    # We use a randomized greedy approach with restarts
    best_result = None
    max_attempts = 50
    for attempt in range(max_attempts):
        random.seed(attempt) # Reproducible attempts
        deg = {leaf: 0 for leaf in leaves}
        new_edges = []
        success = True
        
        # We need to add 3 edges for each leaf
        # Total edges = 3 * L / 2 = 486
        # We pick random pairs
        
        # To ensure we don't get stuck, we prioritize leaves with fewest options?
        # Or just random.
        
        # List of leaves
        leaf_list = list(leaves)
        
        # We will try to fill degrees
        # While there are leaves with deg < 3
        # Pick a random leaf u with deg < 3
        # Pick a random leaf v with deg < 3 and branch(v) != branch(u)
        # If found, add edge, update deg
        # If not, fail
        
        # To speed up, we can pre-compute allowed neighbors
        # But L=324, so O(L^2) is fine.
        
        # Shuffle leaves to break symmetry
        random.shuffle(leaf_list)
        
        # We process leaves in order? No, random.
        # We maintain a list of leaves that still need edges.
        # Actually, we just iterate until all deg are 3.
        
        # To avoid infinite loops, we limit steps
        steps = 0
        max_steps = 10000
        while any(deg[leaf] < 3 for leaf in leaves):
            if steps > max_steps:
                success = False
                break
            steps += 1
            
            # Find a leaf with deg < 3
            # To be efficient, keep a list of unsaturated leaves
            # For simplicity, pick random
            u = random.choice(leaf_list)
            if deg[u] >= 3:
                continue
                
            # Find v
            # v must be in a different branch
            # And deg[v] < 3
            # And (u,v) not in new_edges
            
            # We can pick random v from leaves
            # Check conditions
            found_v = False
            # Shuffle leaf_list to pick random v
            # But we need to pick from ALL leaves
            for v in random.sample(leaf_list, L):
                if v == u:
                    continue
                if deg[v] >= 3:
                    continue
                if leaf_branch[u] == leaf_branch[v]:
                    continue
                # Check if edge exists
                # We can use a set for new_edges
                # But checking set is O(1)
                # We need to store edges
                if (u, v) in [(e[0], e[1]) for e in new_edges] or (v, u) in [(e[0], e[1]) for e in new_edges]:
                    continue
                # Add edge
                new_edges.append((u, v))
                deg[u] += 1
                deg[v] += 1
                found_v = True
                break
                
            if not found_v:
                success = False
                break
                
        if success:
            # Verify
            for leaf in leaves:
                if deg[leaf] != 3:
                    success = False
                    break
            
            if success:
                # Add to adj
                for u, v in new_edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Check girth
                girth = _bfs_girth(adj)
                if girth >= g:
                    return n, edges + new_edges
                else:
                    # Remove new edges
                    for u, v in new_edges:
                        adj[u].remove(v)
                        adj[v].remove(u)
                    # Try next attempt
                    pass
                    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        try:
            result = _construct_moores_tree_cage(k, g)
            if result:
                n, edges = result
                return n, edges
        except Exception:
            pass
            
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        n, edges = _luw_construct(k, g)
        return n, edges
