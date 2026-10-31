import time
import random
import sys
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

    def check_girth(adj, n, girth):
        """Check girth >= girth using non-backtracking BFS."""
        for start in range(n):
            visited = set()
            queue = deque()
            for nb in adj[start]:
                visited.add((start, nb))
                queue.append((start, nb, 1))
            while queue:
                u, v, dist = queue.popleft()
                if dist >= girth:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == start:
                        if dist + 1 < girth:
                            return False
                    else:
                        if (v, w) not in visited:
                            visited.add((v, w))
                            queue.append((v, w, dist + 1))
        return True

    def build_from_tree(depth, rng, seed_val=42):
        """
        Build a 3-regular graph by growing a non-backtracking tree to given depth,
        then closing the boundary vertices to form a 3-regular graph.
        
        Strategy:
        - Start with root vertex 0.
        - Grow a tree where each internal node has 2 children (3-regular: 1 parent + 2 children).
        - At depth 'depth', we have boundary vertices that need 1 more connection each.
        - Pair up boundary vertices to satisfy 3-regularity.
        - Verify girth.
        """
        rng_local = random.Random(seed_val)
        
        # Grow the tree
        # Level 0: [0]
        # Level 1: 2 children of 0
        # Level d: 2^d vertices (for a full binary tree)
        
        # But we want 3-regular, so each non-leaf has degree 3.
        # In a tree, leaves have degree 1, internal non-root have degree 3 (1 parent + 2 children),
        # root has degree 2 (2 children).
        
        # We'll build a tree of given depth, then add edges to make it 3-regular.
        
        # For girth >= g, we need depth >= g//2 roughly.
        # The tree has 1 + 2 + 4 + ... + 2^depth = 2^(depth+1) - 1 vertices.
        
        # Let's try different depths
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Choose depth based on girth
            # For girth g, we need the tree depth to be at least g//2
            # to avoid cycles shorter than g from the tree structure alone.
            # But when we close the boundary, we might create shorter cycles.
            
            depth = g // 2 + attempt  # Try increasing depths
            
            # Number of vertices in full binary tree of depth d: 2^(d+1) - 1
            # For g=16, depth=8: 2^9 - 1 = 511 vertices
            # depth=9: 2^10 - 1 = 1023 vertices
            
            n_tree = (1 << (depth + 1)) - 1
            
            if n_tree > 2000:  # Too large
                continue
            
            # Build the tree
            # Vertex i in level l: its children are 2*i+1 and 2*i+2
            # Parent of vertex i (if i > 0): (i-1)//2
            
            adj = [[] for _ in range(n_tree)]
            
            # Add tree edges
            for i in range(n_tree):
                left = 2 * i + 1
                right = 2 * i + 2
                if left < n_tree:
                    adj[i].append(left)
                    adj[left].append(i)
                if right < n_tree:
                    adj[i].append(right)
                    adj[right].append(i)
            
            # Now, leaves are at level depth.
            # In a full binary tree of depth d, leaves are vertices with indices
            # from 2^depth to 2^(depth+1) - 1.
            # Each leaf has degree 1 (only connected to parent).
            # We need to add edges to make all vertices degree 3.
            
            # Internal vertices (non-root, non-leaf) have degree 3 already.
            # Root has degree 2 (if depth >= 1).
            # Leaves have degree 1.
            
            # We need to add edges:
            # - Root needs 1 more edge
            # - Each leaf needs 2 more edges
            
            # Total edges to add: 1 + 2 * (number of leaves)
            # Number of leaves = 2^depth
            
            # But we can only add edges between vertices that don't already have an edge,
            # and we must maintain simplicity.
            
            # Strategy: 
            # 1. Connect root to one leaf (or some vertex).
            # 2. Pair up remaining leaves and add edges between pairs.
            
            leaves = list(range(1 << depth, n_tree))
            root = 0
            
            # Shuffle leaves for randomness
            rng_local.shuffle(leaves)
            
            # Connect root to first leaf
            if leaves:
                adj[root].append(leaves[0])
                adj[leaves[0]].append(root)
                leaves.remove(leaves[0])
            
            # Now pair up remaining leaves
            # Each pair gets an edge between them
            for i in range(0, len(leaves) - 1, 2):
                u, v = leaves[i], leaves[i + 1]
                if u != v and v not in adj[u]:
                    adj[u].append(v)
                    adj[v].append(u)
            
            # If odd number of leaves left, we have a problem.
            # But 2^depth - 1 leaves after removing one for root.
            # 2^depth is even, so 2^depth - 1 is odd.
            # So we'll have one unpaired leaf.
            
            # This approach has issues. Let me try a different strategy.
            
            # Alternative: Use a different construction.
            # Let's use the LUW construction as fallback and try random 3-regular graphs.
            
            break  # Skip to next approach
        
        # Fallback: try random 3-regular graphs
        for n_try in range(512, 1200, 32):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Generate random 3-regular graph using configuration model
                # Then check if it's simple and has girth >= g
                adj = _random_3_regular(n_try, rng)
                if adj is None:
                    continue
                
                if check_girth(adj, n_try, g):
                    edges = []
                    for u in range(n_try):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n_try, edges
        
        return None

    def _random_3_regular(n, rng):
        """Generate a random 3-regular graph using configuration model."""
        # Create half-edges: each vertex has 3 stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        # Randomly pair up stubs
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:  # Loop
                return None
            if v in adj[u]:  # Multi-edge
                return None
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        return adj

    def seed_construct(k, g):
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

    # Main: try random 3-regular graphs first, then fall back to LUW
    rng = random.Random(42)
    
    # Try various sizes, starting from Moore bound
    for n_try in range(512, 1500, 16):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random graphs of this size
        max_attempts = 50 if n_try < 700 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            adj = _random_3_regular(n_try, rng)
            if adj is None:
                continue
            
            if check_girth(adj, n_try, g):
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_try, edges
    
    # Fallback to LUW
    return seed_construct(k, g)
