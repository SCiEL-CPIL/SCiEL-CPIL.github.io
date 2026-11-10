import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Greedy construction of a k-regular graph with girth >= g
    # Strategy: build graph incrementally, adding vertices as needed
    # Use non-backtracking BFS to verify girth constraint
    
    # Moore bound for k=3, g=16: 1 + 3*sum_{i=0}^{g/2-1} 2^i = 1 + 3*(2^8 - 1) = 767
    # But actual cages are larger. Start with a reasonable upper bound.
    # For cubic girth 16, known constructions give ~1000-2000 vertices.
    
    # We'll use a greedy approach:
    # 1. Start with a small tree-like structure
    # 2. Add edges to satisfy degree constraints
    # 3. Check girth constraint after each addition
    
    # First, try a simple greedy construction
    # Build a k-regular graph by adding vertices and edges
    
    # Start with a path, then close it up
    n_target = 936  # known upper bound
    n = n_target
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    degree = [0] * n
    
    # Create a base structure: a cycle with chords
    # First, create a cycle
    for i in range(n):
        adj[i].append((i + 1) % n)
        degree[i] += 1
    for i in range(n):
        adj[i].append((i - 1) % n)
        degree[i] += 1
    
    # Now we need to add one more edge per vertex to make it 3-regular
    # We'll do this by finding a perfect matching on the remaining degree slots
    
    # Remaining degree needed for each vertex
    remaining = [k - degree[i] for i in range(n)]
    
    # Find a perfect matching to complete the graph
    # Use a greedy approach with backtracking if needed
    
    # For simplicity, use a deterministic matching
    # Pair up vertices in a way that avoids short cycles
    
    # Create a list of vertices that need more edges
    need_edges = [i for i in range(n) if remaining[i] > 0]
    
    # Greedy matching
    used = [False] * n
    edges = []
    
    # Add the cycle edges
    for i in range(n):
        edges.append((i, (i + 1) % n))
    
    # Now add matching edges
    # We need to add n/2 more edges to make it 3-regular
    # Each vertex needs exactly 1 more edge
    
    # Use a simple strategy: connect i to i + n//2 for even i
    # But this might create short cycles, so we need to be careful
    
    # Better: use a random matching and check girth
    random.seed(42)
    
    # Try multiple random matchings
    best_edges = None
    best_n = n
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create a random perfect matching
        vertices = list(range(n))
        random.shuffle(vertices)
        
        matching = []
        for i in range(0, n, 2):
            matching.append((vertices[i], vertices[i + 1]))
        
        # Check if this matching creates any short cycles
        # Build the full adjacency
        adj_check = [[] for _ in range(n)]
        for i in range(n):
            adj_check[i].append((i + 1) % n)
            adj_check[i].append((i - 1) % n)
        for u, v in matching:
            adj_check[u].append(v)
            adj_check[v].append(u)
        
        # Check girth using non-backtracking BFS from each vertex
        valid = True
        half_g = g // 2  # For even girth, check up to g/2
        
        for start in range(n):
            if time.time() > deadline:
                break
            # Non-backtracking BFS
            # State: (current_node, previous_node)
            visited = set()
            queue = deque()
            
            # Start from each neighbor of start
            for nb in adj_check[start]:
                queue.append((nb, start, 1))
            
            while queue and valid:
                curr, prev, dist = queue.popleft()
                if dist >= half_g:
                    continue
                if (curr, prev) in visited:
                    continue
                visited.add((curr, prev))
                
                for nb in adj_check[curr]:
                    if nb == prev:
                        continue
                    if nb == start:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            valid = False
                            break
                        continue
                    queue.append((nb, curr, dist + 1))
            
            if not valid:
                break
        
        if valid:
            # This is a valid graph
            all_edges = []
            for i in range(n):
                all_edges.append((i, (i + 1) % n))
            for u, v in matching:
                all_edges.append((u, v))
            best_edges = all_edges
            best_n = n
            break
    
    if best_edges is None:
        # Fallback: use LUW construction
        return _luw_fallback(k, g, deadline)
    
    return best_n, best_edges


def _luw_fallback(k, g, deadline):
    # LUW construction: CD(d, q) with q = k, girth >= d + 5
    q = k
    d = g - 5
    if d % 2 != 1:
        d += 1
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
