import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth g using backtracking search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Symmetry breaking setup
    # We build the graph incrementally.
    # adjacency list
    adj = []
    
    # Helper to add a vertex
    def add_vertex():
        adj.append([])
        return len(adj) - 1
    
    # Helper to add an edge
    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
        
    # Helper to remove an edge
    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        
    # Check if adding edge (u, v) creates a cycle shorter than g
    # We need to check if there is a path of length < g-1 between u and v
    def has_short_path(u, v, max_depth):
        # BFS from u
        if u == v:
            return True # Loop, but we don't add loops
        visited = {u: 0}
        q = deque([(u, 0)])
        while q:
            curr, dist = q.popleft()
            if dist >= max_depth:
                continue
            for nei in adj[curr]:
                if nei == v:
                    return True
                if nei not in visited:
                    visited[nei] = dist + 1
                    q.append((nei, dist + 1))
        return False

    # Check if vertex u can be connected to v
    def can_connect(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        if len(adj[u]) >= k:
            return False
        if len(adj[v]) >= k:
            return False
        # Girth check
        if has_short_path(u, v, g - 2):
            return False
        return True

    # State for backtracking
    # We will maintain a list of vertices and their degrees implicitly via adj
    # We need to track which vertices are "active" or have deficit
    
    # Initialize with symmetry breaking
    # 0 connected to 1, 2, 3
    # 1 connected to 0, 4, 5
    # 2 connected to 0, 6, 7
    # This assumes k=3. If k != 3, this specific breaking might fail or need adjustment.
    # The task says k=3, g=16. But to be safe, let's handle general k if possible, 
    # but optimize for k=3.
    
    if k != 3:
        # Fallback to LUW if k != 3 to ensure validity
        return _luw_construct(k, g)
        
    # Specific k=3, g=16 solver
    
    # Vertices 0, 1, 2, 3, 4, 5, 6, 7
    for _ in range(8):
        add_vertex()
        
    # Add fixed edges
    # 0: 1, 2, 3
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(0, 3)
    
    # 1: 0, 4, 5
    add_edge(1, 4)
    add_edge(1, 5)
    
    # 2: 0, 6, 7
    add_edge(2, 6)
    add_edge(2, 7)
    
    # Vertex 3 needs 2 more neighbors.
    # Vertex 4 needs 2 more.
    # Vertex 5 needs 2 more.
    # Vertex 6 needs 2 more.
    # Vertex 7 needs 2 more.
    
    # We will use a recursive backtracking function.
    # To avoid recursion depth issues and allow for deadline checks, 
    # we can use a stack or just standard recursion with a global time check.
    
    best_n = float('inf')
    best_adj = None
    
    # Heuristic: pick vertex with max deficit, then min index
    def get_next_vertex():
        max_def = -1
        cand = -1
        for i in range(len(adj)):
            def_ = k - len(adj[i])
            if def_ > max_def:
                max_def = def_
                cand = i
        return cand, max_def

    # We store the current state in adj. 
    # We need to backtrack. We'll use a simple recursive function.
    # Depth is number of vertices.
    
    sys.setrecursionlimit(10000)
    
    def backtrack():
        nonlocal best_n, best_adj
        
        if time.time() > deadline:
            return False
            
        # Check if complete
        complete = True
        for i in range(len(adj)):
            if len(adj[i]) != k:
                complete = False
                break
                
        if complete:
            n = len(adj)
            if n < best_n:
                best_n = n
                # Copy adjacency list
                best_adj = [list(nei) for nei in adj]
            return True # Stop searching if we found a valid graph? 
            # Actually, we want the smallest. If we find one, we should try to find smaller.
            # But backtracking usually finds one and then continues to find others.
            # However, for cage problems, the first found is often close to optimal if symmetry is broken well.
            # To ensure we don't waste time, we can return True to stop this branch, 
            # but the caller might try other branches.
            # Let's return True to indicate success, but we might want to explore other branches 
            # if we haven't hit a lower bound.
            # For now, let's just record and return True to prune this path.
            
        # Get next vertex to satisfy
        u, def_ = get_next_vertex()
        
        if def_ <= 0:
            return False # Should not happen if not complete
        
        # Candidates for u:
        # 1. Existing vertices with deficit > 0
        # 2. New vertex
        
        candidates = []
        
        # Existing
        for v in range(len(adj)):
            if v != u and len(adj[v]) < k:
                if can_connect(u, v):
                    candidates.append(v)
                    
        # New vertex
        new_v = len(adj)
        # We don't check can_connect for new vertex yet, 
        # but we know it will be connected to u.
        # We need to ensure it doesn't create short cycles with u's neighbors?
        # has_short_path(u, new_v) will be false because new_v is isolated.
        # So new_v is always a valid candidate to connect to u.
        candidates.append(new_v)
        
        # Shuffle candidates? No, deterministic is better for reproducibility.
        # But randomization can help escape local minima in search.
        # Let's try deterministic first.
        
        for v in candidates:
            if time.time() > deadline:
                return False
                
            if v == new_v:
                # Add new vertex
                add_vertex()
                add_edge(u, v)
                
                if backtrack():
                    # Unwind
                    remove_edge(u, v)
                    adj.pop()
                    return True
                else:
                    # Unwind
                    remove_edge(u, v)
                    adj.pop()
            else:
                add_edge(u, v)
                
                if backtrack():
                    remove_edge(u, v)
                    return True
                else:
                    remove_edge(u, v)
                    
        return False

    # Start backtracking
    # We might need to try different symmetry breakings if the first one fails or is slow.
    # But let's try the standard one.
    
    backtrack()
    
    if best_adj is not None:
        n = best_n
        edges = []
        for i in range(n):
            for j in best_adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    else:
        # Fallback to LUW if backtracking fails to find anything
        return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW implementation as fallback
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
