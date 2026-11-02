import time
import random
import sys
from collections import deque

def _compute_girth(adjacency, k, g_max):
    """Compute the girth of the graph. Returns girth or g_max if no cycle shorter than g_max found."""
    n = len(adjacency)
    min_girth = g_max
    
    for start in range(n):
        # BFS with non-backtracking
        # visited[(vertex, prev)] = depth
        visited = {}
        queue = deque()
        
        for neighbor in adjacency[start]:
            visited[(neighbor, start)] = 1
            queue.append((neighbor, start, 1))
        
        while queue:
            vertex, prev, depth = queue.popleft()
            
            # Check if we can close a cycle
            # If vertex is adjacent to start, we have a cycle of length depth + 1
            if start in adjacency[vertex] and depth + 1 < min_girth:
                min_girth = depth + 1
                if min_girth <= g_max:
                    return min_girth
            
            for neighbor in adjacency[vertex]:
                if neighbor == prev:
                    continue
                state = (neighbor, vertex)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, vertex, depth + 1))
    
    return min_girth

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _backtracking_construct(k, g, n, time_limit=10.0):
    """
    Construct a k-regular graph of girth >= g on n vertices using randomized backtracking.
    Uses limited BFS to check for short cycles.
    """
    start_time = time.time()
    
    # adjacency list
    adj = [set() for _ in range(n)]
    
    # degrees
    degrees = [0] * n
    
    # We build the graph by adding vertices one by one.
    # To ensure regularity, we must be careful.
    # A simpler approach: 
    # 1. Create a list of all required "stubs" (n * k / 2 edges).
    # 2. This is hard to do with backtracking directly.
    
    # Alternative: 
    # We assign edges for vertex v to vertices u < v.
    # This doesn't guarantee regularity easily.
    
    # Better: 
    # Use the "configuration model" but with backtracking on the pairing?
    # Too complex.
    
    # Let's use a "Greedy with Backtracking" on a pre-defined edge order?
    # No, let's use the approach from Inspiration 2 but optimized.
    
    # We maintain a list of "active" vertices that need more edges.
    # However, ensuring the graph remains simple and girth-constrained is the key.
    
    # Let's try a different structure: 
    # We decide the neighbors of vertex i from {0, ..., i-1} and {i+1, ..., n-1}.
    # This is symmetric.
    
    # Let's stick to the "add edge (u,v)" model.
    # We iterate through a random permutation of all possible edges (u,v) with u<v.
    # We maintain degrees.
    # We maintain a "can_add" check.
    
    # This is essentially a random greedy with backtracking.
    
    # To make it feasible, we restrict the search space.
    # For n=500, number of edges is 1250.
    # Possible edges is ~125000.
    
    # We will use a recursive search that picks the "next" vertex to complete.
    
    # State: 
    # - adj: list of sets
    # - degrees: list of ints
    
    # We will try to complete vertex n-1, then n-2, etc.
    # This is not efficient.
    
    # Let's use a simpler heuristic:
    # 1. Shuffle vertices.
    # 2. For each vertex v, pick k neighbors from unvisited vertices.
    # 3. Check girth constraint.
    # 4. If fail, backtrack.
    
    # This is effectively the same as the greedy in Insp 2.
    # The bottleneck in Insp 2 was `has_short_cycle`.
    # We optimize `has_short_cycle` using a precomputed "reachability" matrix for small depths?
    # Or just fast BFS.
    
    # Let's implement the fast BFS check.
    
    def has_short_cycle(u, v, max_depth):
        """
        Returns True if there is a path of length < max_depth between u and v 
        in the current graph (excluding the edge u-v being added).
        If we add u-v, and there is a path of length L < max_depth, 
        we form a cycle of length L+1.
        We require cycle >= g.
        So we require L+1 >= g => L >= g-1.
        So we reject if L < g-1.
        So we check if there is a path of length <= g-2.
        """
        if u == v:
            return True
        target_dist = g - 2
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= target_dist:
                continue
            for neighbor in adj[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    # We will use a recursive backtracking that builds the graph vertex by vertex.
    # This is complex to implement correctly with regularity constraints.
    
    # Instead, let's use a "Local Search" on a random regular graph, 
    # but with a much better acceptance criterion: 
    # Accept if it reduces the number of short cycles.
    
    # This is the "Edge Switching" from the current program, but improved.
    
    # The current program's edge switching was too slow.
    # We can speed it up by:
    # 1. Maintaining a list of "bad edges" (edges in short cycles).
    # 2. Only switching edges that are in bad cycles.
    
    # However, identifying short cycles is expensive.
    
    # Let's try a different approach: **Simulated Annealing on the adjacency matrix**.
    # This is too slow in Python.
    
    # Let's go back to the **Backtracking** idea, but simplify the state.
    # We will try to construct the graph on n vertices.
    # We fix the order of vertices 0..n-1.
    # We decide the neighbors of vertex i from {0..i-1} and {i+1..n-1}.
    
    # This is still hard.
    
    # Let's use the **LUW construction** but with a smaller q?
    # No, q is fixed by k.
    
    # What if we use **Multiple LUW constructions** and take the smallest?
    # The LUW construction is deterministic.
    
    # Let's try to **optimize the LUW construction** by pruning the graph.
    # LUW graphs might have disconnected components?
    # The code returns the connected component of the start vertex.
    # If the graph is not connected, we only get a part of it.
    # But it must be k-regular. If we take a component, it is k-regular.
    # Is the LUW graph connected?
    # Usually yes for small d.
    
    # If LUW gives n=6250, we are stuck.
    
    # Let's try **Cayley Graphs** again.
    # For k=5, we can use the group $A_5$ (order 60) or similar?
    # No, we need 5-regular.
    
    # Let's try a **Randomized Greedy** with **Restarts**.
    # The greedy in Insp 2 failed because it didn't restart effectively.
    # We will run the greedy many times with different seeds.
    
    def _greedy_attempt(n, seed):
        random.seed(seed)
        adj_local = [set() for _ in range(n)]
        degrees = [0] * n
        active = list(range(n))
        random.shuffle(active)
        
        # We need to add k*n/2 edges.
        # We process vertices in random order.
        # When processing u, we pick k neighbors from active vertices (including u? no, u is active).
        # Actually, we should pick neighbors from the whole set, but avoid duplicates.
        
        # A better greedy:
        # 1. Create list of all possible edges.
        # 2. Shuffle them.
        # 3. Iterate. If adding edge keeps degrees < k and no short cycle, add it.
        # 4. At the end, check if all degrees are k.
        
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if degrees[u] >= k or degrees[v] >= k:
                continue
            # Check short cycle
            if has_short_cycle(u, v, g - 2):
                continue
            # Add edge
            adj_local[u].add(v)
            adj_local[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
            if all(d == k for d in degrees):
                # Check if girth is actually >= g
                # The greedy check should ensure this, but verify.
                # For speed, we assume it is.
                return adj_local
        
        return None

    # Try a few attempts
    for i in range(10):
        if time.time() - start_time > time_limit:
            break
        adj_res = _greedy_attempt(n, i)
        if adj_res:
            # Verify
            if all(len(adj_res[i]) == k for i in range(n)):
                # Verify girth
                if _compute_girth(adj_res, k, g) >= g:
                    edges = []
                    for i in range(n):
                        for j in adj_res[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    # Try backtracking/greedy for smaller n
    # Moore bound for k=5, g=9 is ~426.
    # Try n in [450, 1500]
    
    if k == 5 and g == 9:
        target_ns = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1500]
        
        for n in target_ns:
            if time.time() - start_time > time_limit:
                break
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5.0:
                break
            
            # Allocate more time to smaller n as they are harder?
            # Or just fixed time.
            t_alloc = min(10.0, remaining_time / 2)
            
            result = _backtracking_construct(k, g, n, time_limit=t_alloc)
            if result is not None:
                return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
