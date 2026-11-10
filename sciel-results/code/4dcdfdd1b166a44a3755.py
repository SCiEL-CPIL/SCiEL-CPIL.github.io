import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback to LUW if search fails
    def luw_construct(k, g):
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

    # Moore bound calculation
    # For odd girth g = 2r + 1, Moore bound M(k, g) = k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = k * sum((k - 1) ** i for i in range(r))
    
    # Known best upper bound for (5, 9) is 1116. 
    # We will try to find something smaller.
    # Start from Moore bound, go up.
    
    deadline = time.time() + 140  # 140s limit
    
    best_n = None
    best_edges = None
    
    # We can start from a reasonable guess, e.g., 500, or moore_bound
    # Let's iterate n from max(moore_bound, 428) upwards
    start_n = max(moore_bound, 428)
    
    # To avoid getting stuck on very small n where no solution exists, 
    # we give each n a small time slice.
    # If we find a solution, we return it immediately.
    
    for n in range(start_n, 1500):
        if time.time() > deadline - 10:
            break
            
        # Try to find a graph of size n
        # Number of restarts depends on time left
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate random regular graph
            edges = generate_random_regular_graph(n, k)
            if edges is None:
                continue
                
            # Check girth
            girth = calculate_girth(n, edges, k)
            
            if girth >= g:
                return n, edges
            
            # Local search
            # We want to increase girth
            # Time limit for local search per restart
            local_deadline = time.time() + 1.0 # 1 second per restart
            
            while time.time() < local_deadline:
                girth = calculate_girth(n, edges, k)
                if girth >= g:
                    return n, edges
                
                # Perform a move
                edges = improve_girth(n, edges, k, g)
                
                if edges is None:
                    break # Stuck?
        
        # If we didn't find a graph for this n, try larger n
        # We don't save best_n here because we want the SMALLEST n.
        # If we fail at n, we go to n+1.
        
    # If we didn't find anything small, fallback to LUW
    # Or return the best we found? 
    # The loop returns immediately if found. 
    # So if we exit the loop, we found nothing.
    # Let's try one more thing: if we didn't find anything, return LUW.
    return luw_construct(k, g)

def generate_random_regular_graph(n, k):
    """
    Generates a random k-regular graph on n vertices.
    Returns list of edges or None if impossible (e.g., n*k odd).
    """
    if n * k % 2 != 0:
        return None
    
    # Use the configuration model
    # Create a list of half-edges
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    random.shuffle(half_edges)
    
    edges = set()
    for i in range(0, n * k, 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u == v:
            return None # Loop found, retry (handled by caller restart)
        if (min(u, v), max(u, v)) in edges:
            return None # Multi-edge found, retry
        edges.add((min(u, v), max(u, v)))
        
    return list(edges)

def calculate_girth(n, edges, k):
    """
    Calculates the girth of the graph.
    Returns infinity if no cycle (tree, impossible for regular k>1)
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
        # BFS
        # dist[v] = (distance, parent)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        found_cycle = False
        while q:
            u = q.popleft()
            if dist[u] * 2 >= min_girth:
                continue # No need to explore deeper if we already have a smaller cycle
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    found_cycle = True
                    # We can break early if min_girth is 3? 
                    # No, we want the exact girth to know if it's >= g.
                    # But for the search, we just need to know if it's < g.
                    # Actually, calculate_girth is used to check success.
                    # If min_girth >= g, we can stop? 
                    # No, we need to be sure.
                    # But if we find a cycle of length L, and L < g, we fail.
                    # If we explore all nodes and min_girth >= g, we succeed.
                    
        # Optimization: If min_girth >= g, we can return early?
        # No, we must check all starts to be sure there isn't a shorter one.
        # But if we just want to check if girth >= g, we can stop BFS at depth g//2.
        # Let's optimize calculate_girth to take a target_girth and return True if girth >= target.
        
    return min_girth

def has_girth_at_least(n, edges, k, g):
    """
    Returns True if girth >= g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex, stop at depth g//2
    # If we find a back-edge to a node at depth < g//2 - 1, we have a cycle < g
    # Actually, if we find a cycle of length L, we need L >= g.
    # A cycle of length L is found when we visit a node v from u, and v is already visited.
    # dist[u] + 1 + dist[v] = L? No, dist[u] + 1 is the path from start to u.
    # The cycle is start -> ... -> u -> v -> ... -> start.
    # Length = dist[u] + 1 + (dist[start] via v)? No.
    # In BFS tree, if (u, v) is an edge and v is visited:
    # If v is parent of u, it's the tree edge.
    # Otherwise, it's a cross edge.
    # Cycle length = dist[u] + dist[v] + 1.
    # We want to ensure no cycle < g.
    # So if dist[u] + dist[v] + 1 < g, return False.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            # If dist[u] >= g//2, we don't need to expand further
            # because any cycle found will be at least dist[u] + 1 + 0 = g//2 + 1?
            # If g is odd, g=2r+1. Max dist to check is r.
            # If dist[u] >= r, then dist[u] + dist[v] + 1 >= r + 0 + 1 = r+1.
            # This doesn't guarantee >= 2r+1.
            # We need dist[u] + dist[v] + 1 >= g.
            # Worst case dist[v]=0. So we need dist[u] + 1 >= g => dist[u] >= g-1.
            # This is too deep.
            
            # Better: Just run full BFS but stop if we find a short cycle.
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def improve_girth(n, edges, k, g):
    """
    Attempts to increase girth by performing a 2-opt swap.
    Returns new edges list or None if no valid move found.
    """
    edge_set = set(edges)
    
    # Find a short cycle
    # We can just pick a random edge and check if it's in a short cycle?
    # Or find the shortest cycle.
    # For simplicity, let's pick a random edge (u, v).
    # Then try to swap it.
    
    if not edges:
        return None
        
    # Pick a random edge
    u, v = random.choice(edges)
    
    # Try to find a swap partner (x, y)
    # We want to remove (u, v) and (x, y) and add (u, x) and (v, y)
    # Constraints:
    # 1. (u, x) not in edge_set
    # 2. (v, y) not in edge_set
    # 3. u != x, v != y (no loops)
    # 4. x != y (not the same edge, which is true if we pick distinct edges)
    # 5. The swap should ideally break a short cycle.
    
    # Heuristic: Pick x as a random neighbor of a vertex NOT in the short cycle?
    # Actually, just random x, y might work with many attempts.
    
    # Let's try a few random swaps
    for _ in range(100):
        # Pick a random vertex x
        x = random.randint(0, n - 1)
        if x == u or x == v:
            continue
            
        # Pick a random vertex y
        y = random.randint(0, n - 1)
        if y == u or y == v or y == x:
            continue
            
        # Check if (u, x) and (v, y) are valid
        if (min(u, x), max(u, x)) in edge_set:
            continue
        if (min(v, y), max(v, y)) in edge_set:
            continue
            
        # Also ensure we are not creating a loop in the new edges
        # u != x, v != y already checked.
        
        # Perform swap
        new_edges = [e for e in edges if e != (u, v)]
        # Find (x, y) in edges? 
        # Wait, we need to remove (x, y) too.
        # But (x, y) might not be an edge!
        # The 2-opt move requires (x, y) to be an edge.
        
        # So we must pick (x, y) from the existing edges.
        pass
        
    # Correct 2-opt:
    # Pick two distinct edges (u, v) and (x, y).
    # Remove them.
    # Add (u, x) and (v, y).
    
    # Pick first edge (u, v)
    u, v = random.choice(edges)
    
    # Pick second edge (x, y)
    # It should be distinct from (u, v)
    candidates = [e for e in edges if e != (u, v)]
    if not candidates:
        return None
        
    x, y = random.choice(candidates)
    
    # Try orientations: (u, x) and (v, y) OR (u, y) and (v, x)
    
    for new_u, new_v, new_x, new_y in [(u, x, v, y), (u, y, v, x)]:
        # Check validity
        if new_u == new_v or new_x == new_y:
            continue
        if (min(new_u, new_v), max(new_u, new_v)) in edge_set:
            continue
        if (min(new_x, new_y), max(new_x, new_y)) in edge_set:
            continue
            
        # Construct new edge list
        # Remove (u, v) and (x, y)
        # Add (new_u, new_v) and (new_x, new_y)
        
        new_edges = []
        for e in edges:
            if e == (u, v) or e == (x, y):
                continue
            new_edges.append(e)
            
        new_edges.append((min(new_u, new_v), max(new_u, new_v)))
        new_edges.append((min(new_x, new_y), max(new_x, new_y)))
        
        return new_edges
        
    return None

# We need to wrap the main logic to handle the deadline properly.
# The current construct function does this.
# However, `improve_girth` is slow if it doesn't find a valid move quickly.
# We should add a check in `improve_girth` to return None quickly if no move is found.

# One issue: `generate_random_regular_graph` might fail often for small n.
# That's fine, we restart.

# Let's refine `construct` to be more robust.

def construct(k, g):
    def luw_construct(k, g):
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

    r = (g - 1) // 2
    moore_bound = k * sum((k - 1) ** i for i in range(r))
    
    deadline = time.time() + 140
    
    start_n = max(moore_bound, 428)
    
    # We will try n from start_n to 1500
    # If we find a valid graph, we return it.
    
    for n in range(start_n, 1500):
        if time.time() > deadline - 10:
            break
            
        # Give this n a time slice
        # If n is close to Moore bound, it's harder. Give it more time?
        # Or just fixed time per n.
        time_slice = 2.0
        
        n_deadline = time.time() + time_slice
        
        found = False
        while time.time() < n_deadline:
            edges = generate_random_regular_graph(n, k)
            if edges is None:
                continue
                
            if has_girth_at_least(n, edges, k, g):
                return n, edges
                
            # Local search
            local_deadline = time.time() + 0.5
            while time.time() < local_deadline:
                if has_girth_at_least(n, edges, k, g):
                    return n, edges
                edges = improve_girth(n, edges, k, g)
                if edges is None:
                    break
                    
    return luw_construct(k, g)

# This looks reasonable.
# One potential issue: `improve_girth` might get stuck in a local minimum.
# But with random restarts, it should be okay.
# Also, `has_girth_at_least` is O(n*m). For n=1000, m=2500, this is 2.5M ops.
# In Python, this might be slow.
# We should optimize `has_girth_at_least`.

def has_girth_at_least(n, edges, k, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

# This is still O(n^2) roughly.
# For n=1000, 1000 * 1000 = 1M. It's fine.

# Let's double check the LUW fallback.
# The current program uses it. It's a valid graph.
# If the random search fails, we return it.
# The score is based on n. LUW n is likely > 1116?
# For k=5, g=9, d=4.
# q=5.
# Number of vertices is 2 * q * q^d ? No.
# LUW graph size is 2 * q * q^{d-1} ?
# Let's check the code.
# `order` grows.
# It's a bipartite graph.
# The size is likely manageable.

# Final check on `improve_girth`.
# It picks random edges. It might not break the cycle.
# But it changes the graph.
# With many iterations, it should eventually find a better girth.

# One improvement: In `improve_girth`, we can check if the swap actually increases girth or at least doesn't decrease it.
# But `has_girth_at_least` is called after the swap in the loop.
# So we just swap and re-check.

# The code seems complete.

