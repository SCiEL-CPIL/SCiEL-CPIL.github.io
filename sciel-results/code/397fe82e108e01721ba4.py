import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second buffer
    
    # Moore Bound calculation
    # For odd girth g = 2r + 1
    # N >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 1:
        r = (g - 1) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
        
    # Fallback to LUW if greedy fails or takes too long
    # Re-implementing a simplified LUW or just returning a valid but larger graph
    # For now, we will try greedy for N from moore_bound to moore_bound + 500
    
    best_n = None
    best_edges = None
    
    # Try a range of N
    # We know n(5,9) is 1116. Moore bound for 5,9 is 426.
    # We should try N close to known best if possible, but we don't know it.
    # So we scan upwards.
    
    # Optimization: If we find a solution, we can try to improve by restarting with slightly smaller N?
    # No, if greedy fails for N, it might fail for N+1 too if the structure is bad.
    # But smaller N is harder.
    
    # Let's try N from max(moore_bound, 100) to 1500
    # We will use multiple random seeds for each N
    
    for N in range(max(moore_bound, 100), 1500):
        if time.time() > deadline:
            break
            
        # Try a few seeds for this N
        for seed in range(5):
            if time.time() > deadline:
                break
                
            edges = greedy_construct(k, g, N, seed)
            if edges is not None:
                # Verify validity
                if is_valid(k, g, N, edges):
                    return N, edges

    # Fallback: Return a valid graph, even if large
    # We can use the LUW construction from the original code as a fallback
    # Or just return a cycle if k=2, but k is likely > 2
    # Let's implement a simple fallback: a large complete graph? No, girth constraint.
    # We will use the LUW logic from the prompt as a fallback.
    
    return luw_construct(k, g)

def greedy_construct(k, g, N, seed):
    random.seed(seed)
    adj = [set() for _ in range(N)]
    degrees = [0] * N
    
    # To speed up, we can track which vertices are "active" (degree < k)
    active = list(range(N))
    
    while active:
        # Pick a random active vertex
        idx = random.randint(0, len(active) - 1)
        u = active[idx]
        
        # Try to find a valid partner v
        found = False
        # Shuffle active list to randomize order
        random.shuffle(active)
        
        for v in active:
            if v == u or v in adj[u]:
                continue
            
            # Check girth
            if not creates_short_cycle(adj, u, v, g):
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                
                # Update active list
                if degrees[u] == k:
                    active.pop(idx)
                    # Note: v might also reach k, but we are iterating over a copy/shuffled list
                    # We need to remove v from active if it reaches k
                    # Since we are modifying active, it's tricky.
                    # Better to use a list of vertices with degree < k and rebuild it periodically
                    # Or just check degrees at the end.
                    
                if degrees[v] == k:
                    # Remove v from active if present
                    if v in active:
                        active.remove(v)
                        
                found = True
                break
        
        if not found:
            return None
            
    # Verify all degrees are k
    if all(d == k for d in degrees):
        edges = []
        for i in range(N):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges
    else:
        return None

def creates_short_cycle(adj, u, v, g):
    # Check if adding edge (u,v) creates a cycle of length < g
    # We do a simultaneous BFS from u and v
    # dist_u[w] = distance from u to w
    # dist_v[w] = distance from v to w
    
    dist_u = {u: 0}
    dist_v = {v: 0}
    
    queue_u = [u]
    queue_v = [v]
    
    # We only need to explore up to depth (g-1)//2 - 1?
    # If dist_u[w] + dist_v[w] + 1 < g, cycle exists.
    # Max dist needed: if g=9, max sum is 7. So dist 3 and 4 is enough.
    max_depth = (g - 1) // 2
    
    while queue_u or queue_v:
        if queue_u:
            curr = queue_u.pop(0)
            d_u = dist_u[curr]
            if d_u >= max_depth:
                continue
            for neighbor in adj[curr]:
                if neighbor == v:
                    continue # We are checking if adding (u,v) creates cycle, so (u,v) is not in adj yet
                if neighbor not in dist_u:
                    dist_u[neighbor] = d_u + 1
                    queue_u.append(neighbor)
                    
                    # Check if this neighbor is in dist_v
                    if neighbor in dist_v:
                        d_v = dist_v[neighbor]
                        if d_u + 1 + d_v + 1 < g: # +1 for edge to neighbor from u, +1 for edge from v to neighbor?
                            # Path: u -> ... -> neighbor <- ... <- v
                            # Length: dist_u[neighbor] + dist_v[neighbor] + 1 (edge u-v? No)
                            # Wait, the cycle is u -> ... -> neighbor -> ... -> v -> u
                            # Length: dist_u[neighbor] + dist_v[neighbor] + 1
                            # We want this >= g
                            pass
                        # Actually, the check is:
                        # If we add edge (u,v), a cycle exists if there is a path from u to v in the existing graph.
                        # The length of the shortest path from u to v in G is L.
                        # The new cycle has length L + 1.
                        # We need L + 1 >= g => L >= g - 1.
                        # So we just need to check if the distance between u and v in G is < g - 1.
        
        if queue_v:
            curr = queue_v.pop(0)
            d_v = dist_v[curr]
            if d_v >= max_depth:
                continue
            for neighbor in adj[curr]:
                if neighbor == u:
                    continue
                if neighbor not in dist_v:
                    dist_v[neighbor] = d_v + 1
                    queue_v.append(neighbor)
                    
    # The above BFS is a bit messy. Let's simplify.
    # Just compute shortest path distance between u and v in the current graph G.
    # If dist(u, v) < g - 1, then adding (u,v) creates a cycle of length dist(u,v) + 1 < g.
    
    # BFS from u to find v
    if u == v:
        return True # Loop, but we don't add loops
        
    visited = {u: 0}
    queue = [u]
    
    while queue:
        curr = queue.pop(0)
        d = visited[curr]
        if d >= g - 1:
            break
        for neighbor in adj[curr]:
            if neighbor == v:
                # Found v at distance d + 1
                if d + 1 < g - 1:
                    return True
                else:
                    return False
            if neighbor not in visited:
                visited[neighbor] = d + 1
                queue.append(neighbor)
                
    return False

def is_valid(k, g, N, edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(N):
        if len(adj[i]) != k:
            return False
            
    # Check girth
    # For each edge (u,v), check if there is a path from u to v of length < g-1
    # This is O(E * N * k) which might be slow.
    # But we only call this on the final candidate.
    
    for u in range(N):
        for v in adj[u]:
            if u > v:
                continue
            # BFS from u to v
            visited = {u: 0}
            queue = [u]
            found_short = False
            while queue:
                curr = queue.pop(0)
                d = visited[curr]
                if d >= g - 1:
                    break
                for neighbor in adj[curr]:
                    if neighbor == v:
                        if d + 1 < g - 1:
                            found_short = True
                        break
                    if neighbor not in visited:
                        visited[neighbor] = d + 1
                        queue.append(neighbor)
            if found_short:
                return False
                
    return True

def luw_construct(k, g):
    # Fallback from original code
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
