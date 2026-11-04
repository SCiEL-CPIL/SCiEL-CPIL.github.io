import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Moore bound for k=3, g=16 is 766
    # We search for n starting from the Moore bound
    # If k != 3 or g != 16, we might need a general fallback, 
    # but the task is specific to this case likely.
    # However, the interface takes k and g.
    
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters to be safe
        # But the task is "Small 3-regular graph of girth 16"
        # So we assume k=3, g=16.
        pass
        
    moore_bound = 1 + k * sum((k-1)**i for i in range((g-1)//2 + 1))
    
    best_n = None
    best_edges = None
    
    # Search n from moore_bound upwards
    # We can set a max_n to avoid infinite loop if no graph exists (unlikely for large n)
    # But for girth 16, it exists.
    
    for n in range(moore_bound, 5000):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this n
        found = False
        # Number of attempts per n
        # For small n, probability might be low. For large n, high.
        attempts = 0
        max_attempts = 1000 if n < 1000 else 100
        
        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
                
            edges = generate_random_3_regular(n)
            if edges is None:
                continue
                
            if check_girth(edges, n, g):
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            break
            
    if best_n is None:
        # If we didn't find one, return the LUW graph as fallback
        # Re-implement LUW here or just return a large graph?
        # Better to return the LUW graph from the seed if possible, 
        # but we are rewriting.
        # Let's just return a valid LUW graph if search fails.
        return construct_luw(k, g)
        
    return best_n, best_edges

def generate_random_3_regular(n):
    # Configuration model
    # Stubs
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
        
    random.shuffle(stubs)
    
    edges = []
    used = [False] * n # Not needed, we just check pairs
    
    # Pair stubs
    # stubs[0] with stubs[1], stubs[2] with stubs[3], etc.
    # This creates a multigraph. We need to check for self-loops and multi-edges.
    
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Self-loop
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return None # Multi-edge
        edge_set.add((u, v))
        edges.append((u, v))
        
    return edges

def check_girth(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check if girth >= g
    # For each vertex, BFS up to depth g//2 - 1
    # If we find a cycle shorter than g, return False
    
    half_g = g // 2
    # We need to ensure no cycle of length < g
    # A cycle of length L < g passing through v implies 
    # a path of length L-1 between two neighbors of v?
    # Or simpler: BFS from v. If we reach a neighbor of v at depth d < g-1, 
    # then we have a cycle of length d+1 < g.
    # So we need to ensure no neighbor is reached at depth < g-1.
    # Max depth to check: g-2.
    
    for v in range(n):
        # BFS from v
        dist = [-1] * n
        parent = [-1] * n
        dist[v] = 0
        queue = [v]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if dist[u] >= g - 1:
                continue
                
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                else:
                    # Found a back edge
                    # Cycle length is dist[u] + dist[w] + 1
                    # But this logic is for finding *the* girth.
                    # We just need to know if girth < g.
                    # If we encounter a visited node w (w != parent[u]),
                    # we have a cycle.
                    if w != parent[u]:
                        # The cycle length is dist[u] + dist[w] + 1
                        # Wait, dist[w] is the distance from v to w.
                        # dist[u] is distance from v to u.
                        # The path is v -> ... -> w -> ... -> u -> w? No.
                        # The edge is (u, w).
                        # Path v -> ... -> u (len dist[u])
                        # Path v -> ... -> w (len dist[w])
                        # Cycle: u -> w (1) + path w->v + path v->u?
                        # No, the cycle is formed by the tree paths to u and w and the edge (u,w).
                        # Length = dist[u] + dist[w] + 1.
                        # If this length < g, we have a short cycle.
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < g:
                            return False
                            
    return True

def construct_luw(k, g):
    # Fallback LUW implementation
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
