import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin

    # Calculate Moore Bound
    # If g is even: 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # If g is odd: 2 * sum_{i=0}^{(g-1)/2} (k-1)^i
    if g % 2 == 0:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range((g+1)//2))
        
    min_n = moore_bound
    if (min_n * k) % 2 != 0:
        min_n += 1

    # 1. Fallback: Lazebnik-Ustimenko-Woldar (LUW) construction
    # This ensures we always return a valid graph, even if not optimal.
    n_luw, edges_luw = _luw_construct(k, g)
    
    # If LUW is close to the Moore bound or smaller than our target search space, return it.
    # For k=4, g=10, LUW is 320. Moore bound is 161.
    # We want to search [min_n, n_luw - 1] for a better solution.
    
    best_result = (n_luw, edges_luw)
    
    # 2. Local Search for smaller n
    # We iterate n from min_n up to n_luw - 1.
    # To save time, we only try a subset of n values or limit the time per n.
    # Given the gap (161 to 319), trying every n with a heavy search is risky.
    # However, for k=4, the search space is manageable if we are efficient.
    # Let's try to find *any* valid graph with n < n_luw.
    
    # Optimization: Skip n if n*k is odd.
    # Also, we can skip n if n is too small to support the girth (Moore bound covers this).
    
    current_n = min_n
    # We will try to find a graph. If we find one, return it immediately (since n is increasing).
    # If we don't find one within the time limit, return the LUW result.
    
    while current_n < n_luw:
        if time.time() > deadline - 5:
            break
            
        # Check parity
        if (current_n * k) % 2 != 0:
            current_n += 1
            continue
            
        # Try to find a valid graph of size current_n
        # Time limit for this specific n: 2 seconds? 
        # If we have 100s and 158 candidates, 0.6s each is tight.
        # Let's use a dynamic budget: if we have lots of time left, spend more per n.
        # If we are close to deadline, spend less.
        
        t_start = time.time()
        t_limit = 2.0
        if time.time() + t_limit > deadline - 2:
             t_limit = max(0.1, deadline - 2 - time.time())
             
        result = _local_search_construct(k, g, current_n, t_start, t_limit)
        
        if result is not None:
            n_found, edges_found = result
            # This is the smallest n found so far (since we iterate upwards)
            return n_found, edges_found
            
        current_n += 1
        
    return n_luw, edges_luw


def _luw_construct(k, g):
    # Original LUW code from the prompt
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


def _local_search_construct(k, g, n, t_start, t_limit):
    """
    Try to construct a k-regular graph of size n with girth >= g.
    Returns (n, edges) if successful, None otherwise.
    """
    # 1. Generate initial k-regular graph
    adj = _generate_random_k_regular(n, k)
    if adj is None:
        return None
        
    # 2. Local Search
    # We want to maximize girth. 
    # Heuristic: Randomly swap edges to break short cycles.
    # We check girth periodically.
    
    # To speed up girth check, we can keep a list of "bad" edges or just re-check.
    # Re-checking is O(n^2) roughly, which is fine for n~300.
    
    best_adj = None
    best_girth = 0
    
    # We don't know the girth initially, but we can estimate or just check.
    # Let's just check girth every few steps.
    
    steps = 0
    while time.time() < t_start + t_limit:
        # Check girth
        current_girth = _get_girth(adj, n, g) # Returns girth or g if >= g
        
        if current_girth >= g:
            # Found a valid graph
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # Update best if we found a longer girth (not strictly necessary if we just want >= g)
        # But it helps if we get stuck.
        
        # Perform a random edge swap
        # Pick two random edges (u1, v1) and (u2, v2) that are disjoint.
        # Swap to (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2).
        
        # To pick random edges efficiently, we can maintain a list of edges.
        # But rebuilding the list every step is O(n*k).
        # For n=300, k=4, edges=600. Rebuilding is fast.
        
        edges_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges_list.append((u, v))
        
        if len(edges_list) < 2:
            continue
            
        # Pick two random distinct edges
        idx1 = random.randint(0, len(edges_list)-1)
        idx2 = random.randint(0, len(edges_list)-1)
        if idx1 == idx2:
            continue
            
        u1, v1 = edges_list[idx1]
        u2, v2 = edges_list[idx2]
        
        # Check if edges are disjoint
        if len({u1, v1, u2, v2}) < 4:
            continue
            
        # Try swap: (u1, u2) and (v1, v2)
        # Check if new edges exist
        if u2 in adj[u1] or v2 in adj[v1]:
            # Try other swap: (u1, v2) and (v1, u2)
            if v2 in adj[u1] or u2 in adj[v1]:
                continue
            else:
                # Perform swap 2
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].add(v2)
                adj[v2].add(u1)
                adj[v1].add(u2)
                adj[u2].add(v1)
        else:
            # Perform swap 1
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            
        steps += 1
        
    return None


def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices."""
    # Configuration model
    # 1. Create a list of n*k stubs.
    # 2. Shuffle.
    # 3. Pair up.
    # 4. Check for loops and multi-edges.
    
    # To avoid infinite loops on failure, we limit retries.
    for _ in range(50):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edge_set or (v, u) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            edges.append((u, v))
        
        if valid:
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj
            
    return None


def _get_girth(adj, n, g_limit):
    """
    Estimate the girth of the graph.
    Returns the girth if found < g_limit, otherwise returns g_limit.
    Uses BFS from each vertex to find shortest cycles.
    """
    # Optimization: We only care if girth < g_limit.
    # So we can stop BFS at depth g_limit // 2.
    
    min_cycle = g_limit
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= g_limit // 2:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < g_limit:
                            return min_cycle
                            
    return min_cycle
