import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    n_luw, edges_luw = _luw_construct(k, g)
    best_n, best_edges = n_luw, edges_luw
    
    # Moore Bound estimate
    if g % 2 == 0:
        # Even girth
        moore = 1 + k * sum([(k-1)**i for i in range(g//2 - 1)])
    else:
        # Odd girth
        moore = 1 + k * (1 + sum([(k-1)**i for i in range((g-3)//2)]))
    
    # We try to find a graph smaller than the LUW one
    # Start from a bit above the Moore bound
    # Note: Moore bound is a lower bound, actual cages are often larger
    
    # Heuristic: try sizes from max(moore, 2*k) up to n_luw - 1
    # Step size depends on k
    step = max(1, k // 2)
    
    # We will iterate through candidate sizes
    # To save time, we prioritize sizes that are "likely" to work.
    # Often, the optimal size is close to the Moore bound.
    
    candidates = []
    current_n = max(int(moore), 2 * k)
    while current_n < n_luw:
        candidates.append(current_n)
        current_n += step
    
    # Shuffle candidates slightly to avoid bias? Or just go up.
    # Going up is safer because smaller N is harder to satisfy girth.
    # But if we start too small, we waste time.
    # Let's try a few random sizes in the range [moore, n_luw]
    
    # Actually, a better strategy:
    # 1. Try the Moore bound + small offset.
    # 2. If fails, increase.
    
    # To maximize chance of beating LUW, we should try the smallest N first.
    
    for n_try in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Generate a random k-regular graph of size n_try
        edges = _random_regular_graph(n_try, k)
        if not edges:
            continue
            
        # Run local search
        valid, optimized_edges = _local_search(n_try, edges, k, g, deadline)
        
        if valid:
            if n_try < best_n:
                best_n = n_try
                best_edges = optimized_edges
                # If we found a valid graph, can we find a smaller one?
                # Yes, continue searching for smaller N? 
                # No, we are iterating N upwards. So this is the smallest N found so far.
                # We should break and return? 
                # Wait, the loop goes from small to large. 
                # So the first valid one is the smallest.
                # We can return immediately?
                # It might be worth trying to optimize the *current* graph further?
                # The local search already does that.
                # So we can return immediately.
                return best_n, best_edges

    return best_n, best_edges

def _luw_construct(k, g):
    """The original seed implementation."""
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

def _random_regular_graph(n, k):
    """Generates a random k-regular graph using the configuration model."""
    if n * k % 2 != 0:
        return []
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self-loop, retry or skip? For simplicity, if too many, return None
            # A better approach is to reject and restart, but that's expensive.
            # For now, we just return None if we hit a loop, assuming it's rare for large n
            return []
        if (v, u) in edges:
            return []
        edges.add((u, v))
        
    return list(edges)

def _local_search(n, edges, k, g, deadline):
    """
    Local search to eliminate cycles shorter than g.
    Returns (is_valid, optimized_edges)
    """
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    def count_short_cycles():
        """Counts number of cycles of length < g."""
        # For simplicity and speed, we just check if ANY cycle < g exists.
        # If we want to minimize count, we need to count them.
        # BFS from each node to depth g//2
        count = 0
        # Optimization: if g is large, this is expensive.
        # We'll just check for existence first.
        # Actually, for local search, we need a score.
        # Score = sum(1/len(cycle)) for all cycles len < g?
        # Or just count of shortest cycles?
        
        # Let's just find the shortest cycle length.
        min_len = g
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] * 2 >= min_len:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
                            if min_len == 3: # Triangle is smallest possible
                                return min_len, 1000 # High penalty
        return min_len, 0
        
    # Initial check
    min_len, _ = count_short_cycles()
    if min_len >= g:
        return True, edges
        
    # Local Search Loop
    # We want to reduce min_len.
    # Strategy: Find a cycle, then try to break it with an edge swap.
    
    # To speed up, we will only try a limited number of iterations or time.
    last_improvement = time.time()
    
    while time.time() < deadline - 2.0:
        # Find a shortest cycle
        # Re-implementing cycle finding to return the actual cycle
        cycle = _find_shortest_cycle(n, adj, g)
        if not cycle:
            return True, edges
            
        # Try to break the cycle
        # Pick a random edge (u, v) in the cycle
        # Pick a random edge (x, y) NOT in the cycle and NOT adjacent to u or v
        # Swap (u, v) and (x, y) with (u, x) and (v, y)
        
        swapped = False
        for _ in range(50): # Try 50 random swaps
            u, v = random.choice(cycle)
            # Find candidate x, y
            # x must not be u, v, or neighbor of u, v
            # y must not be x, u, v, or neighbor of u, v
            # And (u, x), (v, y) must not exist
            
            # Get neighbors of u and v
            nu = adj[u]
            nv = adj[v]
            forbidden = set(nu) | set(nv) | {u, v}
            
            # Pick random x
            x = random.randint(0, n - 1)
            if x in forbidden:
                continue
            if x in nu or x in nv:
                continue
            
            # Pick random y
            y = random.randint(0, n - 1)
            if y in forbidden or y == x:
                continue
            if y in nu or y in y in nv: # Typo in thought, check code
                continue
            if y in adj[x]:
                continue
                
            # Check if new edges exist
            if (min(u, x), max(u, x)) in edge_set:
                continue
            if (min(v, y), max(v, y)) in edge_set:
                continue
            if (min(u, y), max(u, y)) in edge_set:
                continue
            if (min(v, x), max(v, x)) in edge_set:
                continue
                
            # Perform swap
            # Remove (u, v) and (x, y)
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((min(u, v), max(u, v)))
            
            adj[x].remove(y)
            adj[y].remove(x)
            edge_set.remove((min(x, y), max(x, y)))
            
            # Add (u, x) and (v, y)
            adj[u].add(x)
            adj[x].add(u)
            edge_set.add((min(u, x), max(u, x)))
            
            adj[v].add(y)
            adj[y].add(v)
            edge_set.add((min(v, y), max(v, y)))
            
            # Check if improvement
            new_min_len, _ = count_short_cycles()
            if new_min_len > len(cycle) or (new_min_len == len(cycle) and random.random() < 0.1):
                # Accept
                swapped = True
                break
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                edge_set.remove((min(u, x), max(u, x)))
                
                adj[v].remove(y)
                adj[y].remove(v)
                edge_set.remove((min(v, y), max(v, y)))
                
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))
                
                adj[x].add(y)
                adj[y].add(x)
                edge_set.add((min(x, y), max(x, y)))
                
        if not swapped:
            # If we can't improve, break to avoid infinite loop
            # Or try a different cycle?
            # For now, break
            break
            
    final_min_len, _ = count_short_cycles()
    if final_min_len >= g:
        return True, list(edge_set)
    else:
        return False, list(edge_set)

def _find_shortest_cycle(n, adj, g):
    """Finds a cycle of length < g. Returns list of edges or None."""
    # BFS from each node
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] * 2 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Cycle found
                    cycle_edges = []
                    # Reconstruct path
                    # u -> ... -> start -> ... -> v
                    # Path u to start
                    path1 = []
                    curr = u
                    while curr != start:
                        p = parent[curr]
                        path1.append((curr, p))
                        curr = p
                    # Path v to start
                    path2 = []
                    curr = v
                    while curr != start:
                        p = parent[curr]
                        path2.append((curr, p))
                        curr = p
                    # Add edge (u, v)
                    cycle_edges.append((u, v))
                    cycle_edges.extend(path1)
                    cycle_edges.extend(path2)
                    return cycle_edges
    return None

# EVOLVE-BLOCK-END
