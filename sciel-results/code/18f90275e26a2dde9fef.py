"""Construct a 4-regular graph with girth >= 10 on as few vertices as possible."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _compute_girth(adj, n, max_girth=None):
    """Compute the girth of a graph using BFS from each vertex.
    Returns the length of the shortest cycle, or inf if no cycle found within max_girth."""
    if max_girth is None:
        max_girth = 2 * n
    girth = float('inf')
    
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
                        if girth <= 2:
                            return girth
    return girth


def _count_short_cycles(adj, n, max_len):
    """Count cycles of length <= max_len using non-backtracking walks."""
    count = 0
    for start in range(n):
        # Non-backtracking BFS
        # State: (current_vertex, previous_vertex)
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start))
            visited[(v, start)] = 1
        
        while queue:
            u, prev = queue.popleft()
            depth = visited[(u, prev)]
            if depth >= max_len:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start and depth + 1 >= 3:
                    count += 1
                elif (w, u) not in visited:
                    visited[(w, u)] = depth + 1
                    queue.append((w, u))
    return count


def _find_short_cycle(adj, n, max_len=10):
    """Find a specific short cycle (length <= max_len) using non-backtracking BFS.
    Returns a list of vertices forming the cycle, or None."""
    for start in range(n):
        # Non-backtracking BFS
        # State: (current_vertex, previous_vertex) -> (depth, path)
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start))
            visited[(v, start)] = (1, [start, v])
        
        while queue:
            u, prev = queue.popleft()
            depth, path = visited[(u, prev)]
            if depth >= max_len:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start and depth + 1 >= 3:
                    return path + [start]
                elif (w, u) not in visited:
                    visited[(w, u)] = (depth + 1, path + [w])
                    queue.append((w, u))
    return None


def _random_k_regular(n, k, rng):
    """Generate a random k-regular graph using configuration model, then fix multi-edges."""
    if n * k % 2 != 0:
        raise ValueError("n*k must be even")
    
    # Configuration model
    half_edges = list(range(n * k))
    for i in range(n):
        for j in range(k):
            half_edges[i * k + j] = i
    
    rng.shuffle(half_edges)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    # Fix multi-edges and self-loops by swapping
    for _ in range(1000):
        # Find problematic edges
        fixed = True
        for u in range(n):
            if len(adj[u]) != k:
                fixed = False
                break
            # Check for multi-edges (shouldn't happen with sets, but check connectivity)
        
        if fixed:
            break
        
        # Try to fix by finding a pair of edges to swap
        for u in range(n):
            if len(adj[u]) < k:
                # Need to add an edge from u
                candidates = [v for v in range(n) if v != u and v not in adj[u] and len(adj[v]) < k]
                if candidates:
                    v = rng.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    # Find an edge to break
                    for x in range(n):
                        if x != u and x in adj[u]:
                            adj[u].discard(x)
                            adj[x].discard(u)
                            # Find a new partner for x
                            candidates2 = [v for v in range(n) if v != x and v not in adj[x] and len(adj[v]) < k]
                            if candidates2:
                                v = rng.choice(candidates2)
                                adj[x].add(v)
                                adj[v].add(x)
                            break
            elif len(adj[u]) > k:
                # Remove an edge
                for x in list(adj[u]):
                    adj[u].discard(x)
                    adj[x].discard(u)
                    # Find a new partner for x
                    candidates = [v for v in range(n) if v != x and v not in adj[x] and len(adj[v]) < k]
                    if candidates:
                        v = rng.choice(candidates)
                        adj[x].add(v)
                        adj[v].add(x)
                    break
    
    # Convert to sorted lists
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    # Verify
    for i in range(n):
        if len(adj_list[i]) != k:
            raise ValueError(f"Vertex {i} has degree {len(adj_list[i])} instead of {k}")
    
    return adj_list


def _two_switch(adj, n, k, rng):
    """Apply a random 2-switch. Returns (new_adj, success).
    A 2-switch swaps edges (u,v) and (x,y) with (u,x) and (v,y) (or similar)."""
    # Find two non-adjacent edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if len(edges) < 2:
        return adj, False
    
    # Try several times to find a valid 2-switch
    for _ in range(50):
        e1, e2 = rng.sample(edges, 2)
        u, v = e1
        x, y = e2
        
        # Check they're distinct and non-adjacent
        if len({u, v, x, y}) < 4:
            continue
        
        # Try swap: (u,v) and (x,y) -> (u,x) and (v,y)
        if x not in adj[u] and y not in adj[v]:
            new_adj = [list(s) for s in adj]
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_adj[x].remove(y)
            new_adj[y].remove(x)
            new_adj[u].append(x)
            new_adj[x].append(u)
            new_adj[v].append(y)
            new_adj[y].append(v)
            for i in range(n):
                new_adj[i] = sorted(new_adj[i])
            return new_adj, True
    
    return adj, False


def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 110  # 110s to leave margin
    
    # Moore bound for n(k, g)
    if g % 2 == 0:
        # Even girth: Moore bound is 1 + k * (k-1)^((g/2-1))
        moore = 1 + k * (k - 1) ** (g // 2 - 1)
    else:
        # Odd girth: Moore bound is 2 + k * (k-1)^((g-1)//2 - 1) for g >= 3
        if g >= 3:
            moore = 2 + k * (k - 1) ** ((g - 1) // 2 - 1) if (g-1)//2 - 1 >= 0 else 2 + k
        else:
            moore = 2
    
    # Start from a reasonable vertex count
    # Known best for n(4,10) is 320, Moore bound is 243
    # Try starting near the known best and see if we can do better
    
    rng = random.Random(42)
    
    best_n = None
    best_edges = None
    best_girth = 0
    
    # Try different starting sizes
    start_sizes = [320, 300, 280, 260, 250, 245, 244]
    start_sizes = [s for s in start_sizes if s >= moore and s * k % 2 == 0]
    
    for n_try in start_sizes:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for init in range(5):
            if time.time() > deadline - 10:
                break
            
            try:
                adj = _random_k_regular(n_try, k, rng)
            except Exception:
                continue
            
            # Check initial girth
            current_girth = _compute_girth(adj, n_try, max_girth=g + 2)
            
            if current_girth >= g:
                # Already good!
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                    best_girth = current_girth
                break  # No need to search further for this n_try
            
            # Simulated annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            steps = 0
            max_steps = 50000
            
            while T > T_min and steps < max_steps and time.time() < deadline - 5:
                steps += 1
                
                # Apply a 2-switch
                new_adj, success = _two_switch(adj, n_try, k, rng)
                if not success:
                    T *= cooling
                    continue
                
                # Compute new girth
                new_girth = _compute_girth(new_adj, n_try, max_girth=g + 2)
                
                # Accept if better or with probability exp(-delta/T)
                if new_girth >= current_girth:
                    adj = new_adj
                    current_girth = new_girth
                else:
                    delta = current_girth - new_girth
                    if rng.random() < np.exp(-delta / max(T, 1e-10)):
                        adj = new_adj
                        current_girth = new_girth
                
                T *= cooling
                
                if current_girth >= g:
                    break
            
            if current_girth >= g:
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                    best_girth = current_girth
                break  # Found a valid graph for this n_try, try smaller
    
    # If we found a valid graph, return it
    if best_n is not None and best_edges is not None:
        return best_n, best_edges
    
    # Fallback: return a known valid construction (Lazebnik-Ustimenko-Woldar)
    # This is the seed program's approach
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

# EVOLVE-BLOCK-END
