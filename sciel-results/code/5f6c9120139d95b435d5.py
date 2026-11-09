"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def _compute_girth(n, edges, max_depth=None):
    """Compute girth by BFS from each vertex, tracking shortest cycle back to start."""
    if max_depth is None:
        max_depth = 50
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best = float('inf')
    for start in range(n):
        # BFS tracking (node, dist, prev)
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, -1, 0)])  # (node, parent, dist)
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= max_depth:
                continue
            if dist + 1 >= best:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if visited[nb] == -1:
                    visited[nb] = dist + 1
                    queue.append((nb, node, dist + 1))
                else:
                    # Found a cycle
                    cycle_len = dist + 1 + visited[nb]
                    if cycle_len < best and cycle_len >= 3:
                        best = cycle_len
    return best

def _is_valid(k, g, n, edges):
    """Check if graph is k-regular, simple, and has girth >= g."""
    if len(edges) != n * k // 2:
        return False
    # Check degree
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    # Check girth
    girth = _compute_girth(n, edges)
    return girth >= g

def _random_regular_graph(n, k, seed=42):
    """Generate a random k-regular graph using configuration model + 2-switches."""
    rng = random.Random(seed)
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    
    edges = []
    seen = set()
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self-loop, swap with another
            for j in range(i+2, n*k, 2):
                if stubs[j] != u:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    break
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            edges.append((u, v))
    
    # Check if it's actually k-regular and simple
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    if len(edges) != n * k // 2:
        return None
    
    return edges

def _improve_girth(n, k, edges, target_girth, max_time=100, seed=42):
    """Try to increase girth using edge swaps (2-switches)."""
    rng = random.Random(seed)
    start_time = time.time()
    edges = list(edges)
    
    # Build adjacency
    def build_adj(edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    adj = build_adj(edges)
    current_girth = _compute_girth(n, edges)
    
    if current_girth >= target_girth:
        return edges, current_girth
    
    # Try random edge swaps
    attempts = 0
    while time.time() - start_time < max_time:
        attempts += 1
        # Pick two random edges
        e1_idx = rng.randint(0, len(edges) - 1)
        e2_idx = rng.randint(0, len(edges) - 1)
        if e1_idx == e2_idx:
            continue
        
        u1, v1 = edges[e1_idx]
        u2, v2 = edges[e2_idx]
        
        # Try swapping: (u1,v1),(u2,v2) -> (u1,u2),(v1,v2)
        new_u1, new_v1 = u1, u2
        new_u2, new_v2 = v1, v2
        
        # Check validity
        if new_u1 == new_v1 or new_u2 == new_v2:
            continue
        if new_u1 == new_u2 or new_u1 == new_v2 or new_v1 == new_u2 or new_v1 == new_v2:
            continue
        
        key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
        key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
        
        # Check if these edges already exist
        existing = set()
        for u, v in edges:
            existing.add((min(u, v), max(u, v)))
        
        if key1 in existing and key1 != (min(u1,v1), max(u1,v1)):
            continue
        if key2 in existing and key2 != (min(u2,v2), max(u2,v2)):
            continue
        
        # Apply swap
        edges[e1_idx] = (new_u1, new_v1)
        edges[e2_idx] = (new_u2, new_v2)
        
        # Check girth
        new_girth = _compute_girth(n, edges)
        if new_girth > current_girth:
            current_girth = new_girth
            if current_girth >= target_girth:
                return edges, current_girth
        else:
            # Revert
            edges[e1_idx] = (u1, v1)
            edges[e2_idx] = (u2, v2)
    
    return edges, current_girth

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_n = float('inf')
    best_edges = None
    
    # Strategy 1: Try LUW construction as baseline
    luw_n, luw_edges = _luw_construct(k, g)
    if luw_n < best_n:
        best_n = luw_n
        best_edges = luw_edges
    
    # Strategy 2: Try random regular graphs with girth improvement
    # For (4,10), known best is 320. Try various sizes.
    if k == 4 and g == 10:
        # Try sizes from 250 to 350
        for target_n in [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]:
            if time.time() > deadline - 20:
                break
            if target_n < best_n:
                continue
            
            # Try multiple seeds
            for seed in [42, 123, 777, 999, 555]:
                if time.time() > deadline - 5:
                    break
                
                edges = _random_regular_graph(target_n, k, seed=seed)
                if edges is None:
                    continue
                
                # Check initial girth
                girth = _compute_girth(target_n, edges)
                if girth >= g:
                    if target_n < best_n:
                        best_n = target_n
                        best_edges = edges
                    break
                
                # Try to improve girth
                if girth >= g - 2:  # Only if close
                    improved_edges, new_girth = _improve_girth(
                        target_n, k, edges, g, 
                        max_time=min(10, deadline - time.time() - 5),
                        seed=seed
                    )
                    if new_girth >= g:
                        if target_n < best_n:
                            best_n = target_n
                            best_edges = improved_edges
                        break
    
    # Fallback: use LUW if nothing better
    if best_edges is None:
        best_n, best_edges = luw_n, luw_edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
