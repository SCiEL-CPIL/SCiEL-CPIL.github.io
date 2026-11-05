"""Construct a 3-regular graph of girth at least 16 on as few vertices as possible.

Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction which gives
bipartite q-regular graphs of large girth. For k=3 (prime field GF(3)) we can
get a 3-regular graph. The LUW construction CD(d, q) gives a q-regular bipartite
graph with girth at least d+5 (for appropriate d). For girth >= 16, we need d >= 11.

However, the LUW graph has many vertices (q^d * 2). For q=3, d=11, that's 2*3^11 ≈ 354k vertices.
That's too many. We need a smarter approach.

Alternative: Use a known construction or search for a smaller graph.
The Moore bound for n(3,16) is 512. The best known is 936.

Given the time constraint (150s), let me try a different approach:
1. Start with a random 3-regular graph
2. Use local search to increase girth

Actually, let me think about this more carefully. The problem asks for a 3-regular
graph with girth at least 16. The smallest known has 936 vertices.

A practical approach within the time limit:
- Use the LUW construction with small parameters
- Or use a known explicit construction

Let me try the LUW construction with q=3 and find the smallest d that gives girth >= 16.

For LUW CD(d, q): the girth is at least d+5 for odd d. So for girth >= 16, we need d >= 11.
But the number of vertices is 2*q^d = 2*3^11 = 354294. That's way too large.

Let me try a different approach. I'll use a greedy/random construction:
1. Start with n vertices
2. Randomly assign edges to make a 3-regular graph
3. Check girth
4. If girth < 16, try to fix by rewiring

Actually, within 150 seconds, let me just implement a reasonable construction.

The best practical approach: Use the LUW construction but with the understanding that
it may produce a large graph. The score is n (minimized), so any valid graph with fewer
vertices is better.

Let me try to implement a search-based approach:
- Start with the Moore bound (512) and try to find a 3-regular graph of that size with girth >= 16
- If that fails, increase n

But finding such a graph is extremely hard in general.

Practical solution: Implement the LUW construction correctly and return it. Even if it's
large, it's valid. The score will be worse than 936 but it's a valid construction.

Wait, let me reconsider. The task says the score is n (minimized), and the current seed
gets score 1.0 which means it produces some graph. Let me check what the current LUW
implementation produces.

For k=3, g=16: q=3, d = max(3, 16-5=11 if 11%2 else 16-4=12) = 11 (since 11 is odd).
The graph has 2*3^11 = 354294 vertices. That's a valid 3-regular graph of girth >= 16.

Can I do better? Let me try a local search approach starting from a random 3-regular graph.
"""

import time
import random
import numpy as np
from collections import deque

def _bfs_girth(adj, n):
    """Compute girth using BFS from each vertex, finding shortest cycle through that vertex."""
    min_girth = 2 * n  # larger than any possible girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle: u -> ... -> v and v is already visited
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    return min_girth

def _check_girth(adj, n, g):
    """Check if girth is at least g."""
    if n < g:
        return False
    # Quick check: BFS from each vertex for depth g//2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] <= g // 2:
                        queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    deadline = time.time() + 130  # Leave 20s safety margin
    
    if k != 3:
        # Fallback for other k values
        return _luw_construct(k, g)
    
    # For k=3, g=16: try to find a small 3-regular graph with girth >= 16
    # Moore bound is 512, best known is 936
    
    # Strategy: 
    # 1. Try to use a known small construction if available
    # 2. Otherwise use LUW as fallback
    
    # Let's try a random 3-regular graph search with local improvement
    # Start near the Moore bound and work up
    
    best_n = None
    best_edges = None
    best_adj = None
    
    # Try a few sizes starting from Moore bound
    moore_bound = 2 * (k - 1) + 2 * (k - 1) * (k - 2)  # For girth >= 6, this is the Moore bound
    # For girth 16, Moore bound is larger. Let's use a rough estimate.
    # Moore bound for (3, 16): ~2*sum_{i=0}^{7} 2^i * 2 = 2*(2^8 - 1)*2 = 1022? No...
    # Actually for d-regular graph with girth g, the Moore bound is:
    # n >= 2 * sum_{i=0}^{floor((g-1)/2)} (d-1)^i for even g
    # For d=3, g=16: n >= 2 * sum_{i=0}^{7} 2^i = 2 * 255 = 510
    # So n >= 512 (next even number for 3-regular)
    
    start_n = 512
    max_n = 1000  # Don't go too far above known best
    
    random.seed(42)
    
    # Try a random construction approach
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        if best_n is not None:
            n = best_n - 1  # Try to improve
            if n < start_n:
                n = start_n
        else:
            n = start_n + attempt * 2
        
        if n > max_n or n % 2 != 0:
            if n % 2 != 0:
                n += 1
            if n > max_n:
                break
        
        # Generate random 3-regular graph
        adj = _random_3reg(n, random)
        if adj is None:
            continue
        
        # Check girth
        if _check_girth(adj, n, g):
            best_n = n
            best_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        best_edges.append((u, v))
            # Try to improve further
            n_improve = best_n
            break
    
    # If no small graph found, use LUW as fallback
    if best_n is None or time.time() > deadline - 5:
        if best_n is not None:
            return best_n, best_edges
        n, edges = _luw_construct(3, g)
        return n, edges
    
    return best_n, best_edges

def _random_3reg(n, rng):
    """Generate a random 3-regular graph on n vertices (n must be even, n>=4)."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Use configuration model / random pairing
    # Each vertex gets 3 half-edges
    half_edges = []
    for v in range(n):
        half_edges.extend([v] * 3)
    
    rng.shuffle(half_edges)
    
    # Pair up half-edges
    adj = [set() for _ in range(n)]
    i = 0
    while i < len(half_edges):
        j = i + 1
        u, v = half_edges[i], half_edges[j]
        if u == v:
            # Self-loop, retry
            # Find a different partner
            for j2 in range(j + 1, len(half_edges)):
                if half_edges[j2] != u:
                    j = j2
                    break
            else:
                return None
            v = half_edges[j]
            half_edges[i], half_edges[j] = half_edges[j], half_edges[i]
        if v in adj[u]:
            # Multi-edge, try to swap
            return None
        adj[u].add(v)
        adj[v].add(u)
        i += 2
    
    # Check all degrees are exactly 3
    for u in range(n):
        if len(adj[u]) != 3:
            return None
    
    return adj

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
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
