import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

degrees = [len(adj[i]) for i in range(n)]

def _has_bicolor_cycle(c1, c2, color):
    """Check if subgraph induced by colors c1 and c2 has a cycle using BFS."""
    if c1 == c2:
        return False
    vertices = [v for v in range(n) if color[v] in (c1, c2)]
    if len(vertices) < 3:
        return False
    vset = set(vertices)
    visited = set()
    for start in vertices:
        if start in visited:
            continue
        stack = [(start, -1)]
        visited.add(start)
        while stack:
            cur, par = stack.pop()
            for nb in adj[cur]:
                if nb not in vset:
                    continue
                if nb == par:
                    continue
                if nb in visited:
                    return True
                visited.add(nb)
                stack.append((nb, cur))
    return False

def _is_valid(color, v, c):
    """Check if assigning color c to vertex v is valid."""
    for nb in adj[v]:
        if color[nb] == c:
            return False
    for nb in adj[v]:
        nc = color[nb]
        if nc != -1 and nc != c:
            color[v] = c
            if _has_bicolor_cycle(c, nc, color):
                color[v] = -1
                return False
            color[v] = -1
    return True

def _try_k_colors(k, deadline):
    """Try to find a valid coloring with at most k colors."""
    if k < 2:
        return None
    # Order: process vertices in a specific pattern based on graph structure
    # The graph has a spine 0-1-2-...-49 with chords
    # Process in order that respects the structure
    order = list(range(n))
    
    color = [-1] * n
    
    def backtrack(idx):
        if time.time() > deadline - 1:
            return None
        if idx == n:
            return color[:]
        v = order[idx]
        # Try colors 0..k-1, preferring already used colors
        used = set(color[:idx])
        for c in range(k):
            if c in used and not _is_valid(color, v, c):
                continue
            if c not in used and not _is_valid(color, v, c):
                continue
            color[v] = c
            res = backtrack(idx + 1)
            if res is not None:
                return res
            color[v] = -1
        return None
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Trivial valid solution: all different colors
    fallback = list(range(n))
    
    # Try increasing number of colors
    for k in range(2, n + 1):
        if time.time() > deadline - 3:
            break
        result = _try_k_colors(k, deadline)
        if result is not None:
            return result
    
    return fallback

# EVOLVE-BLOCK-END
