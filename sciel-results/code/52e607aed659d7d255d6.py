import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80

    # Build adjacency from base path
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)

    existing = set()
    for u, v in base_edges:
        existing.add((min(u,v), max(u,v)))

    # All possible candidate edges (non-path edges)
    # Restrict to bipartite edges (even-odd) to guarantee triangle-free
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in existing:
                if (u % 2) != (v % 2):
                    candidates.append((u, v))

    def check_diameter(adj2):
        """Check if diameter of graph with adjacency adj2 is <= 4."""
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj2[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
        return True

    def is_valid(added):
        # Triangle-free is guaranteed by bipartite selection
        # Check connected and diameter
        adj2 = [set(adj[i]) for i in range(n)]
        for u, v in added:
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Connected check
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj2[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        return check_diameter(adj2)

    # Known 7-edge solution (verified to be valid? Let's assume a valid one exists)
    # Let's use a safe 8-edge solution as fallback if search fails
    # Hub 11 and 12
    fallback = [
        [11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23],
        [12, 2], [12, 5], [12, 8], [12, 14], [12, 17], [12, 20]
    ]
    # Actually, let's just use the one from the prompt if it works, or a larger safe one.
    # The prompt's 7-edge solution might be invalid.
    # Let's construct a valid 10-edge solution as absolute fallback.
    # Connect 0-12, 23-11, and all others to nearest hub.
    safe_fallback = []
    for i in range(n):
        if i == 11 or i == 12:
            continue
        if abs(i - 11) < abs(i - 12):
            if (i, 11) not in existing and (11, i) not in existing:
                if (i % 2) != (11 % 2):
                    safe_fallback.append([i, 11])
                else:
                    # Connect to 12 if 11 is same parity
                    if (i, 12) not in existing and (12, i) not in existing:
                        safe_fallback.append([i, 12])
        else:
            if (i, 12) not in existing and (12, i) not in existing:
                if (i % 2) != (12 % 2):
                    safe_fallback.append([i, 12])
                else:
                    if (i, 11) not in existing and (11, i) not in existing:
                        safe_fallback.append([i, 11])
    
    best = safe_fallback
    best_len = len(best)

    # Search for smaller k
    import random
    for k in range(5, min(best_len, 8)):
        if time.time() > deadline:
            break
        found = False
        # Random search with restarts
        for _ in range(50000):
            if time.time() > deadline:
                break
            idxs = random.sample(range(len(candidates)), k)
            added = [candidates[i] for i in idxs]
            if is_valid(added):
                best = added
                found = True
                break
        if found:
            best_len = k

    return [[int(u), int(v)] for u, v in best]
