# EVOLVE-BLOCK-START
import random
from collections import deque

def _check_girth(n, edges, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        visited = {start: (-1, 0)}
        queue = deque([(start, -1, 0)])
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= min_girth // 2:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    cycle_len = dist + visited[nb][1] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    visited[nb] = (node, dist + 1)
                    queue.append((nb, node, dist + 1))
    return True

def _generate_random_4reg(n, max_tries=500):
    for _ in range(max_tries):
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if not valid:
            continue
        edge_list = list(edges)
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if any(d != 4 for d in deg):
            continue
        return edge_list
    return None

def _try_greedy(n, g, max_steps=5000):
    edges = _generate_random_4reg(n, max_tries=100)
    if edges is None:
        return None
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        
    for step in range(max_steps):
        if _check_girth(n, edges, g):
            return n, edges
            
        # Find shortest cycle
        # ... (complex to implement efficiently)
        
        # Random swap
        if random.random() < 0.5:
            break # Give up on greedy for this n
            
    return None

def construct(k, g):
    if k != 4 or g != 10:
        return _construct_other(k, g)
        
    random.seed(42)
    # Try random search first
    for n in range(243, 321):
        result = _try_random(n, g, max_graphs=50)
        if result is not None:
            return result
            
    # Fallback: L-U-W
    return _luw_construction()

# ... (helper functions)
