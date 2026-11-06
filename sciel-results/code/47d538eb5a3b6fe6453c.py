"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _build_edge_index():
    """Map each edge (min,max) to its one-based arrival index."""
    n = P['n']
    edge_idx = {}
    for i, (u, v) in enumerate(P['edge_order']):
        a, b = min(u, v), max(u, v)
        edge_idx[(a, b)] = i + 1  # one-based
    return edge_idx

def _get_edges_up_to(T, edge_idx):
    """Return set of edges with arrival index <= T."""
    n = P['n']
    edges = set()
    for i, (u, v) in enumerate(P['edge_order']):
        if i + 1 > T:
            break
        a, b = min(u, v), max(u, v)
        edges.add((a, b))
    return edges

def _has_hamilton_cycle(n, edges, start=0):
    """Check if the graph with given edges has a Hamilton cycle starting/ending at 'start'.
    Uses backtracking with pruning."""
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check each vertex has degree >= 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None
    
    # Backtracking: build path from start
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    def backtrack():
        if len(path) == n:
            # Check if last vertex connects to start
            if start in adj[path[-1]]:
                return True
            return False
        
        last = path[-1]
        # Try neighbors in order of degree (heuristic: try low-degree vertices first)
        neighbors = sorted(adj[last], key=lambda x: len(adj[x]))
        for nb in neighbors:
            if not visited[nb]:
                # Pruning: if nb is not start and it's the only unvisited vertex left, 
                # check it connects to start
                if len(path) == n - 1:
                    if start not in adj[nb]:
                        continue
                visited[nb] = True
                path.append(nb)
                if backtrack():
                    return True
                path.pop()
                visited[nb] = False
        return False
    
    if backtrack():
        return path
    return None

def _find_hamilton_cycle_with_max_edge(edge_idx, max_T=None):
    """Find a Hamilton cycle minimizing the maximum edge arrival index.
    Uses binary search on T."""
    n = P['n']
    total_edges = len(P['edge_order'])
    
    if max_T is None:
        max_T = total_edges
    
    best_cycle = None
    best_T = max_T + 1
    
    # Binary search on T
    lo, hi = n, max_T  # minimum possible is n (24 edges)
    
    # First, let's try to find the minimum T
    while lo <= hi:
        mid = (lo + hi) // 2
        edges = _get_edges_up_to(mid, edge_idx)
        cycle = _has_hamilton_cycle(n, edges)
        if cycle is not None:
            best_cycle = cycle
            best_T = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle, best_T

def _try_random_search(edge_idx, n, budget=5.0):
    """Try random local search to find a good Hamilton cycle."""
    start_time = time.time()
    best_score = float('inf')
    best_cycle = list(range(n))
    
    # Build edge index lookup
    def edge_index(u, v):
        a, b = min(u, v), max(u, v)
        return edge_idx.get((a, b), float('inf'))
    
    def cycle_score(cycle):
        """Max one-based arrival index of edges in the cycle."""
        max_idx = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            idx = edge_index(u, v)
            if idx > max_idx:
                max_idx = idx
        return max_idx
    
    # Initial: try the identity and a few random permutations
    candidates = [list(range(n))]
    rng = random.Random(614)
    for _ in range(50):
        c = list(range(n))
        rng.shuffle(c)
        candidates.append(c)
    
    for c in candidates:
        s = cycle_score(c)
        if s < best_score:
            best_score = s
            best_cycle = c[:]
    
    # Local search: 2-opt and 3-opt
    def two_opt(cycle, i, j):
        """Reverse segment [i..j]"""
        new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
        return new_cycle
    
    improved = True
    iterations = 0
    max_iter = 50000
    while improved and iterations < max_iter and (time.time() - start_time) < budget:
        improved = False
        iterations += 1
        # Try random 2-opt moves
        for _ in range(200):
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_cycle = two_opt(best_cycle, i, j)
            s = cycle_score(new_cycle)
            if s < best_score:
                best_score = s
                best_cycle = new_cycle
                improved = True
    
    return best_cycle, best_score

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = P['n']
    edge_idx = _build_edge_index()
    
    # Try binary search approach
    cycle, T = _find_hamilton_cycle_with_max_edge(edge_idx)
    
    if cycle is not None:
        # Verify
        def edge_index(u, v):
            a, b = min(u, v), max(u, v)
            return edge_idx.get((a, b), float('inf'))
        score = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            idx = edge_index(u, v)
            if idx > score:
                score = idx
        if score <= T:
            return cycle
        # Score is higher than expected, use it anyway
        return cycle
    
    # Fallback: random search
    if time.time() - start_time < deadline - 5:
        cycle, score = _try_random_search(edge_idx, n, budget=deadline - time.time() - 1)
        return cycle
    
    return list(range(n))

# EVOLVE-BLOCK-END
