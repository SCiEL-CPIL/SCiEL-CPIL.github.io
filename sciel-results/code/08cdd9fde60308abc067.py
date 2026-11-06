import time
import heapq
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Strategy 1: Try all starting vertices with greedy + lookahead
    for start in range(n):
        if time.time() > deadline:
            break
        path = _greedy_path(start, adj, n)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 2: DFS with heuristic ordering from each start
    for start in range(n):
        if time.time() > deadline:
            break
        path = _dfs_heuristic(start, adj, n, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 3: Iterative deepening DFS with pruning
    # Start from current best length and try to extend
    if len(best_path) < n:
        if time.time() < deadline - 10:
            path = _iterative_deepening(adj, n, len(best_path) + 1, deadline)
            if path and len(path) > len(best_path):
                best_path = path
    
    # Strategy 4: Try to extend the best path found so far
    if len(best_path) < n:
        if time.time() < deadline - 5:
            extended = _extend_path(best_path, adj, n, deadline)
            if extended and len(extended) > len(best_path):
                best_path = extended
    
    # Strategy 5: Randomized restarts with different heuristics
    if len(best_path) < n:
        if time.time() < deadline - 2:
            rng = random.Random(748)
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                start = rng.randint(0, n - 1)
                path = _greedy_random(start, adj, n, rng)
                if len(path) > len(best_path):
                    best_path = path
    
    # Final: try to extend once more
    if len(best_path) < n:
        extended = _extend_path(best_path, adj, n, deadline)
        if extended and len(extended) > len(best_path):
            best_path = extended
    
    return best_path


def _greedy_path(start, adj, n):
    """Greedy path using minimum remaining degree heuristic."""
    path = [start]
    seen = {start}
    while True:
        current = path[-1]
        neighbors = adj[current] - seen
        if not neighbors:
            break
        # Pick neighbor with minimum remaining degree (Warnsdorff-like)
        v = min(neighbors, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path


def _dfs_heuristic(start, adj, n, deadline):
    """DFS with heuristic ordering, trying to find a long path."""
    best = []
    
    def dfs(path, seen):
        nonlocal best
        if time.time() > deadline:
            return
        if len(path) > len(best):
            best = list(path)
        
        current = path[-1]
        neighbors = sorted(adj[current] - seen, key=lambda w: (len(adj[w] - seen), w))
        
        for v in neighbors:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
    
    dfs([start], {start})
    return best


def _iterative_deepening(adj, n, target_len, deadline):
    """Try to find a path of at least target_len using DFS with pruning."""
    best = []
    
    # Try from each starting vertex
    for start in range(n):
        if time.time() > deadline:
            return best
        if len(adj[start]) == 0:
            continue
        
        path = [start]
        seen = {start}
        
        def dfs():
            nonlocal best
            if time.time() > deadline:
                return
            if len(path) > len(best):
                best = list(path)
            if len(path) >= target_len:
                return
            
            current = path[-1]
            neighbors = list(adj[current] - seen)
            # Sort by degree heuristic
            neighbors.sort(key=lambda w: (len(adj[w] - seen), w))
            
            for v in neighbors:
                if time.time() > deadline:
                    return
                path.append(v)
                seen.add(v)
                dfs()
                path.pop()
                seen.remove(v)
                if len(path) >= target_len:
                    return
        
        dfs()
        if len(best) >= target_len:
            return best
    
    return best


def _extend_path(path, adj, n, deadline):
    """Try to extend an existing path by adding vertices at either end."""
    if not path:
        return path
    
    seen = set(path)
    extended = list(path)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Try to extend at the beginning
        first = extended[0]
        for v in sorted(adj[first] - seen, key=lambda w: len(adj[w] - seen)):
            if time.time() > deadline:
                return extended
            # Check if adding v at the front is valid
            extended.insert(0, v)
            seen.add(v)
            changed = True
            break
        
        # Try to extend at the end
        last = extended[-1]
        for v in sorted(adj[last] - seen, key=lambda w: len(adj[w] - seen)):
            if time.time() > deadline:
                return extended
            extended.append(v)
            seen.add(v)
            changed = True
            break
    
    return extended


def _greedy_random(start, adj, n, rng):
    """Greedy path with random tie-breaking."""
    path = [start]
    seen = {start}
    while True:
        current = path[-1]
        neighbors = list(adj[current] - seen)
        if not neighbors:
            break
        # Weight by inverse degree, with randomness
        weights = [1.0 / max(1, len(adj[w] - seen)) for w in neighbors]
        total = sum(weights)
        r = rng.random() * total
        cumsum = 0
        chosen = neighbors[-1]
        for w, wt in zip(neighbors, weights):
            cumsum += wt
            if r <= cumsum:
                chosen = w
                break
        path.append(chosen)
        seen.add(chosen)
    return path
# EVOLVE-BLOCK-END
