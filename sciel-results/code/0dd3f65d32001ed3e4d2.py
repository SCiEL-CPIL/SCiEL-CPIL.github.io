"""Rewritten program: multi-strategy greedy + bidirectional extension + 2-opt polish."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

_N = P['n']
_EDGES = P['edges']

def _build_adj():
    adj = [set() for _ in range(_N)]
    for u, v in _EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_path(path, adj):
    if not path:
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj[path[i]]:
            return False
    return True

def _warnsdorff_extend(path, adj, rng=None):
    """Greedy Warnsdorff: pick neighbor with fewest onward options."""
    if rng is None:
        rng = random.Random(0)
    seen = set(path)
    path = list(path)
    while True:
        cur = path[-1]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            break
        if rng.random() < 0.1:
            v = rng.choice(cands)
        else:
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def _dfs_extend(path, adj, max_steps=200000):
    """Limited DFS to try to extend from the end."""
    path = list(path)
    seen = set(path)
    steps = 0
    best = list(path)
    def dfs(cur, depth):
        nonlocal steps, best
        if steps > max_steps:
            return
        steps += 1
        if len(path) > len(best):
            best = list(path)
        cands = [v for v in adj[cur] if v not in seen]
        for v in cands:
            path.append(v)
            seen.add(v)
            dfs(v, depth + 1)
            path.pop()
            seen.discard(v)
            if steps > max_steps:
                return
    dfs(path[-1], 0)
    # Extend best if it's a prefix-compatible extension
    return best

def _bidirectional_extend(path, adj):
    """Extend path from both ends using greedy."""
    path = list(path)
    seen = set(path)
    # Extend forward (from end)
    while True:
        cur = path[-1]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            break
        v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    # Extend backward (from front)
    while True:
        cur = path[0]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            break
        v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        path.insert(0, v)
        seen.add(v)
    return path

def _try_2opt(path, adj):
    """Try 2-opt style reversals to allow further extension.
    For a path, 2-opt means: reverse a sub-segment to create a different
    endpoint configuration, then re-extend."""
    n = len(path)
    if n < 4:
        return path
    best = list(path)
    # Try reversing subpaths to change endpoints
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            # Check if reversing path[i..j] keeps it a valid path
            # After reversal: ... path[i-1], path[j], path[j-1], ..., path[i], path[j+1] ...
            # Need edge (path[i-1], path[j]) and (path[i], path[j+1]) if those indices exist
            left_ok = True
            right_ok = True
            if i > 0:
                if path[j] not in adj[path[i - 1]]:
                    left_ok = False
            if j < n - 1:
                if path[i] not in adj[path[j + 1]]:
                    right_ok = False
            if left_ok and right_ok:
                new_path = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
                # Now try to extend this
                ext = _bidirectional_extend(new_path, adj)
                if len(ext) > len(best):
                    best = ext
    return best

def _greedy_start(start, adj, rng):
    """Start from a vertex and greedily build a path."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            break
        if rng.random() < 0.15:
            v = rng.choice(cands)
        else:
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    adj = _build_adj()
    best = []
    
    # Strategy 1: Warnsdorff from every start vertex
    for start in range(_N):
        if time.time() - t0 > DEADLINE:
            break
        path = _warnsdorff_extend([start], adj)
        if len(path) > len(best):
            best = list(path)
    
    # Strategy 2: Bidirectional extension from each greedy path
    for start in range(_N):
        if time.time() - t0 > DEADLINE:
            break
        path = _warnsdorff_extend([start], adj)
        path = _bidirectional_extend(path, adj)
        if len(path) > len(best):
            best = list(path)
    
    # Strategy 3: Multiple randomized greedy starts + bidirectional
    rng = random.Random(42)
    for trial in range(50):
        if time.time() - t0 > DEADLINE:
            break
        start = rng.randint(0, _N - 1)
        path = _greedy_start(start, adj, rng)
        path = _bidirectional_extend(path, adj)
        if len(path) > len(best):
            best = list(path)
    
    # Strategy 4: 2-opt polish on best path
    if time.time() - t0 < DEADLINE:
        polished = _try_2opt(best, adj)
        if _is_path(polished, adj) and len(polished) > len(best):
            best = polished
    
    # Strategy 5: More randomized trials with different seeds
    for seed in range(10, 60):
        if time.time() - t0 > DEADLINE:
            break
        rng2 = random.Random(seed)
        for start in range(_N):
            if time.time() - t0 > DEADLINE:
                break
            path = _greedy_start(start, adj, rng2)
            path = _bidirectional_extend(path, adj)
            if len(path) > len(best):
                best = list(path)
    
    # Final 2-opt polish
    if time.time() - t0 < DEADLINE and len(best) >= 4:
        polished = _try_2opt(best, adj)
        if _is_path(polished, adj) and len(polished) > len(best):
            best = polished
    
    # Validate
    if not _is_path(best, adj):
        # Fallback: single Warnsdorff
        best = _warnsdorff_extend([0], adj)
    
    if not best:
        best = [0]
    
    return best
# EVOLVE-BLOCK-END
