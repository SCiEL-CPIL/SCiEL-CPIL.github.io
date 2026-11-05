"""Longest path in pinned G(100, 1.8/100) via multi-strategy search."""
import time
import random
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Convert to sorted lists for speed
adj_list = [sorted(adj[i]) for i in range(N)]
adj_set = adj  # keep sets for fast membership


def _greedy_extend(path, seen, rng):
    """Extend path greedily with random tie-breaking."""
    while True:
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in seen]
        if not cands:
            break
        # Weight by fewest remaining unvisited neighbors (Warnsdorff-like)
        cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in seen]), rng.random()))
        v = cands[0]
        path.append(v)
        seen.add(v)
    return path


def _greedy_extend_deterministic(path, seen):
    """Extend path greedily with deterministic tie-breaking."""
    while True:
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in seen]
        if not cands:
            break
        cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in seen]), w))
        v = cands[0]
        path.append(v)
        seen.add(v)
    return path


def _dfs_limited(start, max_depth, seen_init=None, rng=None):
    """DFS from start with limited depth, trying to find long paths."""
    if rng is None:
        rng = random.Random(0)
    best = [start]
    seen = {start}
    
    def dfs(path, seen_set, depth):
        nonlocal best
        if len(path) > len(best):
            best = path[:]
        if depth >= max_depth:
            return
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in seen_set]
        # Sort by fewest unvisited neighbors
        cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in seen_set]), w))
        for v in cands:
            path.append(v)
            seen_set.add(v)
            dfs(path, seen_set, depth + 1)
            path.pop()
            seen_set.discard(v)
    
    dfs([start], seen, 0)
    return best


def _two_opt_improve(path, max_iters=200):
    """Try 2-opt style reversals to extend the path."""
    path = path[:]
    n = len(path)
    if n < 3:
        return path
    
    # Build position map
    pos = {v: i for i, v in enumerate(path)}
    visited = set(path)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing segment [i+1..j]
                # New endpoints: path[i] connects to path[j], path[i+1] connects to path[j+1] (if exists)
                a, b = path[i], path[i + 1]
                c, d = path[j], path[j + 1] if j + 1 < n else None
                
                # Check if reversal creates valid connections
                if b in adj_set[c] and (d is None or a in adj_set[d]):
                    # Also need: if we reverse, path[i]--path[j] and path[i+1]--path[j+1]
                    # Actually let's think differently: try to extend from endpoints
                    pass
        
        # Alternative: try to extend from either end
        for end in range(2):
            if end == 0:
                cur = path[0]
            else:
                cur = path[-1]
            cands = [v for v in adj_list[cur] if v not in visited]
            if cands:
                for v in cands:
                    new_path = [v] + path if end == 0 else path + [v]
                    if len(new_path) > len(path):
                        path = new_path
                        visited.add(v)
                        improved = True
                        break
    
    return path


def _beam_search(start, beam_width=50, max_depth=None):
    """Beam search for longest path from start."""
    if max_depth is None:
        max_depth = N
    
    # Each beam entry: (path, seen_set)
    beams = [([start], {start})]
    best = [start]
    
    for depth in range(max_depth):
        if not beams:
            break
        new_beams = []
        for path, seen in beams:
            cur = path[-1]
            cands = [v for v in adj_list[cur] if v not in seen]
            for v in cands:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beams.append((new_path, new_seen))
                if len(new_path) > len(best):
                    best = new_path
        
        if not new_beams:
            break
        
        # Keep top beam_width by length (then by heuristic)
        new_beams.sort(key=lambda x: (-len(x[0]), sum(len(adj_set[v]) for v in x[0][-1:])))
        beams = new_beams[:beam_width]
    
    return best


def _random_walk_extend(path, seen, rng, steps=100):
    """Try random extensions."""
    path = path[:]
    seen = seen.copy()
    for _ in range(steps):
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in seen]
        if not cands:
            break
        v = rng.choice(cands)
        path.append(v)
        seen.add(v)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s margin
    
    best_path = []
    
    # Phase 1: Deterministic greedy from all starts
    for s in range(N):
        if time.time() > deadline:
            break
        path = [s]
        seen = {s}
        path = _greedy_extend_deterministic(path, seen)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Phase 2: Randomized greedy from all starts (multiple trials)
    rng = random.Random(748)
    for trial in range(200):
        if time.time() > deadline:
            break
        for s in range(N):
            path = [s]
            seen = {s}
            path = _greedy_extend(path, seen, rng)
            if len(path) > len(best_path):
                best_path = path[:]
    
    # Phase 3: Try to extend best_path from both ends
    if len(best_path) < N:
        visited = set(best_path)
        # Try extending from start
        extended = best_path[:]
        cur = extended[0]
        while True:
            cands = [v for v in adj_list[cur] if v not in visited]
            if not cands:
                break
            v = min(cands, key=lambda w: (len([x for x in adj_list[w] if x not in visited]), w))
            extended = [v] + extended
            visited.add(v)
            cur = v
        if len(extended) > len(best_path):
            best_path = extended
        
        # Try extending from end
        extended = best_path[:]
        visited = set(extended)
        cur = extended[-1]
        while True:
            cands = [v for v in adj_list[cur] if v not in visited]
            if not cands:
                break
            v = min(cands, key=lambda w: (len([x for x in adj_list[w] if x not in visited]), w))
            extended = extended + [v]
            visited.add(v)
            cur = v
        if len(extended) > len(best_path):
            best_path = extended
    
    # Phase 4: DFS from promising starts with limited depth
    if time.time() < deadline - 10:
        # Try DFS from vertices with high degree
        deg_sorted = sorted(range(N), key=lambda v: -len(adj_list[v]))
        for s in deg_sorted[:20]:
            if time.time() > deadline - 5:
                break
            # Try DFS with increasing depth limits
            for max_d in [20, 30, 40, 50, 60, 80, 100]:
                if time.time() > deadline - 2:
                    break
                path = _dfs_limited(s, max_d)
                if len(path) > len(best_path):
                    best_path = path[:]
    
    # Phase 5: Beam search from best endpoints
    if time.time() < deadline - 5:
        # Beam search from each vertex with small beam
        for s in range(N):
            if time.time() > deadline - 2:
                break
            path = _beam_search(s, beam_width=30, max_depth=60)
            if len(path) > len(best_path):
                best_path = path[:]
    
    # Phase 6: Final extension attempts
    if best_path:
        visited = set(best_path)
        # Try to extend from both ends greedily
        for _ in range(10):
            changed = False
            # Extend from start
            cur = best_path[0]
            cands = [v for v in adj_list[cur] if v not in visited]
            if cands:
                v = min(cands, key=lambda w: (len([x for x in adj_list[w] if x not in visited]), w))
                best_path = [v] + best_path
                visited.add(v)
                changed = True
            
            # Extend from end
            cur = best_path[-1]
            cands = [v for v in adj_list[cur] if v not in visited]
            if cands:
                v = min(cands, key=lambda w: (len([x for x in adj_list[w] if x not in visited]), w))
                best_path = best_path + [v]
                visited.add(v)
                changed = True
            
            if not changed:
                break
    
    # Phase 7: If we have time, try more randomized approaches
    if time.time() < deadline - 3:
        rng2 = random.Random(12345)
        for trial in range(500):
            if time.time() > deadline - 1:
                break
            # Random start, random greedy
            s = rng2.randint(0, N - 1)
            path = [s]
            seen = {s}
            path = _greedy_extend(path, seen, rng2)
            if len(path) > len(best_path):
                best_path = path[:]
            
            # Try to extend
            visited = set(best_path)
            for end in range(2):
                cur = best_path[0] if end == 0 else best_path[-1]
                cands = [v for v in adj_list[cur] if v not in visited]
                if cands:
                    v = rng2.choice(cands)
                    if end == 0:
                        best_path = [v] + best_path
                    else:
                        best_path = best_path + [v]
                    visited.add(v)
    
    # Ensure non-empty
    if not best_path:
        best_path = [0]
    
    return best_path
