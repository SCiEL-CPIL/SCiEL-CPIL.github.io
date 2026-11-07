"""Solve the few-color bicolored-forest coloring on the pinned 4-regular graph.
Strategy: try small k with a fast feasibility check; if none, do a light local
search to reduce colors from a good initial coloring."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_valid(coloring):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Build edges per color pair
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    # For each pair, check if it's a forest using union-find
    for key, edges in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
    return True


def _num_colors(coloring):
    return len(set(coloring))


def _try_k_colors(k, adj, max_tries=2000000):
    """Try to find a valid coloring with at most k colors via random search."""
    n = P['n']
    rng = random.Random(42 + k)
    # Initial random coloring
    coloring = [rng.randrange(k) for _ in range(n)]
    
    # First try to fix adjacency violations
    for _ in range(50000):
        violations = []
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                violations.append((u, v))
        if not violations:
            break
        u, v = violations[0]
        # Try to recolor u to a different color not used by neighbors
        used = set(coloring[nb] for nb in adj[u])
        avail = [c for c in range(k) if c not in used]
        if avail:
            coloring[u] = rng.choice(avail)
        else:
            # Recolor v
            used_v = set(coloring[nb] for nb in adj[v])
            avail_v = [c for c in range(k) if c not in used_v]
            if avail_v:
                coloring[v] = rng.choice(avail_v)
    
    if not all(coloring[u] != coloring[v] for u, v in P['edges']):
        return None
    
    # Now try to fix bicolored cycles
    for attempt in range(max_tries):
        if _check_valid(coloring):
            return [int(c) for c in coloring]
        # Find a bicolored cycle and try to break it
        # For simplicity, just randomly recolor a vertex
        v = rng.randrange(n)
        used = set(coloring[nb] for nb in adj[v])
        avail = [c for c in range(k) if c not in used]
        if avail:
            coloring[v] = rng.choice(avail)
    
    return None


def _local_search(start_coloring, adj, max_iter=1000000, deadline=None):
    """Local search to reduce number of colors while maintaining validity."""
    coloring = [int(c) for c in start_coloring]
    n = P['n']
    rng = random.Random(12345)
    best = [int(c) for c in coloring]
    best_count = _num_colors(coloring)
    
    # First ensure validity
    if not _check_valid(coloring):
        return None
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Try to recolor a vertex to an existing color to reduce count
        v = rng.randrange(n)
        used = set(coloring[nb] for nb in adj[v])
        current = coloring[v]
        # Try all colors currently in use
        colors_in_use = list(set(coloring))
        rng.shuffle(colors_in_use)
        for c in colors_in_use:
            if c == current or c in used:
                continue
            old = coloring[v]
            coloring[v] = c
            if _check_valid(coloring):
                cnt = _num_colors(coloring)
                if cnt < best_count:
                    best_count = cnt
                    best = [int(x) for x in coloring]
                    break
            coloring[v] = old
    
    return best if best_count < _num_colors(start_coloring) else None


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try k=2, 3, 4, 5 in order
    for k in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        result = _try_k_colors(k, adj, max_tries=500000)
        if result is not None:
            if _check_valid(result):
                return result
    
    # Fallback: use a structured coloring and local search
    # Try several initial colorings
    initial_colorings = [
        [v % 5 for v in range(n)],
        [(v // 10) * 2 + (v % 2) for v in range(n)],
        [(v % 4) + 4 * ((v // 10) % 2) for v in range(n)],
    ]
    
    best_result = None
    best_count = n + 1
    
    for init in initial_colorings:
        if time.time() > deadline - 5:
            break
        # Fix adjacency violations
        coloring = [int(c) for c in init]
        for _ in range(100000):
            violations = [(u, v) for u, v in P['edges'] if coloring[u] == coloring[v]]
            if not violations:
                break
            u, v = violations[0]
            used = set(coloring[nb] for nb in adj[u])
            avail = [c for c in range(10) if c not in used]
            if avail:
                coloring[u] = avail[0]
            else:
                used_v = set(coloring[nb] for nb in adj[v])
                avail_v = [c for c in range(10) if c not in used_v]
                if avail_v:
                    coloring[v] = avail_v[0]
        
        if not _check_valid(coloring):
            continue
        
        cnt = _num_colors(coloring)
        if cnt < best_count:
            best_count = cnt
            best_result = [int(c) for c in coloring]
        
        # Try local search to reduce colors
        if time.time() > deadline - 10:
            break
        improved = _local_search(coloring, adj, max_iter=200000, deadline=deadline - 5)
        if improved is not None:
            cnt2 = _num_colors(improved)
            if cnt2 < best_count:
                best_count = cnt2
                best_result = improved
    
    if best_result is not None:
        return best_result
    
    # Guaranteed valid fallback: all distinct colors
    return list(range(n))


if __name__ == "__main__":
    result = construct()
    print("colors used:", _num_colors(result))
    print("valid:", _check_valid(result))
