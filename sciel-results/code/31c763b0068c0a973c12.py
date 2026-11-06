"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _warnsdorff(start, adj, rng, min_deg=False):
    """Warnsdorff-style greedy: pick the neighbor with fewest onward options."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            break
        if min_deg:
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        else:
            # pick the one with the fewest unvisited neighbors (Warnsdorff)
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def _dfs_longest(start, adj, deadline):
    """Backtracking DFS to find a long simple path from start, with pruning."""
    best = [start]
    best_len = 1
    path = [start]
    visited = {start}

    def dfs():
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best = list(path)
        cur = path[-1]
        # Try neighbors in order of increasing onward-degree (Warnsdorff heuristic)
        cands = [v for v in adj[cur] if v not in visited]
        cands.sort(key=lambda w: (len(adj[w] - visited), w))
        for v in cands:
            path.append(v)
            visited.add(v)
            dfs()
            path.pop()
            visited.discard(v)
            if time.time() > deadline:
                return

    dfs()
    return best

def _bidirectional_greedy(adj, rng):
    """Grow a path from both ends simultaneously."""
    best = []
    for start in range(P['n']):
        # Randomly choose a second endpoint to seed bidirectional growth
        endpoints = list(range(P['n']))
        endpoints.remove(start)
        if not endpoints:
            continue
        end2 = rng.choice(endpoints)
        
        left = [start]
        right = [end2]
        seen = {start, end2}
        
        # Grow from both ends
        changed = True
        while changed and len(left) + len(right) <= P['n']:
            changed = False
            # Try to extend left end
            cur = left[0]
            cands = [v for v in adj[cur] if v not in seen]
            if cands:
                v = min(cands, key=lambda w: (len(adj[w] - seen), w))
                left.insert(0, v)
                seen.add(v)
                changed = True
            # Try to extend right end
            cur = right[-1]
            cands = [v for v in adj[cur] if v not in seen]
            if cands:
                v = min(cands, key=lambda w: (len(adj[w] - seen), w))
                right.append(v)
                seen.add(v)
                changed = True
        
        # Check if left and right can be connected
        full = left + right
        if len(full) > len(best):
            best = full
    
    return best

def construct():
    adj = _build_adj()
    deadline = time.time() + 80  # 80s out of 90s budget
    
    best = []
    
    # Phase 1: Quick greedy from all starts (both min-deg and Warnsdorff)
    for start in range(P['n']):
        p1 = _warnsdorff(start, adj, None, min_deg=True)
        if len(p1) > len(best):
            best = p1
        if time.time() > deadline:
            return best
    
    # Phase 2: Randomized restarts with different seeds
    rng = random.Random(748)
    for trial in range(50):
        if time.time() > deadline:
            break
        # Random start
        start = rng.randint(0, P['n'] - 1)
        p = _warnsdorff(start, adj, rng, min_deg=rng.random() < 0.5)
        if len(p) > len(best):
            best = p
    
    # Phase 3: Bidirectional greedy
    if time.time() < deadline:
        p = _bidirectional_greedy(adj, rng)
        if len(p) > len(best):
            best = p
    
    # Phase 4: DFS from promising starts (those that gave long greedy paths)
    if time.time() < deadline - 5:
        # Find top starts by greedy path length
        start_scores = []
        for start in range(P['n']):
            p = _warnsdorff(start, adj, None, min_deg=True)
            start_scores.append((len(p), start))
        start_scores.sort(reverse=True)
        
        for _, start in start_scores[:10]:
            if time.time() > deadline - 2:
                break
            p = _dfs_longest(start, adj, deadline - 2)
            if len(p) > len(best):
                best = p
    
    return best if best else [0]
# EVOLVE-BLOCK-END
