"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time
import random
from collections import deque

def _build_adj():
    a = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        a[u].add(v)
        a[v].add(u)
    return a

def _greedy_path(adj, start, strategy, rng=None):
    """Greedy path extension with different neighbor-selection strategies."""
    path = [start]
    seen = {start}
    while True:
        cands = adj[path[-1]] - seen
        if not cands:
            break
        if strategy == 'min_deg':
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(cands))
        elif strategy == 'warnsdorff':
            # Prefer vertices with fewest unvisited neighbors (dead-end avoidance)
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _extend_path_both_ends(adj, path, deadline):
    """Try to extend a path from both ends."""
    if not path:
        return path
    path = list(path)
    seen = set(path)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to extend front
        front_cands = adj[path[0]] - seen
        if front_cands:
            v = min(front_cands, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Try to extend back
        back_cands = adj[path[-1]] - seen
        if back_cands:
            v = min(back_cands, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def _backtrack_search(adj, start, max_depth, rng, deadline, best_len=0):
    """DFS with backtracking using Warnsdorff heuristic, limited by depth and time."""
    best_path = [start]
    best_len = [best_len]
    
    def dfs(path, seen, depth):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len[0]:
            best_path = path[:]
            best_len[0] = len(path)
        if depth >= max_depth:
            return
        # Warnsdorff: sort candidates by number of unvisited neighbors (ascending)
        cands = list(adj[path[-1]] - seen)
        if not cands:
            return
        # Score: fewer unvisited neighbors first (to avoid dead ends), then tie-break randomly
        scored = [(len(adj[w] - seen), rng.random(), w) for w in cands]
        scored.sort()
        for _, _, v in scored:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path

def _iterative_deepening(adj, start, initial_depth, max_depth, rng, deadline, best_len=0):
    """Iterative deepening DFS with Warnsdorff heuristic."""
    best_path = [start]
    best_len = [best_len]
    
    for target_depth in range(initial_depth, max_depth + 1):
        if time.time() > deadline:
            break
        
        def dfs(path, seen, depth):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len[0]:
                best_path = path[:]
                best_len[0] = len(path)
            if depth >= target_depth:
                return
            cands = list(adj[path[-1]] - seen)
            if not cands:
                return
            scored = [(len(adj[w] - seen), rng.random(), w) for w in cands]
            scored.sort()
            for _, _, v in scored:
                if time.time() > deadline:
                    return
                path.append(v)
                seen.add(v)
                dfs(path, seen, depth + 1)
                path.pop()
                seen.discard(v)
        
        dfs([start], {start}, 0)
    
    return best_path

def _find_largest_component(adj):
    """Find the largest connected component."""
    visited = [False] * P['n']
    components = []
    for i in range(P['n']):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)
    components.sort(key=len, reverse=True)
    return components[0] if components else []

def construct():
    deadline = time.time() + 85  # 5s safety margin
    adj = _build_adj()
    best = []
    
    # Strategy 1: Deterministic greedy from every start
    for start in range(P['n']):
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Strategy 2: Randomized restarts with different heuristics
    rng = random.Random(42)
    strategies = ['min_deg', 'max_deg', 'random', 'warnsdorff']
    
    for strat in strategies:
        for start in range(P['n']):
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
    
    # Strategy 3: Backtracking search from promising starts with Warnsdorff
    degs = [(len(adj[i]), i) for i in range(P['n'])]
    degs.sort(reverse=True)
    
    # Find largest component for focus
    largest_comp = _find_largest_component(adj)
    
    for deg, start in degs[:20]:
        if time.time() > deadline:
            break
        max_depth = min(P['n'], 40)
        path = _backtrack_search(adj, start, max_depth, rng, deadline, len(best))
        if len(path) > len(best):
            best = path
    
    # Strategy 4: Iterative deepening from high-degree vertices
    for deg, start in degs[:10]:
        if time.time() > deadline:
            break
        path = _iterative_deepening(adj, start, 10, 50, rng, deadline, len(best))
        if len(path) > len(best):
            best = path
    
    # Strategy 5: Two-end growth from pairs of high-degree vertices
    top_verts = [v for _, v in degs[:15]]
    for i in range(len(top_verts)):
        for j in range(i + 1, len(top_verts)):
            if time.time() > deadline:
                break
            # Create initial path between two high-degree vertices if they're connected
            # Simple approach: start from one, grow towards the other
            path1 = _greedy_path(adj, top_verts[i], 'min_deg')
            path2 = _greedy_path(adj, top_verts[j], 'min_deg')
            # Try to merge by extending from both ends
            combined = _extend_path_both_ends(adj, path1, deadline)
            if len(combined) > len(best):
                best = combined
    
    # Final: extend the best path found
    if best:
        extended = _extend_path_both_ends(adj, best, deadline)
        if len(extended) > len(best):
            best = extended
    
    # Final polish: try random restarts one more time if time permits
    for start in range(P['n']):
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, 'random', rng)
            path = _extend_path_both_ends(adj, path, deadline)
            if len(path) > len(best):
                best = path
    
    return best
# EVOLVE-BLOCK-END
