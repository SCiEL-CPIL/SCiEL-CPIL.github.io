"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

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
        elif strategy == 'weighted':
            # Prefer neighbors with fewer unvisited neighbors (dead-end avoidance)
            weights = [1.0 / max(1, len(adj[w] - seen)) for w in cands]
            v = sorted(cands)[weights.index(max(weights))]
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _extend_path(adj, path, deadline):
    """Try to extend a path from both ends by finding insertable vertices."""
    if not path:
        return path
    seen = set(path)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to extend the front
        front = path[0]
        for v in sorted(adj[front]):
            if v not in seen:
                # Check if v can connect to some path vertex (for potential reordering)
                # For simple extension, just check if v connects to front
                path.insert(0, v)
                seen.add(v)
                changed = True
                break
        # Try to extend the back
        if not changed:
            back = path[-1]
            for v in sorted(adj[back]):
                if v not in seen:
                    path.append(v)
                    seen.add(v)
                    changed = True
                    break
    # Try to insert vertices between consecutive path vertices
    if time.time() < deadline:
        i = 0
        while i < len(path) - 1 and time.time() < deadline:
            a, b = path[i], path[i+1]
            # Find a vertex w such that a-w-b forms a path and w is not in path
            for w in sorted(adj[a]):
                if w not in seen and w in adj[b]:
                    path.insert(i+1, w)
                    seen.add(w)
                    i += 1  # Stay at same position to try more insertions
                    changed = True
                    break
            i += 1
    return path

def _backtrack_search(adj, start, max_depth, rng, deadline):
    """DFS with backtracking, limited by depth and time. Uses Warnsdorff-like heuristic."""
    best_path = [start]
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = path[:]
        if depth >= max_depth:
            return
        cands = list(adj[path[-1]] - seen)
        if not cands:
            return
        # Order by number of unvisited neighbors (Warnsdorff's rule) - prefer fewer options first
        cands.sort(key=lambda w: (len(adj[w] - seen), w))
        for v in cands:
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
    visited = set()
    components = []
    for start in range(P['n']):
        if start not in visited:
            comp = []
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)
            components.append(comp)
    return max(components, key=len) if components else []

def construct():
    deadline = time.time() + 80  # 10s safety margin
    adj = _build_adj()
    best = []
    
    # Find the largest component - restrict search to it
    largest_comp = _find_largest_component(adj)
    comp_set = set(largest_comp)
    
    # Strategy 1: Deterministic greedy from every start in largest component
    for start in largest_comp:
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Strategy 2: Randomized restarts with different heuristics
    rng = random.Random(42)
    strategies = ['min_deg', 'max_deg', 'random', 'weighted']
    
    for strat in strategies:
        for start in largest_comp:
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
    
    # Strategy 3: Backtracking search from promising starts in largest component
    degs = [(len(adj[i]), i) for i in largest_comp]
    degs.sort(reverse=True)
    
    # Use deeper search since we're restricted to the component
    for deg, start in degs[:15]:
        if time.time() > deadline:
            break
        max_depth = min(len(largest_comp), 50)  # Allow deeper search
        path = _backtrack_search(adj, start, max_depth, rng, deadline)
        if len(path) > len(best):
            best = path
    
    # Strategy 4: Try to extend the best path found so far
    if best and time.time() < deadline:
        extended = _extend_path(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
    
    # Strategy 5: More randomized searches if time permits
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            start = rng.choice(largest_comp)
            strat = rng.choice(strategies)
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
            # Try to extend
            if time.time() < deadline:
                extended = _extend_path(adj, best[:], deadline)
                if len(extended) > len(best):
                    best = extended
    
    # Final: more backtracking if time permits
    if time.time() < deadline:
        for deg, start in degs[:10]:
            if time.time() > deadline:
                break
            max_depth = min(len(largest_comp), 60)
            path = _backtrack_search(adj, start, max_depth, rng, deadline)
            if len(path) > len(best):
                best = path
            # Extend
            if time.time() < deadline:
                extended = _extend_path(adj, best[:], deadline)
                if len(extended) > len(best):
                    best = extended
    
    return best
# EVOLVE-BLOCK-END
