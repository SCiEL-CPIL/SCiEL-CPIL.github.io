"""Find longest simple path in the pinned graph using multiple strategies."""
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_components():
    """Find all connected components."""
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

def greedy_path_from(start, adj, seen=None):
    """Greedy path starting from start, choosing neighbor with fewest remaining options."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        # Choose the candidate with the fewest unvisited neighbors (Warnsdorff-like)
        v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, adj):
    """Try to extend path from both ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending from the start
        if path:
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        # Try extending from the end
        if path:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path

def try_insert(path, adj):
    """Try to insert unvisited vertices into the path."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    
    improved = True
    while improved and unvisited:
        improved = False
        # Try to insert each unvisited vertex
        for w in list(unvisited):
            # Check if w can be inserted between any two consecutive nodes in path
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                if w in adj[u] and w in adj[v]:
                    # Insert w
                    path.insert(i+1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break # w is now visited, move to next w
            
            if improved:
                break # Restart the loop as path changed
    
    return path

def backtracking_extend(path, adj, max_depth=50):
    """Try to extend path using limited backtracking."""
    path = list(path)
    seen = set(path)
    
    def dfs(cur, depth):
        if depth >= max_depth:
            return False
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return False
        # Sort by degree heuristic
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for v in candidates:
            path.append(v)
            seen.add(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            seen.remove(v)
        return False
    
    # Try extending from the end
    if path:
        dfs(path[-1], 0)
    
    # Try extending from the start
    if path:
        start = path[0]
        candidates = [nb for nb in adj[start] if nb not in seen]
        for v in candidates:
            path.insert(0, v)
            seen.add(v)
            # Continue extending from the new start
            def dfs_start(cur, depth):
                if depth >= max_depth:
                    return False
                cands = [nb for nb in adj[cur] if nb not in seen]
                if not cands:
                    return False
                cands.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                for nv in cands:
                    path.insert(0, nv)
                    seen.add(nv)
                    if dfs_start(nv, depth + 1):
                        return True
                    path.pop(0)
                    seen.remove(nv)
                return False
            dfs_start(v, 0)
            # Revert
            if path and path[0] == v:
                path.pop(0)
                seen.remove(v)
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Find components
    comps = find_components()
    
    # Sort components by size
    comps.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0

    if not comps:
        return [0]
        
    largest_comp = comps[0]

    # Strategy 1: Greedy + Insertion from every start in the largest component
    for start in largest_comp:
        if time.time() > deadline - 10:
            break
        path = greedy_path_from(start, ADJ)
        path = extend_both_ends(path, ADJ)
        path = try_insert(path, ADJ)
        path = extend_both_ends(path, ADJ)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Strategy 2: Randomized Greedy + Insertion
    rng = random.Random(42)
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        start = rng.choice(largest_comp)
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            candidates = [nb for nb in ADJ[cur] if nb not in seen]
            if not candidates:
                break
            # Heuristic: prefer nodes with fewer unvisited neighbors
            # But add some randomness
            if len(candidates) > 1 and rng.random() < 0.3:
                v = rng.choice(candidates)
            else:
                v = min(candidates, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
            path.append(v)
            seen.add(v)
            
        path = extend_both_ends(path, ADJ)
        path = try_insert(path, ADJ)
        path = extend_both_ends(path, ADJ)
        
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
            
    # Strategy 3: Backtracking extension on the best path found so far
    if best_path:
        if time.time() < deadline - 5:
            extended = backtracking_extend(best_path, ADJ, max_depth=50)
            if len(extended) > best_len:
                best_path = extended
                best_len = len(extended)
    
    # Strategy 4: More randomized restarts with insertion
    for seed in range(1000):
        if time.time() > deadline - 2:
            break
        rng = random.Random(seed)
        start = rng.choice(largest_comp)
        # Randomized greedy
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            candidates = [nb for nb in ADJ[cur] if nb not in seen]
            if not candidates:
                break
            # Pick randomly or by heuristic
            if len(candidates) < 3:
                v = candidates[0]
            else:
                # Weighted by inverse degree
                weights = [1.0 / (len(ADJ[w]) + 1) for w in candidates]
                v = rng.choices(candidates, weights=weights)[0]
            path.append(v)
            seen.add(v)
        
        path = extend_both_ends(path, ADJ)
        path = try_insert(path, ADJ)
        path = extend_both_ends(path, ADJ)
        
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Final attempt: extend best path more aggressively
    if best_path and time.time() < deadline - 1:
        extended = extend_both_ends(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(extended)
        extended = try_insert(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(extended)

    # Ensure we return a valid non-empty path
    if not best_path:
        # Fallback: return a single vertex
        best_path = [0]
    
    # Validate the path
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            # Path is invalid, find a valid one
            for start in range(N):
                path = greedy_path_from(start, ADJ)
                if len(path) > 0:
                    best_path = path
                    break
            break
    
    return best_path
