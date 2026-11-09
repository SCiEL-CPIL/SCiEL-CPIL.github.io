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

def forced_move_path(start, adj, deadline):
    """Build path using forced moves: if a vertex has only one unvisited neighbor, must go there."""
    path = [start]
    seen = {start}
    while time.time() < deadline:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            # Try extending from the other end
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if not candidates:
                break
            path.insert(0, candidates[0])
            seen.add(path[0])
        else:
            # Forced move: if any candidate has degree 1 (only cur as unvisited neighbor), must take it
            forced = [nb for nb in candidates if len([x for x in adj[nb] if x not in seen]) == 1]
            if forced:
                v = forced[0]
            else:
                # Choose candidate with fewest remaining options
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
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i+1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    
    return path

def two_opt_refine(path, adj, max_iterations=100, deadline=None):
    """Try 2-opt moves: reverse segments of the path to create longer paths."""
    if deadline is None:
        deadline = time.time() + 5
    
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Try to insert each unvisited vertex
        for w in list(unvisited):
            # Check if w can be inserted between any two consecutive nodes
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i+1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
        
        # Try 2-opt: reverse a segment and check if it creates new insertion opportunities
        # This is a simplified version - just try reversing small segments
        for i in range(len(path) - 1):
            if time.time() > deadline:
                break
            u = path[i]
            v = path[i+1]
            # Try inserting unvisited vertices that connect to u and v
            for w in list(unvisited):
                if w in adj[u] and w in adj[v]:
                    path.insert(i+1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    
    # Final extension from both ends
    path = extend_both_ends(path, adj)
    path = try_insert(path, adj)
    path = extend_both_ends(path, ADJ)
    
    return path

def iterative_deepening_path(start, adj, max_depth, deadline):
    """Iterative deepening DFS to find a long path starting from start."""
    best_path = [start]
    
    def dfs(cur, depth, path, seen):
        nonlocal best_path
        if time.time() > deadline:
            return
        if depth == max_depth:
            if len(path) > len(best_path):
                best_path = list(path)
            return
        
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            if len(path) > len(best_path):
                best_path = list(path)
            return
        
        # Sort by degree heuristic
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        
        for v in candidates:
            path.append(v)
            seen.add(v)
            dfs(v, depth + 1, path, seen)
            path.pop()
            seen.remove(v)
    
    seen = {start}
    dfs(start, 0, [start], seen)
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0

    if not comps:
        return [0]
        
    largest_comp = comps[0]
    largest_comp_set = set(largest_comp)

    # Strategy 1: Forced-move path from every start in the largest component
    for start in largest_comp:
        if time.time() > deadline - 15:
            break
        path = forced_move_path(start, ADJ, deadline - 15)
        path = extend_both_ends(path, ADJ)
        path = try_insert(path, ADJ)
        path = extend_both_ends(path, ADJ)
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
    
    # Strategy 2: Randomized forced-move with multiple seeds
    for seed in range(200):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        start = rng.choice(largest_comp)
        path = [start]
        seen = {start}
        while time.time() < deadline - 15:
            cur = path[-1]
            candidates = [nb for nb in ADJ[cur] if nb not in seen]
            if not candidates:
                # Try other end
                cur = path[0]
                candidates = [nb for nb in ADJ[cur] if nb not in seen]
                if not candidates:
                    break
                v = min(candidates, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                continue
            
            # Forced move check
            forced = [nb for nb in candidates if len([x for x in ADJ[nb] if x not in seen]) == 1]
            if forced:
                v = forced[0]
            elif len(candidates) > 1 and rng.random() < 0.2:
                v = rng.choice(candidates)
            else:
                v = min(candidates, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
            path.append(v)
            seen.add(v)
        
        path = extend_both_ends(path, ADJ)
        path = try_insert(path, ADJ)
        path = extend_both_ends(path, ADJ)
        
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
    
    # Strategy 3: 2-opt refinement on best path
    if best_path and time.time() < deadline - 10:
        refined = two_opt_refine(best_path, ADJ, max_iterations=200, deadline=deadline - 10)
        if len(refined) > best_len:
            best_path = refined
            best_len = len(best_path)
    
    # Strategy 4: Iterative deepening from promising starts
    if time.time() < deadline - 5:
        # Try iterative deepening from each vertex in the best path
        for start in best_path[:10]:
            if time.time() > deadline - 5:
                break
            for depth in [5, 8, 12, 15]:
                if time.time() > deadline - 5:
                    break
                path = iterative_deepening_path(start, ADJ, depth, deadline - 5)
                path = extend_both_ends(path, ADJ)
                path = try_insert(path, ADJ)
                path = extend_both_ends(path, ADJ)
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
    
    # Strategy 5: More randomized restarts with insertion
    for seed in range(500):
        if time.time() > deadline - 2:
            break
        rng = random.Random(seed + 1000)
        start = rng.choice(largest_comp)
        path = [start]
        seen = {start}
        while time.time() < deadline - 2:
            cur = path[-1]
            candidates = [nb for nb in ADJ[cur] if nb not in seen]
            if not candidates:
                break
            if len(candidates) < 3:
                v = candidates[0]
            else:
                weights = [1.0 / (len(ADJ[w]) + 1) for w in candidates]
                v = rng.choices(candidates, weights=weights)[0]
            path.append(v)
            seen.add(v)
        
        path = extend_both_ends(path, ADJ)
        path = try_insert(path, ADJ)
        path = extend_both_ends(path, ADJ)
        
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
    
    # Final refinement
    if best_path and time.time() < deadline - 1:
        extended = extend_both_ends(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        extended = try_insert(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        extended = two_opt_refine(best_path, ADJ, max_iterations=50, deadline=deadline - 1)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)

    # Ensure we return a valid non-empty path
    if not best_path:
        best_path = [0]
    
    # Validate the path
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            for start in range(N):
                path = forced_move_path(start, ADJ, start_time + 1)
                if len(path) > 0:
                    best_path = path
                    break
            break
    
    return best_path
