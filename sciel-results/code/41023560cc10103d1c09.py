"""Find longest simple path in the pinned graph using forced-move and multi-heuristic strategies."""
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

def forced_move_path(start, adj, rng, heuristic='warnsdorff'):
    """Build path using forced moves when only one option, otherwise use heuristic."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        if len(candidates) == 1:
            path.append(candidates[0])
            seen.add(candidates[0])
        else:
            if heuristic == 'warnsdorff':
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
            elif heuristic == 'max_degree':
                v = max(candidates, key=lambda w: (len(adj[w]), w))
            elif heuristic == 'random':
                v = rng.choice(candidates)
            elif heuristic == 'min_degree':
                v = min(candidates, key=lambda w: (len(adj[w]), w))
            else:
                v = rng.choice(candidates)
            path.append(v)
            seen.add(v)
    return path

def extend_both_ends(path, adj):
    """Try to extend path from both ends greedily."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        if path:
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        if path:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path

def try_insert(path, adj, deadline):
    """Try to insert unvisited vertices into the path at valid positions."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        for v in list(unvisited):
            if time.time() > deadline - 2:
                break
            # Try inserting v between path[i] and path[i+1]
            for i in range(len(path) - 1):
                if path[i] in ADJ[v] and path[i+1] in ADJ[v]:
                    new_path = path[:i+1] + [v] + path[i+1:]
                    seen.add(v)
                    unvisited.discard(v)
                    path = new_path
                    improved = True
                    break
            # Try inserting v at the start
            if not improved and path and path[0] in ADJ[v]:
                new_path = [v] + path
                seen.add(v)
                unvisited.discard(v)
                path = new_path
                improved = True
            # Try inserting v at the end
            if not improved and path and path[-1] in ADJ[v]:
                new_path = path + [v]
                seen.add(v)
                unvisited.discard(v)
                path = new_path
                improved = True
    return path

def two_opt_refine(path, adj, deadline):
    """Try 2-opt style moves: reverse a segment if it maintains validity and allows extension."""
    path = list(path)
    seen = set(path)
    
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        n = len(path)
        for i in range(n - 1):
            if time.time() > deadline - 2:
                break
            for j in range(i + 2, n):
                if time.time() > deadline - 2:
                    break
                # Check if reversing path[i+1..j] maintains adjacency
                # Need path[i] - path[j] and path[i+1] - path[j+1] (if j < n-1)
                if path[i] in ADJ[path[j]]:
                    if j == n - 1 or path[i+1] in ADJ[path[j+1]]:
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        # Validate the new path
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k+1] not in ADJ[new_path[k]]:
                                valid = False
                                break
                        if valid:
                            path = new_path
                            improved = True
                            break
            if improved:
                break
    return path

def backtracking_extend(path, adj, max_depth, deadline):
    """Try to extend path using limited backtracking."""
    path = list(path)
    seen = set(path)
    
    def dfs(cur, depth):
        if depth >= max_depth or time.time() > deadline - 2:
            return False
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return False
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for v in candidates:
            path.append(v)
            seen.add(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            seen.remove(v)
        return False
    
    if path:
        dfs(path[-1], 0)
    
    if path:
        start = path[0]
        candidates = [nb for nb in adj[start] if nb not in seen]
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for v in candidates:
            path.insert(0, v)
            seen.add(v)
            
            def dfs_start(cur, depth):
                if depth >= max_depth or time.time() > deadline - 2:
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
            
            if path and path[0] == v:
                path.pop(0)
                seen.remove(v)
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    heuristics = ['warnsdorff', 'max_degree', 'min_degree', 'random']
    
    # Strategy 1: Forced-move paths with different heuristics from every start
    for comp in comps:
        if len(comp) <= len(best_path):
            break
        for start in comp:
            if time.time() > deadline - 20:
                break
            for h in heuristics:
                if time.time() > deadline - 20:
                    break
                rng = random.Random(start * 1000 + hash(h) % 10000)
                path = forced_move_path(start, ADJ, rng, heuristic=h)
                path = extend_both_ends(path, ADJ)
                if len(path) > len(best_path):
                    best_path = path
    
    # Strategy 2: Randomized restarts with forced moves
    rng = random.Random(42)
    for comp in comps:
        if len(comp) <= len(best_path):
            break
        for _ in range(200):
            if time.time() > deadline - 30:
                break
            start = rng.choice(comp)
            h = rng.choice(heuristics)
            path = forced_move_path(start, ADJ, rng, heuristic=h)
            path = extend_both_ends(path, ADJ)
            if len(path) > len(best_path):
                best_path = path
    
    # Strategy 3: Refine best path with insertion
    if best_path and time.time() < deadline - 15:
        refined = try_insert(best_path, ADJ, deadline)
        if len(refined) > len(best_path):
            best_path = refined
    
    # Strategy 4: Backtracking extension
    if best_path and time.time() < deadline - 10:
        extended = backtracking_extend(best_path, ADJ, max_depth=40, deadline=deadline)
        if len(extended) > len(best_path):
            best_path = extended
    
    # Strategy 5: More refinement rounds
    if best_path and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            refined = try_insert(best_path, ADJ, deadline)
            if len(refined) > len(best_path):
                best_path = refined
            else:
                break
    
    # Strategy 6: 2-opt refinement
    if best_path and time.time() < deadline - 3:
        refined = two_opt_refine(best_path, ADJ, deadline)
        if len(refined) > len(best_path):
            best_path = refined
    
    # Final extension
    if best_path and time.time() < deadline - 1:
        extended = extend_both_ends(best_path, ADJ)
        if len(extended) > len(best_path):
            best_path = extended
    
    # Ensure valid non-empty path
    if not best_path:
        best_path = [0]
    
    # Validate
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            for start in range(N):
                path = forced_move_path(start, ADJ, random.Random(0), 'warnsdorff')
                if len(path) > 0:
                    best_path = path
                    break
            break
    
    return best_path
