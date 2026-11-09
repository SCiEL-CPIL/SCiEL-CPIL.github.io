"""Find longest simple path in the pinned graph.

Combines Iterative Deepening DFS (IDDFS) with Warnsdorff heuristic for deep search
and a robust refinement pipeline (extend both ends, insert unvisited vertices)
to maximize the simple path length in the pinned graph.
"""
import time
import random

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
    """Greedy path starting from start (Warnsdorff: fewest remaining options)."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, adj):
    """Try to extend path from both ends (Warnsdorff)."""
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

def try_insert(path, adj):
    """Try to insert unvisited vertices between consecutive path nodes."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and unvisited:
        improved = False
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i + 1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    return path

def refine(path, adj, deadline=None):
    """Polish a candidate path with the refinement pipeline."""
    if deadline is None:
        deadline = time.time() + 5  # Default 5s budget
    
    # Multiple passes of refinement
    for _ in range(3):
        if time.time() > deadline:
            break
        path = extend_both_ends(path, adj)
        path = try_insert(path, adj)
        path = extend_both_ends(path, adj)
    
    # Try backtracking extension for more aggressive improvement
    if time.time() < deadline - 1:
        extended = backtracking_extend(path, adj, deadline)
        if len(extended) > len(path):
            path = extended
    
    # Final extension pass
    path = extend_both_ends(path, adj)
    
    return path

def backtracking_extend(path, adj, deadline, max_steps=50000):
    """
    Extend a path using backtracking at both ends.
    Tries to add vertices to either end of the path using DFS with backtracking.
    """
    path = list(path)
    seen = set(path)
    steps = 0
    
    def try_extend_end(cur_end, is_start):
        """Try to extend from one end of the path using backtracking."""
        nonlocal steps, path, seen
        if time.time() > deadline:
            return False
        
        cur = path[0] if is_start else path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return False
        
        # Sort by Warnsdorff: fewest remaining options first
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        
        for v in candidates:
            steps += 1
            if steps > max_steps or time.time() > deadline:
                break
            
            # Try adding v
            if is_start:
                path.insert(0, v)
            else:
                path.append(v)
            seen.add(v)
            
            # Try to extend further from this new end
            if try_extend_end(v, is_start):
                return True
            
            # Backtrack
            if is_start:
                path.pop(0)
            else:
                path.pop()
            seen.remove(v)
        
        return False
    
    # Try extending from both ends
    try_extend_end(path[0], True)
    try_extend_end(path[-1], False)
    
    return path

def forced_move_path(start, adj, deadline):
    """
    Build a path by following forced moves (vertices with only one unvisited neighbor).
    This can quickly build long paths in sparse regions.
    """
    path = [start]
    seen = {start}
    
    while time.time() < deadline:
        # Check both ends for forced moves
        extended = False
        
        # Check start end
        if path:
            cur = path[0]
            unvisited_neighbors = [nb for nb in adj[cur] if nb not in seen]
            if len(unvisited_neighbors) == 1:
                v = unvisited_neighbors[0]
                path.insert(0, v)
                seen.add(v)
                extended = True
        
        # Check end
        if path:
            cur = path[-1]
            unvisited_neighbors = [nb for nb in adj[cur] if nb not in seen]
            if len(unvisited_neighbors) == 1:
                v = unvisited_neighbors[0]
                path.append(v)
                seen.add(v)
                extended = True
        
        if not extended:
            break
    
    return path

def iddfs_search(start, adj, deadline, initial_depth=5, max_depth=None):
    """
    Iterative Deepening DFS with Warnsdorff heuristic and dead-end pruning.
    Returns the best path found within the time budget.
    """
    if max_depth is None:
        max_depth = N - 1
    
    best_path = [start]
    best_len = 1
    found_deeper = False

    current_depth = initial_depth
    
    while current_depth <= max_depth:
        if time.time() > deadline:
            break
            
        path = [start]
        seen = {start}
        found_at_depth = False
        
        def dfs(cur, depth):
            nonlocal found_at_depth, best_path, best_len
            if time.time() > deadline:
                return
            if depth == current_depth:
                best_path = list(path)
                best_len = current_depth
                found_at_depth = True
                return
            
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if not candidates:
                return
            
            # Warnsdorff heuristic: sort by fewest unvisited neighbors
            candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
            
            for v in candidates:
                # Dead-end pruning: if v has only one unvisited neighbor (besides cur),
                # and that neighbor is not the next one we'd go to, skip it
                # Actually, let's just use simple Warnsdorff for now
                
                path.append(v)
                seen.add(v)
                dfs(v, depth + 1)
                path.pop()
                seen.remove(v)
                
                if found_at_depth or time.time() > deadline:
                    return

        dfs(start, 1)
        
        if found_at_depth:
            current_depth += 1
            if best_len == N:
                break
        else:
            break

    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    comps = find_components()
    comps.sort(key=len, reverse=True)
    if not comps:
        return [0]

    largest_comp = comps[0]
    if not largest_comp:
        return [0]

    best_path = list(largest_comp[:1])
    best_len = len(best_path)

    # Phase 1: Greedy initialization
    # Run greedy from all nodes to get a decent starting point and identify promising starts
    promising_starts = []
    for start in largest_comp:
        if time.time() > deadline - 30:
            break
        p = greedy_path_from(start, ADJ)
        p = refine(p, ADJ)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        # Track starts that yielded long paths
        if len(p) > best_len * 0.8: # Heuristic: keep starts that are "good"
            promising_starts.append(start)

    # If we have a very long path already, we might skip some IDDFS
    if best_len == N:
        return best_path

    # Phase 2: IDDFS from promising starts
    # We prioritize starts that are high degree or were good in greedy
    # Sort promising starts by degree descending to find "central" paths first
    unique_promising = list(set(promising_starts))
    # Add top degree nodes from the largest component if not already present
    deg_sorted = sorted(largest_comp, key=lambda v: len(ADJ[v]), reverse=True)
    for v in deg_sorted[:15]:
        if v not in unique_promising:
            unique_promising.append(v)
            
    # Limit the number of IDDFS runs to ensure we finish in time
    # Each IDDFS run can be expensive. 
    # We have ~50s left. Let's allow ~10-20 runs.
    max_iddfs_runs = 12
    
    for start in unique_promising[:max_iddfs_runs]:
        if time.time() > deadline - 15:
            break
            
        # Start IDDFS from a depth close to current best to save time
        start_depth = max(5, best_len - 3)
        
        p = iddfs_search(start, ADJ, deadline - 15, initial_depth=start_depth)
        p = refine(p, ADJ, deadline - 5)
        
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
            
            if best_len == N:
                return best_path
        
        # Also try forced move extension from this start
        if time.time() < deadline - 10:
            fp = forced_move_path(start, ADJ, deadline - 10)
            fp = refine(fp, ADJ, deadline - 5)
            if len(fp) > best_len:
                best_path = fp
                best_len = len(fp)
                if best_len == N:
                    return best_path

    # Phase 3: Final Refinement
    if best_path and time.time() < deadline - 2:
        extended = refine(best_path, ADJ, deadline - 1)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        
        # One more aggressive backtracking extension
        if time.time() < deadline - 1:
            bt_path = backtracking_extend(best_path, ADJ, deadline - 0.5)
            if len(bt_path) > best_len:
                best_path = bt_path
                best_len = len(best_path)

    # Validate
    if not best_path:
        best_path = [0]
    
    # Final sanity check
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            # Fallback to a valid greedy path
            for s in range(N):
                p = greedy_path_from(s, ADJ)
                if p:
                    best_path = p
                    break
            break

    return best_path
