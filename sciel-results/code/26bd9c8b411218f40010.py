"""Find longest simple path in pinned G(100, 1.8/100) graph using randomized backtracking."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
    visited = [False] * n
    comps = []
    for start in range(n):
        if not visited[start]:
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append((len(comp), comp))
    comps.sort(key=lambda x: -x[0])
    return comps

def _randomized_backtrack(adj, start, n, deadline, best_path, best_len):
    """
    Randomized DFS with backtracking.
    Returns (best_path, best_len) found so far.
    """
    path = [start]
    seen = {start}
    
    def dfs():
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        cur = path[-1]
        candidates = [v for v in adj[cur] if v not in seen]
        if not candidates:
            return
        
        # Heuristic: prioritize neighbors with fewest unvisited connections (dead-ends)
        # But add some randomness to escape local optima
        candidates.sort(key=lambda v: (len(adj[v] - seen), random.random()))
        
        # Limit branching factor to prevent explosion
        max_branch = 5
        if len(candidates) > max_branch:
            # Pick a mix: some lowest degree, some random
            chosen = candidates[:max(1, max_branch//2)]
            remaining = candidates[max(1, max_branch//2):]
            if remaining:
                chosen.extend(random.sample(remaining, min(max_branch - len(chosen), len(remaining))))
            candidates = chosen
            
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs()
            path.pop()
            seen.discard(v)
            
    dfs()
    return best_path, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    n = P['n']
    adj = _build_adj()
    
    # Find components
    comps = _find_components(adj, n)
    
    if not comps or comps[0][0] == 0:
        return [0]
    
    # Focus on the largest component
    largest_comp = comps[0][1]
    comp_set = set(largest_comp)
    
    best_path = []
    best_len = 0
    
    # Initial greedy to get a decent starting point
    # Greedy: always go to neighbor with fewest unvisited neighbors
    def greedy_path(start):
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            candidates = [v for v in adj[cur] if v not in seen]
            if not candidates:
                break
            best_v = min(candidates, key=lambda v: (len(adj[v] - seen), v))
            path.append(best_v)
            seen.add(best_v)
        return path
    
    # Try greedy from a few starts to initialize best_path
    random.seed(748)
    starts_to_try = random.sample(largest_comp, min(5, len(largest_comp)))
    for s in starts_to_try:
        p = greedy_path(s)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Randomized backtracking from various starts
    # Use time budget to run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose start: mix of random, low-degree, and high-degree
        if i % 3 == 0:
            start = random.choice(largest_comp)
        elif i % 3 == 1:
            # Low degree in component
            degs = [(len([u for u in adj[v] if u in comp_set]), v) for v in largest_comp]
            degs.sort()
            start = degs[random.randint(0, min(9, len(degs)-1))][1]
        else:
            # High degree
            degs = [(len([u for u in adj[v] if u in comp_set]), v) for v in largest_comp]
            degs.sort(reverse=True)
            start = degs[random.randint(0, min(9, len(degs)-1))][1]
        
        # Run randomized backtracking with a small time slice
        slice_time = min(1.5, deadline - time.time())
        if slice_time <= 0:
            break
        
        local_deadline = time.time() + slice_time
        best_path, best_len = _randomized_backtrack(adj, start, n, local_deadline, best_path, best_len)
        
        # If we found a path that covers the whole component, we can stop
        if best_len == len(largest_comp):
            break
    
    # Final polish: try to extend the best path found
    if best_path and best_len < len(largest_comp):
        # Try to insert missing vertices
        seen = set(best_path)
        missing = [v for v in largest_comp if v not in seen]
        
        # Try to extend from ends
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try front
            if best_path:
                cur = best_path[0]
                cands = [v for v in adj[cur] if v not in seen]
                if cands:
                    v = min(cands, key=lambda x: len(adj[x] - seen))
                    best_path.insert(0, v)
                    seen.add(v)
                    changed = True
            # Try back
            if best_path:
                cur = best_path[-1]
                cands = [v for v in adj[cur] if v not in seen]
                if cands:
                    v = min(cands, key=lambda x: len(adj[x] - seen))
                    best_path.append(v)
                    seen.add(v)
                    changed = True
        
        # Try to insert between nodes
        i = 0
        while i < len(best_path) - 1 and time.time() < deadline:
            a, b = best_path[i], best_path[i+1]
            cands = [v for v in adj[a] if v in adj[b] and v not in seen]
            if cands:
                v = min(cands, key=lambda x: len(adj[x] - seen))
                best_path.insert(i+1, v)
                seen.add(v)
                # Don't increment i
            else:
                i += 1
        
        best_len = len(best_path)

    if not best_path:
        # Fallback
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
            
    return best_path
