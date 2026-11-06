"""Find longest simple path in pinned G(100, 1.8/100) graph using enhanced randomized backtracking."""
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        adj_list[i] = list(adj[i])
    return adj, adj_list

def _largest_component_size(adj, n, comp_set):
    """Return size of the largest connected component in the induced subgraph on comp_set."""
    if not comp_set:
        return 0
    visited = set()
    best = 0
    for start in comp_set:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            for nb in adj[v]:
                if nb in comp_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if size > best:
            best = size
    return best

def _greedy_path(adj, adj_list, start, comp_set):
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [v for v in adj_list[cur] if v not in seen and v in comp_set]
        if not candidates:
            break
        # Pick neighbor with fewest unvisited neighbors in component (dead-end first)
        best_v = min(candidates, key=lambda v: (len([u for u in adj_list[v] if u not in seen and u in comp_set]), v))
        path.append(best_v)
        seen.add(best_v)
    return path

def _randomized_backtrack(adj, adj_list, start, comp_set, deadline, best_path, best_len):
    path = [start]
    seen = {start}
    comp_size = len(comp_set)
    
    def dfs():
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        cur = path[-1]
        candidates = [v for v in adj_list[cur] if v not in seen and v in comp_set]
        if not candidates:
            return
        
        # Pruning: upper bound on achievable path length
        unvisited = comp_set - seen
        if len(path) + len(unvisited) <= best_len:
            return
        
        # Stronger pruning: largest component of unvisited + 1 (current endpoint can connect to one)
        # Actually, the path can extend into at most one component of unvisited from cur
        # But we can also have the path end at cur and not extend further
        # Upper bound: len(path) + (size of largest component of unvisited that is adjacent to cur or reachable)
        # For simplicity, use: len(path) + _largest_component_size(adj, n, unvisited)
        # But this is expensive. Use a cheaper bound: if unvisited is empty, done.
        # Let's use a moderate check: if len(path) + len(unvisited) <= best_len, prune (already done above)
        
        # Heuristic ordering: prefer neighbors that are dead-ends (few unvisited neighbors)
        # but also avoid isolating large unvisited regions
        def score(v):
            unvisited_neighbors = [u for u in adj_list[v] if u not in seen and u in comp_set]
            # After visiting v, the unvisited set loses v
            # Check if v is an articulation point for unvisited
            # Cheap proxy: count unvisited neighbors of v
            return (len(unvisited_neighbors), random.random())
        
        candidates.sort(key=score)
        
        # Limit branching: take top few by score, plus some random
        max_branch = 4
        if len(candidates) > max_branch:
            chosen = candidates[:max_branch]
            # Add one random from the rest for diversity
            rest = candidates[max_branch:]
            if rest:
                chosen.append(random.choice(rest))
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

def _polish_path(adj, adj_list, path, comp_set, deadline):
    """Try to extend and insert vertices into the path."""
    seen = set(path)
    missing = [v for v in comp_set if v not in seen]
    if not missing:
        return path
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try extending from front
        if path:
            cur = path[0]
            cands = [v for v in adj_list[cur] if v not in seen and v in comp_set]
            if cands:
                v = min(cands, key=lambda x: len([u for u in adj_list[x] if u not in seen and u in comp_set]))
                path.insert(0, v)
                seen.add(v)
                missing.remove(v)
                changed = True
        # Try extending from back
        if path:
            cur = path[-1]
            cands = [v for v in adj_list[cur] if v not in seen and v in comp_set]
            if cands:
                v = min(cands, key=lambda x: len([u for u in adj_list[x] if u not in seen and u in comp_set]))
                path.append(v)
                seen.add(v)
                missing.remove(v)
                changed = True
    
    # Try inserting missing vertices between consecutive pairs
    for v in missing:
        if time.time() > deadline:
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            if v in adj[a] and v in adj[b]:
                path.insert(i+1, v)
                seen.add(v)
                missing.remove(v)
                break
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = P['n']
    adj, adj_list = _build_adj()
    
    # Find connected components
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
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    
    if not comps or not comps[0]:
        return [0]
    
    largest_comp = comps[0]
    comp_set = set(largest_comp)
    
    best_path = []
    best_len = 0
    
    # Initial greedy from multiple starts
    random.seed(748)
    starts_to_try = random.sample(largest_comp, min(10, len(largest_comp)))
    for s in starts_to_try:
        p = _greedy_path(adj, adj_list, s, comp_set)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Randomized backtracking with multiple restarts
    num_restarts = 80
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            break
        
        # Choose start vertex
        if i % 4 == 0:
            start = random.choice(largest_comp)
        elif i % 4 == 1:
            # Low degree in component
            degs = [(len([u for u in adj_list[v] if u in comp_set]), v) for v in largest_comp]
            degs.sort()
            start = degs[random.randint(0, min(14, len(degs)-1))][1]
        elif i % 4 == 2:
            # High degree in component
            degs = [(len([u for u in adj_list[v] if u in comp_set]), v) for v in largest_comp]
            degs.sort(reverse=True)
            start = degs[random.randint(0, min(14, len(degs)-1))][1]
        else:
            # Random
            start = random.choice(largest_comp)
        
        # Time slice for this restart
        slice_time = min(1.0, remaining_time * 0.8)
        if slice_time <= 0.1:
            break
        
        local_deadline = time.time() + slice_time
        best_path, best_len = _randomized_backtrack(adj, adj_list, start, comp_set, local_deadline, best_path, best_len)
        
        if best_len == len(largest_comp):
            break
    
    # Final polish
    if best_path and best_len < len(largest_comp) and time.time() < deadline:
        best_path = _polish_path(adj, adj_list, best_path, comp_set, deadline)
        best_len = len(best_path)
    
    if not best_path:
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
            
    return best_path
