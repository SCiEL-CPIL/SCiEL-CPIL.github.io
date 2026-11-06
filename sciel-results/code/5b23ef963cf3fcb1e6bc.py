import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 58.0  # safety margin for 60s limit
    
    best_path = []
    
    def is_valid_path(path):
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    def greedy_extend(start, strategy='min_degree', rng=None):
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            if strategy == 'min_degree':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'max_degree':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            elif strategy == 'random' and rng is not None:
                v = rng.choice(list(candidates))
            else:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    def try_extend_both_ends(path):
        path = list(path)
        seen = set(path)
        changed = True
        while changed:
            changed = False
            if path:
                candidates = adj[path[0]] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.insert(0, v)
                    seen.add(v)
                    changed = True
            if path:
                candidates = adj[path[-1]] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    seen.add(v)
                    changed = True
        return path
    
    def try_extend_both_ends_max(path):
        path = list(path)
        seen = set(path)
        changed = True
        while changed:
            changed = False
            if path:
                candidates = adj[path[0]] - seen
                if candidates:
                    v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
                    path.insert(0, v)
                    seen.add(v)
                    changed = True
            if path:
                candidates = adj[path[-1]] - seen
                if candidates:
                    v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
                    path.append(v)
                    seen.add(v)
                    changed = True
        return path
    
    def two_opt_improve(path):
        path = list(path)
        n_path = len(path)
        if n_path < 3:
            return path
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_path - 1):
                if time.time() > deadline:
                    return path
                for j in range(i + 1, n_path):
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    if not is_valid_path(new_path):
                        continue
                    extended = try_extend_both_ends(new_path)
                    if len(extended) > len(path):
                        path = extended
                        n_path = len(path)
                        improved = True
                        break
                if improved:
                    break
        return path
    
    def local_search(path):
        path = list(path)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_path = len(path)
            for i in range(1, n_path - 1):
                if time.time() > deadline:
                    return path
                if i + 1 < n_path:
                    new_path = path[:i] + [path[i+1], path[i]] + path[i+2:]
                    if is_valid_path(new_path):
                        extended = try_extend_both_ends(new_path)
                        if len(extended) > len(path):
                            path = extended
                            improved = True
                            break
        return path
    
    # Phase 1: Greedy from every vertex with multiple strategies
    strategies = ['min_degree', 'max_degree']
    for strategy in strategies:
        for start in range(n):
            if time.time() > deadline:
                return best_path
            path = greedy_extend(start, strategy)
            path = try_extend_both_ends(path)
            if len(path) > len(best_path):
                best_path = path
    
    # Phase 2: 2-opt improvement on best path
    if time.time() < deadline and len(best_path) > 0:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_path = two_opt_improve(best_path)
            if len(new_path) > len(best_path):
                best_path = new_path
                improved = True
            new_path = local_search(best_path)
            if len(new_path) > len(best_path):
                best_path = new_path
                improved = True
    
    # Phase 3: Edge-pair combination
    if time.time() < deadline:
        for u in range(n):
            if time.time() > deadline:
                break
            for v in sorted(adj[u]):
                if time.time() > deadline:
                    break
                # Build path from u side (avoiding v initially)
                path_u = [u]
                seen_u = {u, v}
                while True:
                    candidates = adj[path_u[-1]] - seen_u
                    if not candidates:
                        break
                    w = min(candidates, key=lambda x: (len(adj[x] - seen_u), x))
                    path_u.append(w)
                    seen_u.add(w)
                
                # Build path from v side (avoiding u initially)
                path_v = [v]
                seen_v = {u, v}
                while True:
                    candidates = adj[path_v[-1]] - seen_v
                    if not candidates:
                        break
                    w = min(candidates, key=lambda x: (len(adj[x] - seen_v), x))
                    path_v.append(w)
                    seen_v.add(w)
                
                # Combine: path_u + [v] + reversed(path_v without v)
                # Actually: path_u ends at some node, we need to connect to v
                # Let's try: reverse path_u, put v, then path_v
                # path_u is [u, ..., last_u], path_v is [v, ..., last_v]
                # Combined: [last_u, ..., u, v, ..., last_v]
                combined = path_u[::-1] + path_v[1:]
                if is_valid_path(combined):
                    extended = try_extend_both_ends(combined)
                    if len(extended) > len(best_path):
                        best_path = extended
    
    # Phase 4: Randomized restarts with Warnsdorff
    rng = random.Random(42)
    if time.time() < deadline:
        # Find largest component for start selection
        visited = [False] * n
        components = []
        for i in range(n):
            if not visited[i]:
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nei in adj[node]:
                        if not visited[nei]:
                            visited[nei] = True
                            stack.append(nei)
                components.append(comp)
        components.sort(key=len, reverse=True)
        start_nodes = components[0] if components and len(components[0]) > 10 else list(range(n))
        
        while time.time() < deadline:
            start = rng.choice(start_nodes)
            path = [start]
            seen = {start}
            while True:
                candidates = adj[path[-1]] - seen
                if not candidates:
                    break
                # Warnsdorff: pick neighbor with fewest unvisited neighbors
                if rng.random() < 0.15:
                    v = rng.choice(list(candidates))
                else:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
            
            path = try_extend_both_ends(path)
            if len(path) > len(best_path):
                best_path = path
    
    # Final improvement pass
    if time.time() < deadline and len(best_path) > 0:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_path = two_opt_improve(best_path)
            if len(new_path) > len(best_path):
                best_path = new_path
                improved = True
            new_path = local_search(best_path)
            if len(new_path) > len(best_path):
                best_path = new_path
                improved = True
    
    return best_path
