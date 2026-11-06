"""Rewritten program: stronger longest-path heuristic with time budget."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_path = []
    
    def greedy_extend(start, strategy='min_degree'):
        """Greedy path extension from start vertex."""
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
            elif strategy == 'random':
                import random
                v = random.choice(list(candidates))
            else:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    def try_extend_both_ends(path):
        """Try to extend a path from both ends."""
        path = list(path)
        seen = set(path)
        changed = True
        while changed:
            changed = False
            # Try extending from start
            if path:
                candidates = adj[path[0]] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.insert(0, v)
                    seen.add(v)
                    changed = True
            # Try extending from end
            if path:
                candidates = adj[path[-1]] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    seen.add(v)
                    changed = True
        return path
    
    def two_opt_improve(path):
        """Try 2-opt style rearrangements to enable further extension."""
        path = list(path)
        n_path = len(path)
        if n_path < 3:
            return path
        improved = True
        while improved:
            improved = False
            for i in range(n_path - 1):
                for j in range(i + 1, n_path):
                    # Check if reversing segment i..j creates a valid path
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    # Check validity
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k+1] not in adj[new_path[k]]:
                            valid = False
                            break
                    if valid:
                        # Try to extend
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
        """Local search: try swapping adjacent vertices or small rearrangements."""
        path = list(path)
        improved = True
        while improved:
            improved = False
            n_path = len(path)
            for i in range(1, n_path - 1):
                # Try swapping path[i] with path[i+1] if valid
                if i + 1 < n_path:
                    new_path = path[:i] + [path[i+1], path[i]] + path[i+2:]
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k+1] not in adj[new_path[k]]:
                            valid = False
                            break
                    if valid:
                        extended = try_extend_both_ends(new_path)
                        if len(extended) > len(path):
                            path = extended
                            improved = True
                            break
        return path
    
    # Phase 1: Try greedy from every vertex with different strategies
    strategies = ['min_degree', 'max_degree']
    for strategy in strategies:
        for start in range(n):
            if time.time() > deadline:
                break
            path = greedy_extend(start, strategy)
            if len(path) > len(best_path):
                best_path = path
        if time.time() > deadline:
            break
    
    # Phase 2: Try extending from both ends and local search on best paths
    if time.time() <= deadline:
        for start in range(n):
            if time.time() > deadline:
                break
            for strategy in ['min_degree', 'max_degree']:
                path = greedy_extend(start, strategy)
                path = try_extend_both_ends(path)
                if len(path) > len(best_path):
                    best_path = path
    
    # Phase 3: Local search on the best path found
    if time.time() <= deadline and len(best_path) > 0:
        improved = True
        while improved and time.time() <= deadline:
            improved = False
            # Try 2-opt
            new_path = two_opt_improve(best_path)
            if len(new_path) > len(best_path):
                best_path = new_path
                improved = True
            # Try local search
            new_path = local_search(best_path)
            if len(new_path) > len(best_path):
                best_path = new_path
                improved = True
    
    # Phase 4: More aggressive search - try all pairs as internal edges
    if time.time() <= deadline:
        for u in range(n):
            for v in adj[u]:
                if time.time() > deadline:
                    break
                # Try path that goes through edge u-v
                # Greedy from u avoiding v initially, then connect
                path_from_u = [u]
                seen = {u, v}
                while True:
                    candidates = adj[path_from_u[-1]] - seen
                    if not candidates:
                        break
                    w = min(candidates, key=lambda x: (len(adj[x] - seen), x))
                    path_from_u.append(w)
                    seen.add(w)
                
                path_from_v = [v]
                seen2 = {u, v}
                while True:
                    candidates = adj[path_from_v[-1]] - seen2
                    if not candidates:
                        break
                    w = min(candidates, key=lambda x: (len(adj[x] - seen2), x))
                    path_from_v.append(w)
                    seen2.add(w)
                
                # Combine: path_from_u reversed + path_from_v
                combined = path_from_u[::-1] + path_from_v
                # Validate
                valid = True
                for k in range(len(combined) - 1):
                    if combined[k+1] not in adj[combined[k]]:
                        valid = False
                        break
                if valid and len(combined) > len(best_path):
                    best_path = combined
            if time.time() > deadline:
                break
    
    # Final: ensure we have at least one vertex
    if not best_path:
        best_path = [0]
    
    return best_path
