import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    P_data = {
        'n': 100,
        'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    }
    
    n = P_data['n']
    adj = [set() for _ in range(n)]
    for u, v in P_data['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find largest component
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
            
    largest_comp = max(components, key=len)
    largest_comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    best_path = []
    
    def greedy_walk(start, strategy, max_depth=10000):
        """
        strategy: 'warnsdorff', 'random', 'max_deg'
        """
        path = [start]
        visited_set = {start}
        steps = 0
        while steps < max_depth:
            steps += 1
            current = path[-1]
            neighbors = [v for v in adj[current] if v not in visited_set]
            if not neighbors:
                break
            
            if strategy == 'warnsdorff':
                # Prefer neighbors with fewer unvisited neighbors (classic Warnsdorff)
                def score(v):
                    return len([u for u in adj[v] if u not in visited_set])
                neighbors.sort(key=score)
                # Use top-2 for slight diversity but mostly deterministic
                if len(neighbors) > 2:
                    k = 2
                    top_k = neighbors[:k]
                    next_node = top_k[0] if random.random() < 0.7 else top_k[1]
                elif len(neighbors) == 2:
                    next_node = neighbors[0] if random.random() < 0.6 else neighbors[1]
                else:
                    next_node = neighbors[0]
            elif strategy == 'random':
                next_node = random.choice(neighbors)
            elif strategy == 'max_deg':
                def score(v):
                    return len([u for u in adj[v] if u not in visited_set])
                neighbors.sort(key=score, reverse=True)
                next_node = neighbors[0]
            else:
                next_node = random.choice(neighbors)
                
            path.append(next_node)
            visited_set.add(next_node)
            
        return path

    def extend_path(path):
        """Try to extend the path at both ends greedily with Warnsdorff-like preference."""
        path = path[:]
        visited_set = set(path)
        changed = True
        while changed:
            changed = False
            # Extend start
            if path:
                start_node = path[0]
                best_nei = None
                best_score = -1
                for nei in adj[start_node]:
                    if nei not in visited_set:
                        s = len([u for u in adj[nei] if u not in visited_set])
                        if best_score == -1 or s < best_score:
                            best_score = s
                            best_nei = nei
                if best_nei is not None:
                    path.insert(0, best_nei)
                    visited_set.add(best_nei)
                    changed = True
            # Extend end
            if path:
                end_node = path[-1]
                best_nei = None
                best_score = -1
                for nei in adj[end_node]:
                    if nei not in visited_set:
                        s = len([u for u in adj[nei] if u not in visited_set])
                        if best_score == -1 or s < best_score:
                            best_score = s
                            best_nei = nei
                if best_nei is not None:
                    path.append(best_nei)
                    visited_set.add(best_nei)
                    changed = True
        return path

    def local_improve(path, max_iter=100):
        """
        Try to insert unvisited vertices into the path at valid positions.
        """
        if not path:
            return path
        path = path[:]
        visited_set = set(path)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            unvisited = [v for v in largest_comp if v not in visited_set]
            if not unvisited:
                break
                
            for v in unvisited:
                if time.time() >= deadline:
                    break
                for i in range(len(path) - 1):
                    a, b = path[i], path[i+1]
                    if v in adj[a] and v in adj[b]:
                        if v not in visited_set:
                            new_path = path[:i+1] + [v] + path[i+1:]
                            valid = True
                            for j in range(len(new_path) - 1):
                                if new_path[j+1] not in adj[new_path[j]]:
                                    valid = False
                                    break
                            if valid and len(new_path) == len(set(new_path)):
                                path = new_path
                                visited_set = set(path)
                                improved = True
                                break
                if improved:
                    break
                    
        return path

    def backtrack_search(path, max_backtrack=10000):
        """
        Try to improve path by backtracking from various positions.
        Returns improved path or None.
        """
        if not path:
            return None
        
        best = path[:]
        path = path[:]
        
        # Try backtracking from multiple positions
        positions = list(range(len(path) - 1, max(len(path) - 12, -1), -1))
        if len(path) > 20:
            mid = len(path) // 2
            positions.extend([mid - 3, mid - 2, mid - 1, mid, mid + 1, mid + 2, mid + 3])
            positions = [p for p in positions if 0 <= p < len(path)]
            positions = sorted(set(positions), reverse=True)
            
        for i in positions:
            if time.time() >= deadline:
                break
            if i < 0 or i >= len(path):
                continue
            suffix = path[i:]
            prefix = path[:i]
            forbidden = set(prefix)
            
            def dfs_extend(sub_path, sub_visited, depth_limit=15):
                if time.time() >= deadline or depth_limit <= 0:
                    return sub_path
                current = sub_path[-1]
                neighbors = [v for v in adj[current] if v not in sub_visited and v not in forbidden]
                if not neighbors:
                    return sub_path
                
                def score(v):
                    return len([u for u in adj[v] if u not in sub_visited and u not in forbidden])
                neighbors.sort(key=score)
                
                best_ext = sub_path[:]
                for nei in neighbors[:min(4, len(neighbors))]:
                    new_path = sub_path + [nei]
                    new_visited = sub_visited | {nei}
                    ext = dfs_extend(new_path, new_visited, depth_limit - 1)
                    if len(ext) > len(best_ext):
                        best_ext = ext
                return best_ext
            
            extended_suffix = dfs_extend(suffix, set(suffix))
            candidate = prefix + extended_suffix
            
            if len(candidate) == len(set(candidate)):
                valid = True
                for j in range(len(candidate) - 1):
                    if candidate[j+1] not in adj[candidate[j]]:
                        valid = False
                        break
                if valid and len(candidate) > len(best):
                    best = candidate
        
        return best if len(best) > len(path) else None

    def merge_paths(path1, path2):
        """
        Try to merge two paths into one longer path.
        path1 and path2 are lists of distinct vertices.
        Returns merged path or None.
        """
        if not path1 or not path2:
            return None
        
        set1 = set(path1)
        set2 = set(path2)
        
        # If they share vertices, we can't simply concatenate
        if set1 & set2:
            return None
        
        # Try to connect end of path1 to start of path2
        # Or start of path1 to end of path2
        # Or end of path1 to end of path2 (reversing path2)
        # Or start of path1 to start of path2 (reversing path2)
        
        candidates = []
        
        # 1. path1 + path2
        if path1[-1] in adj[path2[0]]:
            candidates.append(path1 + path2)
        
        # 2. path1 + reversed(path2)
        if path1[-1] in adj[path2[-1]]:
            candidates.append(path1 + path2[::-1])
        
        # 3. reversed(path1) + path2
        if path1[0] in adj[path2[0]]:
            candidates.append(path1[::-1] + path2)
        
        # 4. reversed(path1) + reversed(path2)
        if path1[0] in adj[path2[-1]]:
            candidates.append(path1[::-1] + path2[::-1])
        
        if not candidates:
            return None
        
        # Pick the longest (they should all be same length, but just in case)
        best = max(candidates, key=len)
        
        # Validate
        if len(best) != len(set(best)):
            return None
        for i in range(len(best) - 1):
            if best[i+1] not in adj[best[i]]:
                return None
                
        return best

    strategies = ['warnsdorff', 'random', 'max_deg']
    
    path_pool = []
    
    while time.time() < deadline - 5:
        for strat in strategies:
            if time.time() >= deadline - 5:
                break
            start = random.choice(largest_comp)
            path = greedy_walk(start, strat)
            path = extend_path(path)
            
            # Local improvement
            if len(path) < comp_size and time.time() < deadline - 10:
                path = local_improve(path, max_iter=50)
                path = extend_path(path)
            
            # Backtracking improvement
            if len(path) < comp_size and time.time() < deadline - 10:
                improved = backtrack_search(path)
                if improved and len(improved) > len(path):
                    path = improved
                    path = extend_path(path)
                    path = local_improve(path, max_iter=30)
                    path = extend_path(path)
            
            if len(path) > len(best_path):
                best_path = path
            
            if len(path) >= 10:
                path_pool.append(path)
            
            if len(path_pool) > 60:
                path_pool.pop(0)
        
        # Try merging
        if len(path_pool) >= 2 and time.time() < deadline - 5:
            for _ in range(8):
                if time.time() >= deadline - 5:
                    break
                i = random.randint(0, len(path_pool) - 1)
                j = random.randint(0, len(path_pool) - 1)
                if i == j:
                    continue
                merged = merge_paths(path_pool[i], path_pool[j])
                if merged:
                    merged = extend_path(merged)
                    merged = local_improve(merged, max_iter=30)
                    merged = extend_path(merged)
                    if len(merged) > len(best_path):
                        best_path = merged
                    if len(merged) >= 10:
                        path_pool.append(merged)
                        if len(path_pool) > 60:
                            path_pool.pop(0)

    # Final polish
    if best_path:
        best_path = local_improve(best_path, max_iter=100)
        best_path = extend_path(best_path)
        if len(best_path) < comp_size:
            improved = backtrack_search(best_path)
            if improved and len(improved) > len(best_path):
                best_path = improved
            best_path = extend_path(best_path)
            best_path = local_improve(best_path, max_iter=50)
            best_path = extend_path(best_path)

    if not best_path:
        best_path = [largest_comp[0]]
    
    # Final validation
    if len(best_path) != len(set(best_path)):
        seen = set()
        valid_path = []
        for node in best_path:
            if node not in seen:
                valid_path.append(node)
                seen.add(node)
            else:
                break
        best_path = valid_path
        
        for i in range(len(best_path) - 1):
            if best_path[i+1] not in adj[best_path[i]]:
                best_path = [best_path[0]]
                break
    
    return best_path
