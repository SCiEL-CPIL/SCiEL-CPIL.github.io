"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 80.0  # 10s safety margin before 90s limit
    
    def greedy_extend(path, seen):
        """Extend path greedily from both ends, picking more constrained end first."""
        changed = True
        while changed:
            changed = False
            # Try extending from the end
            end = path[-1]
            candidates = adj[end] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                changed = True
            # Try extending from the start
            start = path[0]
            candidates = adj[start] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        return path
    
    def two_opt(path):
        """Apply 2-opt local search to rearrange path segments."""
        n_path = len(path)
        if n_path < 4:
            return path
        improved = True
        while improved:
            improved = False
            for i in range(n_path - 2):
                if time.time() > deadline:
                    return path
                for j in range(i + 2, n_path):
                    if time.time() > deadline:
                        return path
                    # Check if reversing path[i+1:j+1] maintains valid edges
                    # Edge before i: path[i]-path[i+1] becomes path[i]-path[j]
                    # Edge after j: path[j]-path[j+1] becomes path[i+1]-path[j+1] (if j+1 exists)
                    a, b = path[i], path[i + 1]
                    c, d = path[j], path[j + 1] if j + 1 < n_path else None
                    
                    # New edges would be: a-c and b-d (if d exists)
                    if c in adj[a] and (d is None or b in adj[d]):
                        # Also need to check that we don't create duplicate vertices
                        # (which we won't since we're just reversing a segment)
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        # Verify it's still a valid path
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k+1] not in adj[new_path[k]]:
                                valid = False
                                break
                        if valid:
                            # Check if this actually improves (different arrangement)
                            if new_path != path:
                                path = new_path
                                improved = True
                                break
                if improved:
                    break
        return path
    
    def dfs_longest(start, seen, path, best_path):
        """DFS to find longest path from start."""
        if len(path) > len(best_path):
            best_path = path[:]
        
        current = path[-1]
        neighbors = sorted(adj[current] - seen, key=lambda w: (len(adj[w] - seen), w))
        
        for v in neighbors:
            if time.time() > deadline:
                return best_path
            seen.add(v)
            path.append(v)
            best_path = dfs_longest(v, seen, path, best_path)
            path.pop()
            seen.remove(v)
        
        return best_path
    
    best = [0]
    
    # Phase 1: Greedy extension from each start vertex
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        path = greedy_extend(path, seen)
        if len(path) > len(best):
            best = path[:]
    
    # Phase 2: 2-opt local search on best path
    if time.time() < deadline:
        improved_path = two_opt(best)
        if len(improved_path) > len(best):
            best = improved_path
    
    # Phase 3: Extend after 2-opt
    if time.time() < deadline:
        seen = set(best)
        extended = greedy_extend(best[:], seen)
        if len(extended) > len(best):
            best = extended
    
    # Phase 4: Try DFS from endpoints of best path
    if time.time() < deadline and len(best) >= 2:
        # Try extending from the start of best path via DFS
        for endpoint in [best[0], best[-1]]:
            if time.time() > deadline:
                break
            # Remove endpoint from best to allow DFS to find longer path
            base_path = best[1:] if endpoint == best[0] else best[:-1]
            seen = set(base_path)
            dfs_path = dfs_longest(endpoint, seen, [endpoint], [endpoint])
            # Combine: dfs_path + base_path (if they connect)
            if len(dfs_path) >= 2 and dfs_path[-1] in adj[base_path[0]]:
                combined = dfs_path + base_path
                if len(combined) > len(best):
                    best = combined
    
    # Phase 5: Multiple random restarts with greedy
    if time.time() < deadline:
        rng = random.Random(748)
        for _ in range(50):
            if time.time() > deadline:
                break
            start = rng.randint(0, n - 1)
            path = [start]
            seen = {start}
            # Randomized greedy
            while adj[path[-1]] - seen:
                candidates = list(adj[path[-1]] - seen)
                rng.shuffle(candidates)
                candidates.sort(key=lambda w: (len(adj[w] - seen), w))
                # Pick from top few with some randomness
                top_k = min(3, len(candidates))
                v = candidates[rng.randint(0, top_k - 1)]
                path.append(v)
                seen.add(v)
            path = greedy_extend(path, seen)
            if len(path) > len(best):
                best = path[:]
    
    return best
# EVOLVE-BLOCK-END
