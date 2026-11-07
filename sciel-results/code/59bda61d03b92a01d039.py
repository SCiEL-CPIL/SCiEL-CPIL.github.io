"""Rewritten program for large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import itertools
    from collections import deque

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def find_max_clique():
        best = []
        def dfs(candidates, current):
            nonlocal best
            if len(current) > len(best):
                best = current[:]
            if not candidates:
                return
            if len(current) + len(candidates) <= len(best):
                return
            for i in range(len(candidates)):
                v = candidates[i]
                new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
                dfs(new_candidates, current + [v])
        dfs(list(range(n)), [])
        return best

    max_clique = find_max_clique()
    if len(max_clique) >= 2:
        branches = list(max_clique)
        paths = []
        for u, v in itertools.combinations(branches, 2):
            if v in adj[u]:
                paths.append([u, v])
            else:
                paths.append([u, v])
        return {'branches': branches, 'paths': paths}

    def find_shortest_paths(u, v, max_len=6):
        """Find all simple paths from u to v of length <= max_len using BFS-like DFS."""
        paths = []
        def dfs(current, visited, path):
            if len(path) - 1 > max_len:
                return
            if current == v and len(path) > 1:
                paths.append(path[:])
                return
            # Heuristic: prioritize neighbors with higher degree? Or just standard DFS.
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        dfs(u, {u}, [u])
        # Sort by length to prefer shorter paths
        paths.sort(key=len)
        return paths

    def try_find_subdivision(k, max_path_len=5):
        for combo in itertools.combinations(range(n), k):
            branches = list(combo)
            pairs = list(itertools.combinations(branches, 2))
            
            candidate_paths = {}
            valid = True
            for u, v in pairs:
                ps = find_shortest_paths(u, v, max_path_len)
                if not ps:
                    valid = False
                    break
                candidate_paths[(u, v)] = ps
            
            if not valid:
                continue
            
            # Order pairs by number of candidates (most constrained first)
            ordered_pairs = sorted(pairs, key=lambda p: len(candidate_paths[p]))
            
            def backtrack(idx, used_interiors, assigned):
                if idx == len(ordered_pairs):
                    return True
                u, v = ordered_pairs[idx]
                for path in candidate_paths[(u, v)]:
                    interior = set(path[1:-1])
                    if interior.isdisjoint(used_interiors):
                        new_used = used_interiors | interior
                        assigned.append(path)
                        if backtrack(idx + 1, new_used, assigned):
                            return True
                        assigned.pop()
                return False
            
            assigned = []
            if backtrack(0, set(), assigned):
                return {'branches': branches, 'paths': assigned}
        return None

    # Try k from 6 down to 3
    for k in range(6, 2, -1):
        result = try_find_subdivision(k, max_path_len=5)
        if result:
            return result

    # Fallback: return the max clique found earlier (which was size 3)
    branches = list(max_clique)
    paths = []
    for u, v in itertools.combinations(branches, 2):
        if v in adj[u]:
            paths.append([u, v])
        else:
            paths.append([u, v])
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
