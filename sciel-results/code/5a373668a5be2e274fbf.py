"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_disjoint_paths(branches, used_interior):
        """Find internally disjoint paths between all pairs of branches."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        paths = {}
        
        # Try to find paths one by one, backtracking if needed
        def backtrack(idx, used_int):
            if time.time() > deadline:
                return False
            if idx == len(pairs):
                return True
            u, v = pairs[idx]
            # BFS to find shortest path from u to v avoiding used_int and other branches
            queue = deque([(u, [u])])
            visited = {u}
            found_path = None
            while queue:
                if time.time() > deadline:
                    return False
                node, path = queue.popleft()
                if node == v:
                    interior = set(path[1:-1])
                    if not (interior & used_int) and not (interior & branch_set):
                        found_path = path
                        break
                    continue
                for nbr in sorted(adj[node]):
                    if nbr in visited:
                        continue
                    if nbr in branch_set and nbr != v:
                        continue
                    if nbr in used_int:
                        continue
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
            
            if found_path is None:
                return False
            
            interior = set(found_path[1:-1])
            paths[(u, v)] = found_path
            if backtrack(idx + 1, used_int | interior):
                return True
            del paths[(u, v)]
            return False
        
        if backtrack(0, used_interior):
            return paths
        return None
    
    # Find all cliques of size >= 3, try largest first
    # Use a simple approach: check all subsets of size 4, 5, 6...
    best = None
    
    # First try to find a 4-clique with valid subdivision
    for size in range(6, 2, -1):  # Try 6, 5, 4
        if time.time() > deadline:
            break
        found = False
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            # Check if it's a clique
            is_clique = True
            for a, b in combinations(combo, 2):
                if b not in adj[a]:
                    is_clique = False
                    break
            if not is_clique:
                continue
            
            # Try to find disjoint paths
            paths = find_disjoint_paths(list(combo), set())
            if paths is not None:
                result_paths = []
                for (u, v) in combinations(combo, 2):
                    result_paths.append(list(paths[(u, v)]))
                if best is None or size > len(best[0]):
                    best = (list(combo), result_paths)
                found = True
                break
        
        if found:
            break
    
    if best is None:
        # Fallback to triangle
        for row in combinations(range(n), 3):
            if all(b in adj[a] for a, b in combinations(row, 2)):
                best = (list(row), [list(e) for e in combinations(row, 2)])
                break
    
    if best is None:
        u, v = P['edges'][0]
        best = ([u, v], [[u, v]])
    
    branches, paths = best
    return {'branches': [int(x) for x in branches], 'paths': [[int(x) for x in p] for p in paths]}

# EVOLVE-BLOCK-END
