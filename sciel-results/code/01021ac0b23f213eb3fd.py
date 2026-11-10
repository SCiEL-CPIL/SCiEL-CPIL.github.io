import time
from collections import deque
from itertools import combinations
import random
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 58.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def time_ok():
        return time.time() < deadline

    def find_all_paths(start, end, forbidden, max_len=8):
        """Find all simple paths from start to end of length <= max_len avoiding forbidden vertices."""
        if start == end:
            return [[start]]
        if start not in adj and end not in adj:
            return []
        paths = []
        
        def dfs(node, path, visited):
            if len(paths) > 200:
                return
            if len(path) - 1 > max_len:
                return
            if node == end:
                paths.append(path[:])
                return
            for nb in sorted(adj[node]):
                if nb in visited:
                    continue
                if nb in forbidden and nb != end:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
        
        dfs(start, [start], {start})
        paths.sort(key=len)
        return paths

    def check_hall(remaining_pairs, used_int, branch_set):
        """
        Check Hall's marriage condition: for every subset of remaining pairs,
        the union of their available interior vertices must be at least as large
        as the number of vertices needed.
        
        We use a simpler sufficient condition: for each pair, it must have at least
        enough available vertices, and for small subsets of pairs, the combined
        available vertices must suffice.
        """
        if not remaining_pairs:
            return True
        
        # Simple check: each pair needs at least 1 available vertex (for direct edge case)
        # Actually, a path from a to b needs at least 0 interior vertices (direct edge)
        # or more. Let's check that each pair has at least one possible path.
        for a, b in remaining_pairs:
            forbidden = used_int | (branch_set - {a, b})
            # Check if there's any edge a-b or a common neighbor
            if b in adj[a]:
                continue
            common = (adj[a] & adj[b]) - forbidden
            if not common:
                # No direct edge and no common neighbor means no path of length 2
                # Check length 3
                found = False
                for x in adj[a]:
                    if x in forbidden or x == a or x == b:
                        continue
                    for y in adj[x]:
                        if y == b and y not in forbidden:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False
        return True

    def solve_csp(branches):
        """
        Path-centric CSP: assign a vertex-disjoint path to each pair of branches.
        Variables: the pairs (unordered pairs of branch vertices)
        Values: possible simple paths between the two branch vertices
        Constraints: paths must be vertex-disjoint in their interiors
        """
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Precompute all possible paths for each pair
        pair_paths = {}
        for a, b in pairs:
            forbidden = branch_set - {a, b}
            paths = find_all_paths(a, b, forbidden, max_len=8)
            pair_paths[(a, b)] = paths
            if not paths:
                return None  # No path exists for this pair
        
        # Order pairs by number of options (fewest first - most constrained)
        pairs_sorted = sorted(pairs, key=lambda p: len(pair_paths[p]))
        
        def backtrack(assigned_pairs, used_int, current_solution):
            if not time_ok():
                return None
            if len(assigned_pairs) == len(pairs):
                return current_solution
            
            # Most constrained variable: pick the unassigned pair with fewest remaining options
            best_idx = -1
            best_count = float('inf')
            for i, (a, b) in enumerate(pairs_sorted):
                if (a, b) in assigned_pairs:
                    continue
                # Filter paths that don't conflict with used_int
                valid_paths = [p for p in pair_paths[(a, b)] if not (set(p[1:-1]) & used_int)]
                if not valid_paths:
                    return None  # Dead end
                if len(valid_paths) < best_count:
                    best_count = len(valid_paths)
                    best_idx = i
            
            a, b = pairs_sorted[best_idx]
            valid_paths = [p for p in pair_paths[(a, b)] if not (set(p[1:-1]) & used_int)]
            
            # Sort paths by length (shorter first - uses fewer vertices)
            valid_paths.sort(key=len)
            
            for path in valid_paths:
                interior = set(path[1:-1])
                new_used = used_int | interior
                new_assigned = assigned_pairs | {(a, b)}
                new_solution = current_solution + [path]
                
                result = backtrack(new_assigned, new_used, new_solution)
                if result is not None:
                    return result
            return None
        
        return backtrack(set(), set(), [])

    def find_max_cliques():
        cliques = []
        R = set()
        Pset = set(range(n))
        X = set()
        def bronk(R, Pset, X):
            if not time_ok():
                return
            if not Pset and not X:
                cliques.append(frozenset(R))
                return
            if Pset or X:
                U = Pset | X
                pivot = max(U, key=lambda v: len(Pset & adj[v]))
                candidates = Pset - adj[pivot]
            else:
                candidates = Pset
            for v in list(candidates):
                bronk(R | {v}, Pset & adj[v], X & adj[v])
                Pset.remove(v)
                X.add(v)
        bronk(R, Pset, X)
        return cliques

    # Strategy: try branch sets from largest to smallest
    # 1. Find maximal cliques
    # 2. Try each clique and subsets
    # 3. Try random subsets

    cliques = find_max_cliques()
    cliques = sorted(cliques, key=lambda c: -len(c))

    best_branches = None
    best_paths = None

    # Try each maximal clique
    for clique in cliques:
        if not time_ok():
            break
        clique_list = sorted(clique)
        if best_branches is not None and len(clique_list) <= len(best_branches):
            continue
        result = solve_csp(clique_list)
        if result is not None:
            best_branches = clique_list
            best_paths = result

    # Try to expand
    if best_branches is not None:
        current = best_branches[:]
        expanded = True
        while expanded and time_ok():
            expanded = False
            current_set = set(current)
            candidates = [v for v in range(n) if v not in current_set]
            candidates.sort(key=lambda v: -len(adj[v]))
            for v in candidates:
                if not time_ok():
                    break
                new_branches = sorted(current + [v])
                result = solve_csp(new_branches)
                if result is not None:
                    current = new_branches
                    best_branches = current[:]
                    best_paths = result
                    expanded = True
                    break

    # Try random subsets of various sizes
    rng = random.Random(42)
    for size in range(n, 2, -1):
        if not time_ok():
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(30):
            if not time_ok():
                break
            random_set = sorted(rng.sample(range(n), size))
            result = solve_csp(random_set)
            if result is not None:
                best_branches = random_set
                best_paths = result
                break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]

    return {'branches': best_branches, 'paths': best_paths}
