"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    from itertools import combinations, permutations
    import sys
    sys.setrecursionlimit(10000)
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_path(start, end, forbidden, max_len=4):
        """Find a simple path from start to end avoiding forbidden vertices (except endpoints)."""
        if start == end:
            return None
        if end in adj[start]:
            return [start, end]
        # BFS up to max_len
        from collections import deque
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            if len(path) > max_len:
                continue
            for nb in adj[node]:
                if nb == end:
                    return path + [end]
                if nb not in forbidden and nb not in visited and nb != start:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def try_complete_clique(branches, paths, used_interiors, candidate_order):
        """Try to expand the current clique subdivision by adding candidates."""
        best = (len(branches), list(branches), [list(p) for p in paths])
        
        for cand in candidate_order:
            if cand in branches:
                continue
            # Check if cand can connect to all existing branches
            new_paths = []
            new_interiors = set(used_interiors)
            success = True
            for b in branches:
                if b == cand:
                    continue
                # Try direct edge first
                if cand in adj[b]:
                    new_paths.append([b, cand])
                else:
                    # Find path avoiding all branches and used interiors
                    forbidden = set(branches) | new_interiors
                    forbidden.discard(b)
                    forbidden.discard(cand)
                    p = find_path(b, cand, forbidden, max_len=3)
                    if p is None:
                        success = False
                        break
                    new_paths.append(p)
                    for v in p[1:-1]:
                        new_interiors.add(v)
            
            if success:
                new_branches = branches + [cand]
                # Now try to complete with remaining candidates
                remaining = [c for c in candidate_order if c not in new_branches]
                sub_result = try_complete_clique(new_branches, new_paths, new_interiors, remaining)
                if len(sub_result[0]) > len(best[0]):
                    best = sub_result
        
        return best
    
    # Find all maximal cliques to use as starting points
    def find_cliques(min_size=2):
        """Find all cliques of size >= min_size using Bron-Kerbosch."""
        cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= min_size:
                    cliques.append(frozenset(R))
                return
            # Choose pivot to reduce recursion
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = set()
            
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(set(), set(range(n)), set())
        return cliques
    
    # Strategy: try different starting cliques and expand greedily
    best_result = None
    best_size = 0
    
    # Get all cliques of size 2+
    cliques = find_cliques(min_size=2)
    
    # Sort by size descending, then by vertex sum for determinism
    cliques.sort(key=lambda c: (-len(c), sum(c)))
    
    # Limit number of starting cliques to try
    max_starts = 50
    for clique in cliques[:max_starts]:
        branch_list = sorted(clique)
        # Initial paths: direct edges for clique members
        paths = []
        for u, v in combinations(branch_list, 2):
            if v in adj[u]:
                paths.append([u, v])
            else:
                # Shouldn't happen for a clique, but handle gracefully
                paths.append([u, v])
        
        # Candidate order: vertices sorted by connectivity to branches
        candidates = [v for v in range(n) if v not in clique]
        def score_candidate(v):
            return sum(1 for b in branch_list if v in adj[b])
        candidates.sort(key=lambda v: (-score_candidate(v), v))
        
        result = try_complete_clique(branch_list, paths, set(), candidates)
        if result[0] > best_size:
            best_size = result[0]
            best_result = result
    
    # Fallback: find any triangle
    if best_result is None or best_result[0] < 2:
        for row in combinations(range(n), 3):
            if all(e in adj[e[0]] for e in combinations(row, 2)):
                best_result = (3, list(row), [list(e) for e in combinations(row, 2)])
                break
    
    if best_result is None:
        u, v = edges[0]
        best_result = (2, [u, v], [[u, v]])
    
    size, branches, paths = best_result
    
    # Ensure types are correct
    branches = [int(b) for b in branches]
    paths = [[int(v) for v in p] for p in paths]
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
