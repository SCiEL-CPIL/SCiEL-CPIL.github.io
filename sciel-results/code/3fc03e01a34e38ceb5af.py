"""Rewritten program for large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    from collections import deque
    import itertools
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # We want to find a set of branch vertices B such that for every pair (u,v) in B,
    # there is a simple path from u to v whose interior is disjoint from all other path interiors and from B.
    
    # Strategy: Try to find a clique subdivision with as many branches as possible.
    # Start by trying to find a K_k subdivision.
    
    # First, let's find the maximum clique in the graph to get an upper bound.
    # But actually, we can have a subdivision of a larger clique than the clique number? No, 
    # a subdivision of K_k requires k branch vertices. The branch vertices don't need to form a clique.
    
    # Let's try to find a set of k vertices that can be connected by internally disjoint paths.
    # This is equivalent to finding a topological minor of K_k.
    
    # For small k, we can try all combinations.
    # Let's start from k=6 and go down.
    
    # Actually, let's first check if there's a large clique, since that's the easiest case.
    # If we find a clique of size k, we're done with k branches.
    
    # Let's find the maximum clique using a simple branch and bound or just try all subsets for small n.
    # n=24 is small enough for some heuristics.
    
    # Let's try to find the largest k such that K_k is a topological minor.
    
    # We'll iterate k from high to low.
    # Upper bound on k: the number of vertices is 24, but we need at least 2 vertices per edge path (if no direct edge),
    # so roughly k(k-1)/2 paths, each needing at least 2 vertices (endpoints shared), plus internal vertices.
    # Total vertices needed >= k + k(k-1)/2 * 1 (if all paths have length 1) = k + k(k-1)/2.
    # For k=6: 6 + 15 = 21. For k=7: 7 + 21 = 28 > 24. So k <= 6 if all paths have length >= 1.
    # Wait, paths can share endpoints but interiors must be disjoint.
    # If all paths are length 1 (direct edges), then we just need a clique.
    # If some paths are longer, we use more vertices.
    # So k=6 is possible if we have enough vertices.
    
    # Let's try k from 6 down to 2.
    
    def find_clique_subdivision(k):
        """Try to find k branch vertices with internally disjoint paths between all pairs."""
        # For each combination of k vertices, check if they can form a K_k subdivision.
        # This is NP-hard in general, but for small k and n=24, we can try.
        
        # For a given set of branch vertices B, we need to find |B| choose 2 internally disjoint paths.
        # This is a disjoint paths problem.
        
        # For small k, we can use a flow-based approach or just try to find paths greedily.
        
        # Let's use a max-flow approach for vertex-disjoint paths.
        # Actually, for k=6, we have 15 pairs. Finding 15 internally vertex-disjoint paths is hard.
        
        # Alternative: First find a clique of size k. If found, we're done.
        # If not, try to find a subdivision.
        
        pass
    
    # Let's first try to find the maximum clique.
    def find_max_clique():
        # Use a simple recursive search with pruning.
        best = []
        def dfs(candidates, current):
            nonlocal best
            if len(current) > len(best):
                best = current[:]
            if not candidates:
                return
            # Pruning: if len(current) + len(candidates) <= len(best), return
            if len(current) + len(candidates) <= len(best):
                return
            for i in range(len(candidates)):
                v = candidates[i]
                # New candidates are those adjacent to v and after i in the list
                new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
                dfs(new_candidates, current + [v])
        dfs(list(range(n)), [])
        return best
    
    max_clique = find_max_clique()
    k_clique = len(max_clique)
    
    if k_clique >= 2:
        branches = list(max_clique)
        paths = []
        for u, v in itertools.combinations(branches, 2):
            # Check if u and v are adjacent
            if v in adj[u]:
                paths.append([u, v])
            else:
                # This shouldn't happen for a clique, but just in case
                paths.append([u, v])
        return {'branches': branches, 'paths': paths}
    
    # If no clique of size >= 2 (shouldn't happen), fall back to an edge.
    # But we need to try for subdivisions of non-cliques.
    
    # Let's try to find a K_4 subdivision, K_5 subdivision, etc.
    # For k=4, we need 6 internally disjoint paths.
    
    # Let's try a different approach: find 4 vertices that are "well-connected".
    
    # Actually, let's just try all combinations of 4 vertices and see if we can find 6 disjoint paths.
    # For each set of 4 vertices, we need paths for all 6 pairs.
    
    # This is getting complex. Let's use a simpler heuristic:
    # Try to find a set of k vertices where each pair has a short path, and then check for disjointness.
    
    # For now, let's try k=4, 5, 6 with a flow-based check.
    
    # Actually, the current program finds a triangle (k=3) and returns it.
    # Let's try to find a larger clique or a subdivision.
    
    # Let's try to find a K_4 subdivision.
    # Pick 4 branch vertices. For each pair, find a path. Check if all paths are internally disjoint.
    
    # To make this tractable, we can use the following:
    # 1. Find all simple paths of length <= 3 between each pair of candidate branch vertices.
    # 2. Use a backtracking search to assign paths to pairs such that interiors are disjoint.
    
    # This is still exponential, but for small path lengths it might work.
    
    # Let's try k=4 first.
    
    def find_paths_between(u, v, max_len=3):
        """Find all simple paths from u to v of length <= max_len."""
        paths = []
        def dfs(current, visited, path):
            if len(path) - 1 > max_len:
                return
            if current == v and len(path) > 1:
                paths.append(path[:])
                return
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        dfs(u, {u}, [u])
        return paths
    
    # Try to find a K_4 subdivision
    found = False
    for combo in itertools.combinations(range(n), 4):
        branches = list(combo)
        pairs = list(itertools.combinations(branches, 2))
        
        # For each pair, get candidate paths
        candidate_paths = {}
        for u, v in pairs:
            candidate_paths[(u, v)] = find_paths_between(u, v, max_len=4)
            if not candidate_paths[(u, v)]:
                break
        
        if len(candidate_paths) < 6:
            continue
        
        # Now try to assign one path to each pair such that interiors are disjoint
        # Use backtracking
        def backtrack(pairs_idx, used_interiors, assigned):
            if pairs_idx == len(pairs):
                return True
            u, v = pairs[pairs_idx]
            for path in candidate_paths[(u, v)]:
                interior = set(path[1:-1])
                if interior.isdisjoint(used_interiors):
                    new_used = used_interiors | interior
                    assigned.append(path)
                    if backtrack(pairs_idx + 1, new_used, assigned):
                        return True
                    assigned.pop()
            return False
        
        assigned = []
        if backtrack(0, set(), assigned):
            return {'branches': branches, 'paths': assigned}
    
    # Try k=5
    for combo in itertools.combinations(range(n), 5):
        branches = list(combo)
        pairs = list(itertools.combinations(branches, 2))
        
        candidate_paths = {}
        for u, v in pairs:
            candidate_paths[(u, v)] = find_paths_between(u, v, max_len=3)
            if not candidate_paths[(u, v)]:
                break
        
        if len(candidate_paths) < 10:
            continue
        
        def backtrack(pairs_idx, used_interiors, assigned):
            if pairs_idx == len(pairs):
                return True
            u, v = pairs[pairs_idx]
            for path in candidate_paths[(u, v)]:
                interior = set(path[1:-1])
                if interior.isdisjoint(used_interiors):
                    new_used = used_interiors | interior
                    assigned.append(path)
                    if backtrack(pairs_idx + 1, new_used, assigned):
                        return True
                    assigned.pop()
            return False
        
        assigned = []
        if backtrack(0, set(), assigned):
            return {'branches': branches, 'paths': assigned}
    
    # Try k=6
    for combo in itertools.combinations(range(n), 6):
        branches = list(combo)
        pairs = list(itertools.combinations(branches, 2))
        
        candidate_paths = {}
        for u, v in pairs:
            candidate_paths[(u, v)] = find_paths_between(u, v, max_len=3)
            if not candidate_paths[(u, v)]:
                break
        
        if len(candidate_paths) < 15:
            continue
        
        def backtrack(pairs_idx, used_interiors, assigned):
            if pairs_idx == len(pairs):
                return True
            u, v = pairs[pairs_idx]
            for path in candidate_paths[(u, v)]:
                interior = set(path[1:-1])
                if interior.isdisjoint(used_interiors):
                    new_used = used_interiors | interior
                    assigned.append(path)
                    if backtrack(pairs_idx + 1, new_used, assigned):
                        return True
                    assigned.pop()
            return False
        
        assigned = []
        if backtrack(0, set(), assigned):
            return {'branches': branches, 'paths': assigned}
    
    # If all else fails, return a triangle or an edge
    for row in itertools.combinations(range(n), 3):
        if all((a in adj[b]) for a, b in itertools.combinations(row, 2)):
            return {'branches': list(row), 'paths': [list(e) for e in itertools.combinations(row, 2)]}
    
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
