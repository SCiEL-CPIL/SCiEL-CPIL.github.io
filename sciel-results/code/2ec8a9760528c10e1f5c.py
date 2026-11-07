"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    es = {tuple(e) for e in edges}
    es |= {tuple(sorted(e)) for e in edges}
    
    def bfs_path(s, t, forbidden):
        """Find a simple path from s to t avoiding 'forbidden' vertices in interior."""
        if s == t:
            return None
        # BFS
        from collections import deque
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                # Reconstruct path
                path = []
                cur = t
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj[u]:
                if v not in parent:
                    if v == t or v not in forbidden:
                        parent[v] = u
                        q.append(v)
        return None
    
    def check_clique_subdivision(branches):
        """
        Check if the given set of branches can be the branch vertices of a clique subdivision.
        Returns a dict of paths or None.
        branches: list of vertices
        """
        k = len(branches)
        if k < 2:
            return None
        
        # For each pair of branches, find a path. Interior vertices must be disjoint across all paths.
        # Greedy: process pairs in some order, assign shortest available path.
        # To maximize success, process pairs with fewest options first? Or just try all orders?
        # For now, try a heuristic: for each pair, find shortest path avoiding other branch vertices and already-used interior vertices.
        
        pairs = list(combinations(branches, 2))
        
        # Try multiple orderings of pairs
        import random
        best_paths = None
        
        # First, try natural order
        for attempt in range(20):
            used_interiors = set()
            paths = {}
            success = True
            
            # Order pairs: if attempt==0, natural; else shuffled
            if attempt == 0:
                order = pairs
            else:
                order = list(pairs)
                random.shuffle(order)
            
            for (u, v) in order:
                forbidden = set(branches) - {u, v}
                forbidden |= used_interiors
                
                # BFS for shortest path from u to v avoiding forbidden in interior
                # We can use BFS on the graph where forbidden vertices are removed (except endpoints)
                # Actually, we need a path from u to v where all interior vertices are not in forbidden.
                
                from collections import deque
                # Modified BFS
                parent = {u: None}
                q = deque([u])
                found = False
                while q:
                    x = q.popleft()
                    if x == v:
                        found = True
                        break
                    for y in adj[x]:
                        if y not in parent:
                            # y can be v (endpoint) or any vertex not in forbidden
                            if y == v or y not in forbidden:
                                parent[y] = x
                                q.append(y)
                
                if not found:
                    success = False
                    break
                
                # Reconstruct path
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                
                # Check interior vertices are not in forbidden
                interior = set(path[1:-1])
                if interior & forbidden:
                    success = False
                    break
                
                used_interiors |= interior
                paths[(u, v)] = path
            
            if success:
                # Verify all paths
                all_interiors = set()
                for (u, v), path in paths.items():
                    interior = set(path[1:-1])
                    if interior & all_interiors:
                        success = False
                        break
                    if interior & set(branches):
                        success = False
                        break
                    all_interiors |= interior
                
                if success:
                    # Build result
                    result_paths = []
                    for (u, v) in pairs:
                        result_paths.append(list(paths[(u, v)]))
                    return {'branches': list(branches), 'paths': result_paths}
        
        return None
    
    # Find the largest clique subdivision
    # Strategy: find all maximal cliques, then for each clique, try to extend it by adding vertices.
    # Or: brute force over subsets of vertices, check if they form a clique subdivision.
    # n=24, so 2^24 is too large. But we can use the fact that branches must be a set where every pair has a path.
    
    # Better: enumerate cliques in the graph, then for each clique, see if we can find disjoint paths.
    # Actually, the branches don't need to form a clique in the original graph; they just need paths between every pair.
    # So the "clique" is in the subdivision sense.
    
    # Approach: 
    # 1. Find all maximal cliques in the graph (these are candidate branch sets where direct edges exist).
    # 2. For each clique, check if it's a valid clique subdivision (paths exist).
    # 3. Also try adding vertices to cliques.
    
    # But the branches don't have to be a clique in the original graph. For example, two non-adjacent vertices can be branches if there's a path between them.
    
    # Let's think differently: the problem is to find a set S of vertices and for every pair in S, a path, such that all interior vertices are disjoint.
    # This is equivalent to finding a clique in the "path graph" where two vertices are connected if there's a path between them (trivially true for connected graphs).
    # The constraint is the disjointness of interiors.
    
    # Greedy approach:
    # Start with a pair of vertices. Add vertices one by one, checking if we can find disjoint paths.
    
    # Let's try: for each subset of size k (from large to small), check if it's a valid branch set.
    # But 2^24 is too large.
    
    # Alternative: use the fact that the maximum number of branches is bounded by n - (number of interior vertices used).
    # Each path between a pair uses at least 1 interior vertex (unless the pair is adjacent, in which case 0).
    # For k branches, there are k*(k-1)/2 pairs. If all are adjacent, 0 interior vertices. Otherwise, more.
    
    # Let's try a greedy growth:
    # Start with the largest clique in the graph.
    # Then try to add vertices one by one.
    
    def find_max_clique():
        # Bron-Kerbosch
        best = []
        def bronk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            # Choose pivot
            pivot = list(P | X)[0] if (P | X) else None
            candidates = P - adj[pivot] if pivot is not None else P
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bronk(set(), set(range(n)), set())
        return best
    
    max_clique = find_max_clique()
    best_result = None
    
    # Try the max clique first
    if len(max_clique) >= 2:
        result = check_clique_subdivision(max_clique)
        if result:
            best_result = result
    
    # Try to extend the clique by adding vertices
    # For each vertex not in the clique, try to add it
    current = set(max_clique)
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if v not in current:
                new_set = current | {v}
                result = check_clique_subdivision(list(new_set))
                if result:
                    current = new_set
                    best_result = result
                    improved = True
                    break
    
    # Also try other cliques
    # Enumerate all cliques of size >= 2
    all_cliques = []
    def enumerate_cliques(R, P):
        if len(R) >= 2:
            all_cliques.append(list(R))
        for v in list(P):
            enumerate_cliques(R | {v}, P & adj[v])
            P.remove(v)
    
    # This might be too slow. Let's just try the max clique and a few others.
    
    # Try all pairs
    for u, v in edges:
        if len(best_result) == 0 or best_result is None:
            result = check_clique_subdivision([u, v])
            if result:
                best_result = result
    
    if best_result is None:
        # Fallback: just two adjacent vertices
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Ensure types are correct
    best_result['branches'] = [int(x) for x in best_result['branches']]
    best_result['paths'] = [[int(x) for x in path] for path in best_result['paths']]
    
    return best_result

# EVOLVE-BLOCK-END
