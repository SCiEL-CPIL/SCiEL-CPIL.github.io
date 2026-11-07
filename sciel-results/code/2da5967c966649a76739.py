"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import sys
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all-pairs shortest paths for use in greedy expansion
    # dist[i][j] = shortest path length from i to j
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        q = deque([i])
        while q:
            curr = q.popleft()
            for nxt in adj[curr]:
                if dist[i][nxt] == -1:
                    dist[i][nxt] = dist[i][curr] + 1
                    q.append(nxt)
    
    # Find all maximal cliques (or near-maximal) using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P_set, X, clique):
        if not P_set and not X:
            if len(R) >= 3:
                clique.append(list(R))
            return
        # Choose pivot u from P_set ∪ X to maximize |P_set ∩ N(u)|
        U = P_set | X
        if not U:
            if len(R) >= 3:
                clique.append(list(R))
            return
        pivot = max(U, key=lambda u: len(P_set & adj[u]))
        candidates = P_set - adj[pivot]
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], clique)
            P_set.remove(v)
            X.add(v)
    
    cliques = []
    bron_kerbosch(set(), set(range(n)), set(), cliques)
    
    # Sort cliques by size descending
    cliques.sort(key=len, reverse=True)
    
    # Also consider near-cliques: sets of vertices where most pairs are adjacent
    # This helps if no large clique exists but a large "almost-clique" with paths does
    
    def find_disjoint_paths(branches):
        """Find vertex-disjoint paths between all pairs of branches using max-flow."""
        m = len(branches)
        if m < 2:
            return None
        
        # Build flow network
        # Nodes: for each original vertex v, create v_in and v_out
        # Branch vertices get special treatment
        # We need to find k vertex-disjoint paths covering all C(k,2) pairs
        
        # Actually, let's use a different approach: sequential path finding with backtracking
        # but with a smarter ordering and more thorough search
        
        pairs = list(itertools.combinations(range(m), 2))
        
        def solve(idx, used, current_paths):
            if idx == len(pairs):
                return True, current_paths
            
            i, j = pairs[idx]
            u, v = branches[i], branches[j]
            
            # Blocked: used vertices and all branches except u, v
            blocked = used | (set(branches) - {u, v})
            
            # Try to find a path from u to v avoiding blocked
            # Use BFS to find shortest path
            if u == v:
                return False, None
            
            visited = {u}
            queue = deque([(u, [u])])
            best_path = None
            while queue:
                curr, path = queue.popleft()
                for nxt in sorted(adj[curr]):
                    if nxt in visited:
                        continue
                    if nxt in blocked and nxt not in (u, v):
                        continue
                    if nxt == v:
                        # Found a path
                        candidate = path + [v]
                        internal = set(candidate[1:-1])
                        if not (internal & used):
                            new_used = used | internal
                            res = solve(idx + 1, new_used, current_paths + [candidate])
                            if res[0]:
                                return res
                        best_path = candidate
                        break
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
            
            return False, None
        
        res, paths = solve(0, set(), [])
        if res:
            return paths
        return None
    
    # Try each clique (and near-clique) in order of size
    best_branches = None
    best_paths = None
    
    # First try actual cliques
    for clique in cliques:
        if len(clique) < 3:
            continue
        # Try subsets of the clique, starting with the full clique
        for k in range(len(clique), 2, -1):
            found = False
            for subset in itertools.combinations(clique, k):
                subset = list(subset)
                # Verify it's actually a clique
                is_clique = True
                for a, b in itertools.combinations(subset, 2):
                    if b not in adj[a]:
                        is_clique = False
                        break
                if not is_clique:
                    continue
                paths = find_disjoint_paths(subset)
                if paths:
                    best_branches = subset
                    best_paths = paths
                    found = True
                    break
            if found:
                break
        if best_branches:
            break
    
    # If no clique works, try near-cliques (sets with high edge density)
    if not best_branches:
        # Try all subsets of size 4-6 from high-degree vertices
        vertices = list(range(n))
        vertices.sort(key=lambda x: len(adj[x]), reverse=True)
        top_vertices = vertices[:12]
        
        for k in range(6, 2, -1):
            found = False
            for subset in itertools.combinations(top_vertices, k):
                subset = list(subset)
                paths = find_disjoint_paths(subset)
                if paths:
                    best_branches = subset
                    best_paths = paths
                    found = True
                    break
            if found:
                break
    
    # Fallback
    if not best_branches:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Convert to proper Python int types
    best_branches = [int(b) for b in best_branches]
    best_paths = [[int(p) for p in path] for path in best_paths]
    
    return {'branches': best_branches, 'paths': best_paths}
# EVOLVE-BLOCK-END
