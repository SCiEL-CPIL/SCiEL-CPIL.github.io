"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute common neighbors for length-3 path feasibility
    # common[u][v] = set of w such that u-w-v is a valid path (w != u, w != v)
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cn = adj[u] & adj[v]
            cn.discard(u)
            cn.discard(v)
            common[u][v] = cn
    
    # Bron-Kerbosch for max clique (baseline)
    best_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    # If max clique is large enough, use it directly
    if len(best_clique) >= 7:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Bipartite matching approach for length-3 subdivisions
    # For a candidate set B of k vertices, non-adjacent pairs need unique internal vertices
    # This is a bipartite matching problem: pairs -> vertices
    
    def try_length3_subdivision(B):
        """Try to find a length-3 subdivision for branch set B.
        Returns (paths, branch_list) or None."""
        k = len(B)
        B_set = set(B)
        
        # Identify non-adjacent pairs
        non_adj_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                if B[j] not in adj[B[i]]:
                    non_adj_pairs.append((i, j))
        
        # Pigeonhole: need at least as many available vertices as non-adj pairs
        if len(non_adj_pairs) > n - k:
            return None
        
        # Build assignment: each non-adj pair needs a unique internal vertex
        # pair_to_verts[pair_idx] = list of valid internal vertices
        pair_to_verts = []
        for idx, (i, j) in enumerate(non_adj_pairs):
            valid = [w for w in common[B[i]][B[j]] if w not in B_set]
            pair_to_verts.append(valid)
            if not valid:
                return None
        
        # Kuhn's algorithm for bipartite matching
        # match_vertex[v] = pair_idx assigned to vertex v, or -1
        match_vertex = {}
        
        def try_kuhn(pair_idx, seen):
            for w in pair_to_verts[pair_idx]:
                if w in seen:
                    continue
                seen.add(w)
                if w not in match_vertex or try_kuhn(match_vertex[w], seen):
                    match_vertex[w] = pair_idx
                    return True
            return False
        
        for idx in range(len(non_adj_pairs)):
            if not try_kuhn(idx, set()):
                return None
        
        # Matching found! Construct paths
        paths = []
        # Build lookup: (i,j) -> internal vertex
        pair_internal = {}
        for w, pair_idx in match_vertex.items():
            i, j = non_adj_pairs[pair_idx]
            pair_internal[(i, j)] = w
        
        for i in range(k):
            for j in range(i + 1, k):
                if B[j] in adj[B[i]]:
                    paths.append([B[i], B[j]])
                else:
                    w = pair_internal[(i, j)]
                    paths.append([B[i], w, B[j]])
        
        return paths
    
    # Try k from 9 down to 3
    best_result = None
    
    for k in range(9, 2, -1):
        if time.time() > deadline:
            break
        
        # Candidate pool: top vertices by degree
        # For larger k, use a slightly larger pool to find valid sets
        pool_size = min(n, max(15, k + 5))
        candidates = sorted_vertices[:pool_size]
        
        # Generate combinations
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            
            B = list(combo)
            result = try_length3_subdivision(B)
            if result is not None:
                best_result = (B, result)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        branches, paths = best_result
        # Ensure proper types
        branches = [int(b) for b in branches]
        paths = [[int(v) for v in p] for p in paths]
        return {'branches': branches, 'paths': paths}
    
    # Fallback: use max clique
    if len(best_clique) >= 3:
        branches = sorted([int(b) for b in best_clique])
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Fallback: triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    for a, b, c in combinations(range(n), 3):
        if (min(a, b), max(a, b)) in edge_set and \
           (min(a, c), max(a, c)) in edge_set and \
           (min(b, c), max(b, c)) in edge_set:
            branches = sorted([a, b, c])
            paths = [[branches[0], branches[1]], [branches[0], branches[2]], [branches[1], branches[2]]]
            return {'branches': branches, 'paths': paths}
    
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
