"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Degrees for heuristic ordering
    degrees = [len(adj[i]) for i in range(n)]
    
    # 1. Find Max Clique (Baseline)
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
    
    # If max clique is already large, return it
    if len(best_clique) >= 5:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}

    # 2. Try to find a K5 or K6 subdivision
    # Heuristic: Branches should be high-degree vertices
    # Sort vertices by degree descending
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    # We will try to find a valid subdivision for k=6, then k=5
    # Time limit 60s, use 50s for search
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_subdivision = None
    
    # Function to check if a set of branches has a valid subdivision
    def check_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
        
        # Create list of pairs, ordered by difficulty (fewer direct connections first)
        all_pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        
        # Order pairs: pairs with no direct edge first (harder), then by number of common neighbors
        def pair_difficulty(pair):
            i, j = pair
            b_i, b_j = branches[i], branches[j]
            # No direct edge = harder
            direct = 1 if b_j in adj[b_i] else 0
            # Fewer common non-branch neighbors = harder
            common = len((adj[b_i] & adj[b_j]) - set(branches))
            return (direct, common)
        
        pairs = sorted(all_pairs, key=pair_difficulty)
        
        used = set()
        paths = {}
        
        def find_paths(b_i, b_j, forbidden, max_paths=3):
            """Find up to max_paths simple paths from b_i to b_j avoiding forbidden.
            Returns list of paths ordered by length (shortest first)."""
            from collections import deque
            result_paths = []
            seen_paths = set()
            
            # BFS with path tracking, collect multiple shortest paths
            q = deque()
            q.append((b_i, [b_i]))
            visited = {b_i}
            
            while q and len(result_paths) < max_paths:
                curr, path = q.popleft()
                if curr == b_j:
                    path_tuple = tuple(path)
                    if path_tuple not in seen_paths:
                        seen_paths.add(path_tuple)
                        result_paths.append(path)
                    continue
                for nb in sorted(adj[curr]):  # Sort for determinism
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        q.append((nb, path + [nb]))
            
            return result_paths
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(pairs):
                # Success
                nonlocal best_subdivision
                res_paths = []
                for i in range(k):
                    for j in range(i+1, k):
                        res_paths.append(paths[(i,j)])
                best_subdivision = (branches, res_paths)
                return True
                
            i, j = pairs[idx]
            b_i = branches[i]
            b_j = branches[j]
            
            forbidden = used.copy()
            for b in branches:
                if b != b_i and b != b_j:
                    forbidden.add(b)
            
            # Try multiple path options
            path_options = find_paths(b_i, b_j, forbidden, max_paths=5)
            
            for path_found in path_options:
                internal = set(path_found[1:-1])
                
                paths[(i,j)] = path_found
                used.update(internal)
                
                if backtrack(idx + 1):
                    return True
                
                used.difference_update(internal)
                del paths[(i,j)]
            
            return False
        
        return backtrack(0)

    # Try k=6 first
    found = False
    candidates6 = sorted_vertices[:10]
    for combo in combinations(candidates6, 6):
        if time.time() > deadline:
            break
        branches = list(combo)
        if check_subdivision(branches):
            found = True
            break
            
    if found and best_subdivision:
        branches, paths = best_subdivision
        return {'branches': branches, 'paths': paths}

    # Try k=5
    candidates5 = sorted_vertices[:12]
    for combo in combinations(candidates5, 5):
        if time.time() > deadline:
            break
        branches = list(combo)
        if check_subdivision(branches):
            found = True
            break
            
    if found and best_subdivision:
        branches, paths = best_subdivision
        return {'branches': branches, 'paths': paths}

    # Fallback to Max Clique
    if len(best_clique) >= 3:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Fallback to Triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
        
    for a, b, c in combinations(range(n), 3):
        if (min(a,b), max(a,b)) in edge_set and (min(a,c), max(a,c)) in edge_set and (min(b,c), max(b,c)) in edge_set:
            branches = sorted([a,b,c])
            paths = [[branches[0], branches[1]], [branches[0], branches[2]], [branches[1], branches[2]]]
            return {'branches': branches, 'paths': paths}
            
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
