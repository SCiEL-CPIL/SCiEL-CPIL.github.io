"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_paths_for_branches(branches, deadline):
        """Given a set of branch vertices, try to find internally disjoint paths
        between every pair using backtracking with BFS."""
        branch_set = set(branches)
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(itertools.combinations(branches, 2))
        paths = {}
        
        def bfs_path(src, dst, forbidden):
            """Find shortest simple path from src to dst avoiding forbidden."""
            if src == dst:
                return None
            visited = {src}
            parent = {src: None}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                if cur == dst:
                    path = []
                    node = dst
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                for nb in adj[cur]:
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        parent[nb] = cur
                        queue.append(nb)
            return None
        
        def get_candidate_paths(u, v, forbidden, max_len=4):
            """Get candidate paths from u to v of length up to max_len, avoiding forbidden."""
            candidates = []
            # Direct edge
            if v in adj[u]:
                candidates.append([u, v])
            # Length 3: u-x-v
            for x in adj[u]:
                if x not in forbidden and x not in (u, v) and v in adj[x]:
                    candidates.append([u, x, v])
            # Length 4: u-x-y-v
            if max_len >= 4:
                for x in adj[u]:
                    if x in forbidden or x == v:
                        continue
                    for y in adj[x]:
                        if y in forbidden or y == u or y == x:
                            continue
                        if v in adj[y]:
                            candidates.append([u, x, y, v])
            # Length 5: u-x-y-z-v
            if max_len >= 5:
                for x in adj[u]:
                    if x in forbidden or x == v:
                        continue
                    for y in adj[x]:
                        if y in forbidden or y == u or y == x:
                            continue
                        for z in adj[y]:
                            if z in forbidden or z == u or z == x or z == y:
                                continue
                            if v in adj[z]:
                                candidates.append([u, x, y, z, v])
            return candidates
        
        def backtrack(idx, used_internal):
            if time.time() > deadline:
                return False
            if idx == len(pairs):
                return True
            u, v = pairs[idx]
            forbidden = used_internal | (branch_set - {u, v})
            
            cands = get_candidate_paths(u, v, forbidden)
            # Sort by length (prefer shorter paths to leave more room)
            cands.sort(key=len)
            
            for path in cands:
                interior = set(path[1:-1])
                if interior & forbidden:
                    continue
                paths[(u, v)] = path
                if backtrack(idx + 1, used_internal | interior):
                    return True
                del paths[(u, v)]
            
            return False
        
        # Order pairs: adjacent pairs first, then by number of common neighbors
        pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1, -len(adj[p[0]] & adj[p[1]])))
        
        if backtrack(0, set()):
            result_paths = [paths[u, v] for u, v in pairs]
            return result_paths
        return None
    
    def find_best_subdivision(deadline):
        best_result = None
        best_k = 0
        
        # Find all cliques
        def find_cliques(min_size=2, max_size=12):
            cliques = []
            def expand(current, candidates):
                if time.time() > deadline:
                    return
                if len(current) >= min_size:
                    cliques.append(list(current))
                if len(current) >= max_size:
                    return
                for i, v in enumerate(candidates):
                    new_candidates = [w for w in candidates[i+1:] if w in adj[v]]
                    expand(current + [v], new_candidates)
            expand([], list(range(n)))
            return cliques
        
        cliques = find_cliques(2, 12)
        cliques.sort(key=len, reverse=True)
        
        for clique in cliques:
            if time.time() > deadline:
                break
            current_branches = list(clique)
            
            # Try to extend greedily
            extended = True
            while extended:
                extended = False
                if time.time() > deadline:
                    break
                candidates = [v for v in range(n) if v not in current_branches]
                candidates.sort(key=lambda v: len(adj[v]), reverse=True)
                
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    new_branches = current_branches + [cand]
                    paths = find_paths_for_branches(new_branches, deadline)
                    if paths is not None:
                        current_branches = new_branches
                        extended = True
                        if len(current_branches) > best_k:
                            best_k = len(current_branches)
                            best_result = {'branches': list(current_branches), 'paths': paths}
                        break
        
        # Try random starts
        if best_k < 6:
            import random
            rng = random.Random(123)
            for attempt in range(100):
                if time.time() > deadline:
                    break
                u, v = P['edges'][rng.randint(0, len(P['edges'])-1)]
                current_branches = [u, v]
                extended = True
                while extended:
                    extended = False
                    if time.time() > deadline:
                        break
                    candidates = [v for v in range(n) if v not in current_branches]
                    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
                    for cand in candidates:
                        if time.time() > deadline:
                            break
                        new_branches = current_branches + [cand]
                        paths = find_paths_for_branches(new_branches, deadline)
                        if paths is not None:
                            current_branches = new_branches
                            extended = True
                            if len(current_branches) > best_k:
                                best_k = len(current_branches)
                                best_result = {'branches': list(current_branches), 'paths': paths}
                            break
        
        if best_result is not None:
            return best_result
        
        # Fallback: triangle
        es = {tuple(e) for e in P['edges']}
        es = es | {(b, a) for a, b in es}
        for row in itertools.combinations(range(n), 3):
            if all(e in es for e in itertools.combinations(row, 2)):
                return {'branches': list(row), 'paths': [list(e) for e in itertools.combinations(row, 2)]}
        
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    result = find_best_subdivision(deadline)
    
    if result is None:
        u, v = P['edges'][0]
        result = {'branches': [u, v], 'paths': [[u, v]]}
    
    branches = [int(b) for b in result['branches']]
    paths = [[int(v) for v in p] for p in result['paths']]
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
