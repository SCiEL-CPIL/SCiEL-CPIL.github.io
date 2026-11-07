import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85  # safety margin
    
    # Sort vertices by degree descending
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    best_result = None
    best_k = 2
    
    # Upper bound on k: we need binom(k,2) paths, each using at least 1 internal vertex (or 0 if adjacent)
    # But internal vertices must be disjoint. Max internal vertices = n - k
    # So binom(k,2) <= n - k  =>  k(k-1)/2 <= n - k  =>  k^2 + k - 2n <= 0
    # For n=24: k^2 + k - 48 <= 0 => k <= ~6.3, so k <= 6 is a loose bound.
    # But paths can be length 1 (no internal vertex), so the bound is weaker.
    # Let's just try k from 10 down to 3, but limit candidate sets.
    
    # Try k from high to low
    for k in range(10, 2, -1):
        if time.time() > deadline:
            break
        if best_result is not None and k <= best_k:
            break
        
        # Candidate vertices: top min(n, 2*k) by degree, plus maybe all if n small
        cand_limit = min(n, max(2 * k, k + 3))
        candidates = order[:cand_limit]
        
        # Enumerate combinations of k from candidates
        for B in combinations(candidates, k):
            if time.time() > deadline:
                break
            B_set = set(B)
            # Try to assign disjoint paths
            paths = {}
            used_internal = set()
            pairs = list(combinations(B, 2))
            
            # Order pairs: maybe by difficulty (longest path first?) or just fixed
            # Try to find paths greedily with backtracking
            success = True
            paths_list = []
            
            # Use recursive backtracking
            def backtrack(idx, used_int, paths_acc):
                if time.time() > deadline:
                    return False
                if idx == len(pairs):
                    return True
                u, v = pairs[idx]
                # Find a simple path from u to v avoiding used_int and B_set (except endpoints)
                # BFS with visited
                visited = set(B_set) | used_int
                visited.discard(u)
                visited.discard(v)
                # BFS
                queue = deque([(u, [u])])
                found = False
                while queue:
                    cur, path = queue.popleft()
                    if cur == v:
                        # Check if path is valid (simple, endpoints correct)
                        # path[0] should be u, path[-1] should be v
                        if path[0] == u and path[-1] == v:
                            internal = set(path[1:-1])
                            if not internal & (used_int | B_set):
                                paths_acc.append(path)
                                used_int_new = used_int | internal
                                if backtrack(idx + 1, used_int_new, paths_acc):
                                    return True
                                paths_acc.pop()
                        found = True
                        break
                    for nb in adj[cur]:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
                return False
            
            if backtrack(0, set(), paths_list):
                best_result = {
                    'branches': list(B),
                    'paths': [list(p) for p in paths_list]
                }
                best_k = k
                break  # found for this k, go to smaller k? No, we want max, so break inner and outer
            else:
                continue
        
        if best_result is not None and best_k == k:
            break
    
    if best_result is None:
        # Fallback: find any triangle or edge
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_result = {'branches': [u, v], 'paths': [[u, v]]}
                    break
            if best_result:
                break
    
    # Ensure types
    branches = [int(x) for x in best_result['branches']]
    paths = [[int(x) for x in p] for p in best_result['paths']]
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
