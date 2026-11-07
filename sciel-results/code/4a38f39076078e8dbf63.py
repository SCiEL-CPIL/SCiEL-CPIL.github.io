import time
import sys
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    n = P['n']
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def flow_based_verify(branches):
        """
        Use max-flow to verify if branches admit a clique subdivision.
        For each pair (u,v), we need vertex-disjoint paths.
        We model this as a multi-commodity flow, but since that's NP-hard,
        we use a sequential approach: for each pair, check if a path exists
        in the residual graph after removing used interior vertices.
        Returns paths dict if valid, None otherwise.
        """
        b_set = set(branches)
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(sorted(branches), 2))
        paths = {}
        used_interior = set()
        
        # Sort pairs: direct edges first, then by estimated "difficulty"
        def pair_difficulty(p):
            u, v = p
            if v in adj[u]:
                return (0, 0, u, v)
            # Count available interior vertices
            avail = 0
            for w in range(n):
                if w not in b_set and w not in used_interior:
                    if (w in adj[u] or w in adj[v]):
                        avail += 1
            return (1, -avail, u, v)
        
        # Try multiple orderings
        orderings = []
        # Default: by difficulty
        orderings.append(sorted(pairs, key=pair_difficulty))
        
        # Random orderings
        num_random = min(30, max(5, 150 // k))
        for _ in range(num_random):
            if time.time() > deadline:
                break
            ro = pairs[:]
            random.shuffle(ro)
            orderings.append(ro)
        
        for order in orderings:
            if time.time() > deadline:
                break
            paths = {}
            used_interior = set()
            success = True
            
            for u, v in order:
                if v in adj[u]:
                    paths[(u, v)] = [u, v]
                else:
                    # BFS for shortest path avoiding forbidden
                    forbidden = b_set - {u, v}
                    forbidden |= used_interior
                    visited = {u}
                    parent = {u: None}
                    queue = deque([u])
                    found = False
                    while queue:
                        cur = queue.popleft()
                        for nb in adj[cur]:
                            if nb not in visited and (nb == v or nb not in forbidden):
                                visited.add(nb)
                                parent[nb] = cur
                                if nb == v:
                                    found = True
                                    break
                                queue.append(nb)
                        if found:
                            break
                    
                    if not found:
                        success = False
                        break
                    
                    # Reconstruct path
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    paths[(u, v)] = path
                    for mid in path[1:-1]:
                        used_interior.add(mid)
            
            if success:
                return paths
        
        return None
    
    # Find initial branch set: start with a large clique, then extend
    # First, find max clique
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    
    current_branches = list(best_clique)
    current_paths = flow_based_verify(current_branches)
    
    if current_paths is None:
        # Fallback: try smaller subsets of the clique
        for size in range(len(current_branches) - 1, 1, -1):
            if time.time() > deadline:
                break
            for subset in combinations(sorted(current_branches), size):
                test = flow_based_verify(list(subset))
                if test is not None:
                    current_branches = list(subset)
                    current_paths = test
                    break
            if current_paths is not None:
                break
    
    if current_paths is None:
        # Last resort: just two adjacent vertices
        for u, v in P['edges']:
            current_branches = [u, v]
            current_paths = {(u, v): [u, v]} if u < v else {(v, u): [v, u]}
            break
    
    # Greedy extension: try adding vertices
    remaining = [v for v in range(n) if v not in set(current_branches)]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort remaining by "connectivity" to current branches
        def connectivity_score(v):
            score = 0
            for b in current_branches:
                if b in adj[v]:
                    score += 1
            return -score  # higher connectivity = lower score = tried first
        
        remaining.sort(key=connectivity_score)
        
        for v in remaining[:]:
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            test_paths = flow_based_verify(test_branches)
            if test_paths is not None:
                current_branches = test_branches
                current_paths = test_paths
                remaining.remove(v)
                improved = True
                break  # Restart loop with new set
    
    # Build result
    branches = [int(b) for b in sorted(current_branches)]
    paths_list = []
    for u, v in combinations(sorted(branches), 2):
        p = current_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
