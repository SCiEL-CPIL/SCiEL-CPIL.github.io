"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85.0

    def find_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def find_k_shortest_paths(src, dst, forbidden, k=3):
        """Find up to k shortest paths from src to dst avoiding forbidden."""
        if src == dst:
            return [[src]]
        # Use Yen's algorithm simplified: BFS with path tracking
        all_paths = []
        visited = {src: [src]}
        queue = deque([(src, [src])])
        while queue and len(all_paths) < k:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    new_path = path + [dst]
                    all_paths.append(new_path)
                    if len(all_paths) >= k:
                        return all_paths
                elif nb not in forbidden and nb not in visited:
                    visited[nb] = path + [nb]
                    queue.append((nb, path + [nb]))
        return all_paths

    def try_build_subdivision_csp(branches, deadline):
        """Try to find internally disjoint paths using CSP-style backtracking.
        For each pair, we have candidate paths. We need to assign one path per pair
        such that no internal vertex is used by more than one path."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        if not pairs:
            return []
        
        # For each pair, find candidate paths
        pair_candidates = {}
        for u, v in pairs:
            forbidden = branch_set - {u, v}
            # Direct edge is always a valid candidate
            if v in adj[u]:
                pair_candidates[(u, v)] = [[u, v]]
            else:
                paths = find_k_shortest_paths(u, v, forbidden, k=3)
                if not paths:
                    return None
                pair_candidates[(u, v)] = paths
        
        # Order pairs by number of candidates (most constrained first)
        pair_order = sorted(pairs, key=lambda p: len(pair_candidates[p]))
        
        # Backtracking search
        assignment = {}
        used_internal = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pair_order):
                return True
            
            u, v = pair_order[idx]
            candidates = pair_candidates[(u, v)]
            
            # Try each candidate path
            for path in candidates:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & branch_set:
                    continue
                
                # Assign this path
                assignment[(u, v)] = path
                used_internal.update(internal)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                used_internal.difference_update(internal)
                del assignment[(u, v)]
            
            return False
        
        if backtrack(0):
            return [assignment[p] for p in pairs]
        
        return None

    def try_build_subdivision_greedy(branches, deadline):
        """Greedy approach with multiple orderings."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        if not pairs:
            return []
        
        # Try multiple orderings
        for ordering_type in range(3):
            if ordering_type == 0:
                # Direct edges first
                pair_order = sorted(pairs, key=lambda p: (0 if p[1] in adj[p[0]] else 1, len(adj[p[0]]) + len(adj[p[1]])))
            elif ordering_type == 1:
                # Non-direct first
                pair_order = sorted(pairs, key=lambda p: (1 if p[1] in adj[p[0]] else 0, len(adj[p[0]]) + len(adj[p[1]])))
            else:
                pair_order = list(pairs)
                random.shuffle(pair_order)
            
            paths = {}
            used_internal = set()
            success = True
            
            for u, v in pair_order:
                if time.time() > deadline:
                    return None
                forbidden = branch_set - {u, v} | used_internal
                path = find_shortest_path(u, v, forbidden)
                if path is None:
                    success = False
                    break
                paths[(u, v)] = path
                for x in path[1:-1]:
                    used_internal.add(x)
            
            if success:
                return [paths[p] for p in pairs]
        
        return None

    def try_build_subdivision(branches, deadline):
        """Try multiple strategies to build subdivision."""
        # First try CSP (more powerful but slower)
        result = try_build_subdivision_csp(branches, deadline)
        if result is not None:
            return result
        # Then try greedy
        result = try_build_subdivision_greedy(branches, deadline)
        if result is not None:
            return result
        return None

    def find_max_clique(deadline):
        """Find maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        cliques.sort(key=len, reverse=True)
        return cliques

    # Phase 1: Find largest clique
    cliques = find_max_clique(deadline)
    best_result = None
    best_size = 0
    
    for clique in cliques:
        if time.time() > deadline:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        result = try_build_subdivision(clique, deadline)
        if result is not None:
            best_result = {'branches': list(clique), 'paths': result}
            best_size = len(clique)
    
    # Phase 2: Greedy expansion
    if best_size >= 2:
        base_branches = list(best_result['branches'])
    else:
        u, v = P['edges'][0]
        base_branches = [u, v]
        best_size = 2
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    current_branches = list(base_branches)
    
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        candidates = [v for v in range(n) if v not in set(current_branches)]
        candidates.sort(key=lambda v: len(adj[v]))
        random.shuffle(candidates)
        
        added = False
        for v in candidates:
            if time.time() > deadline - 3:
                break
            new_branches = current_branches + [v]
            result = try_build_subdivision(new_branches, deadline)
            if result is not None:
                current_branches = new_branches
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = {'branches': list(current_branches), 'paths': result}
                added = True
                break
        
        if not added:
            if len(current_branches) >= 3:
                swapped = False
                for i in range(len(current_branches)):
                    if time.time() > deadline - 3:
                        break
                    temp = current_branches[:i] + current_branches[i+1:]
                    cands2 = [v for v in range(n) if v not in set(temp)]
                    cands2.sort(key=lambda v: len(adj[v]))
                    random.shuffle(cands2)
                    for v in cands2[:25]:
                        if time.time() > deadline - 3:
                            break
                        new_b = temp + [v]
                        result = try_build_subdivision(new_b, deadline)
                        if result is not None:
                            current_branches = new_b
                            if len(current_branches) > best_size:
                                best_size = len(current_branches)
                                best_result = {'branches': list(current_branches), 'paths': result}
                            swapped = True
                            break
                    if swapped:
                        break
    
    # Phase 3: Try other cliques
    if time.time() < deadline - 5:
        for clique in cliques[:30]:
            if time.time() > deadline - 5:
                break
            if len(clique) < 2:
                continue
            rb = list(clique)
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                candidates = [v for v in range(n) if v not in set(rb)]
                candidates.sort(key=lambda v: len(adj[v]))
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    new_branches = rb + [v]
                    result = try_build_subdivision(new_branches, deadline)
                    if result is not None:
                        rb = new_branches
                        if len(rb) > best_size:
                            best_size = len(rb)
                            best_result = {'branches': list(rb), 'paths': result}
                        added = True
                        break
                if not added:
                    break
    
    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
