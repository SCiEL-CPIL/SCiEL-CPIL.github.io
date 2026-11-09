import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def time_up():
        return time.time() - t0 > DEADLINE - 5.0  # Larger safety margin

    # Quick fallback
    u, v = edges[0]
    best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    best_size = 2
    
    # Find maximum clique using optimized Bron-Kerbosch with pivot
    def max_clique_size():
        """Find the maximum clique size using Bron-Kerbosch."""
        best_clique = []
        
        def bk(R, P_set, X):
            nonlocal best_clique
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            # Pivot selection
            candidates = P_set | X
            if candidates:
                pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
                to_try = P_set - adj[pivot]
            else:
                to_try = P_set
            for v in list(to_try):
                bk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return best_clique

    # Fast path finding heuristic
    def find_disjoint_paths(branches):
        """
        Try to find pairwise internally disjoint paths for all pairs in branches.
        Uses a greedy/randomized approach with strict time limits.
        Returns list of paths or None.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        if not pairs:
            return []
        
        # If all pairs are directly connected, paths are trivial
        all_direct = all(v in adj[u] for u, v in pairs)
        if all_direct:
            return [[u, v] for u, v in pairs]
        
        direct_pairs = [(u, v) for u, v in pairs if v in adj[u]]
        non_direct_pairs = [(u, v) for u, v in pairs if v not in adj[u]]
        
        # Start with direct pairs
        paths = {}
        used_internal = set()
        for u, v in direct_pairs:
            paths[(u, v)] = [u, v]
        
        remaining = list(non_direct_pairs)
        
        # Shuffle remaining to allow randomization
        rng = random.Random(42)
        
        # Try multiple random orderings
        for attempt in range(10):
            if time_up():
                return None
            
            # Reset
            paths = {}
            for u, v in direct_pairs:
                paths[(u, v)] = [u, v]
            used_internal = set()
            
            # Order remaining pairs
            curr_remaining = list(remaining)
            if attempt > 0:
                rng.shuffle(curr_remaining)
            
            success = True
            for u, v in curr_remaining:
                # Find a path from u to v avoiding used_internal and bs
                # Use BFS for shortest path
                path = None
                # Try shortest path first
                def bfs_path(s, t, blocked):
                    if t in adj[s] and t not in blocked:
                        return [s, t]
                    parent = {s: None}
                    q = deque([s])
                    while q:
                        curr = q.popleft()
                        for nb in adj[curr]:
                            if nb == t:
                                if t not in blocked:
                                    parent[t] = curr
                                    path = [t]
                                    node = t
                                    while parent[node] is not None:
                                        node = parent[node]
                                        path.append(node)
                                    path.reverse()
                                    return path
                                else:
                                    continue
                            if nb not in blocked and nb not in parent:
                                parent[nb] = curr
                                q.append(nb)
                    return None
                
                path = bfs_path(u, v, used_internal | bs)
                
                if path is None:
                    # Try a slightly longer path? Or just fail this attempt.
                    # To keep it fast, we just fail this ordering.
                    success = False
                    break
                
                internal = set(path[1:-1])
                if internal & used_internal or internal & bs:
                    success = False
                    break
                
                paths[(u, v)] = path
                used_internal.update(internal)
            
            if success:
                # Build result
                result = []
                for u, v in pairs:
                    if (u, v) in paths:
                        result.append(paths[(u, v)])
                    elif (v, u) in paths:
                        result.append(list(reversed(paths[(v, u)])))
                    else:
                        return None # Should not happen if success is True
                return result
        
        return None

    # Phase 1: Find maximum clique
    max_cli = max_clique_size()
    if len(max_cli) > best_size:
        cli_sorted = sorted(max_cli)
        paths = find_disjoint_paths(cli_sorted)
        if paths is not None:
            best = {'branches': [int(x) for x in cli_sorted], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(cli_sorted)
    
    # Phase 2: Try to extend the best clique/set
    # We try to add vertices one by one to the current best set
    current_best_set = set(best['branches'])
    
    # Sort candidates by degree or connectivity
    candidates = [v for v in range(n) if v not in current_best_set]
    candidates.sort(key=lambda x: len(adj[x]), reverse=True)
    
    improved = True
    while improved and not time_up():
        improved = False
        for v in candidates:
            if time_up():
                break
            new_set = current_best_set | {v}
            paths = find_disjoint_paths(sorted(new_set))
            if paths is not None:
                best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                best_size = len(new_set)
                current_best_set = new_set
                # Remove v from candidates
                candidates.remove(v)
                improved = True
                break # Restart loop with new best set
    
    return best
