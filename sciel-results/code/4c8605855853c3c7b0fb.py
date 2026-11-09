import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 50.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def time_up():
        return time.time() - t0 > DEADLINE - 2.0
    
    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Max-flow based verification using vertex splitting
    # For a set of branches B, we need k(k-1)/2 pairwise internally disjoint paths
    # Model: split each non-branch vertex v into v_in -> v_out with capacity 1
    # For each pair (u,v) in B, we need a path. We can model this as:
    # Create a super-source connected to each branch, and check if we can route
    # all pairs simultaneously. Actually, this is the disjoint paths problem which
    # is NP-hard in general, but for small k we can use backtracking.
    
    # Better approach: Use the fact that for k branches, we need C(k,2) paths.
    # We'll use backtracking with BFS for path finding, but with better heuristics.
    
    def find_paths_for_branches(branches):
        """
        Returns a list of paths for all pairs in branches if valid, else None.
        Uses backtracking with BFS shortest path finding.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        
        if not pairs:
            return []
            
        # Order pairs: most constrained first (fewest common neighbors)
        def pair_constraint(p):
            u, v = p
            common = (adj[u] & adj[v]) - bs
            return len(common)
            
        # Try a few orderings
        orderings = [
            sorted(pairs, key=pair_constraint),
            sorted(pairs, key=lambda p: -pair_constraint(p)),
            list(pairs)
        ]
        
        rng = random.Random(42)
        for _ in range(3):
            sh = list(pairs)
            rng.shuffle(sh)
            orderings.append(sh)
            
        def find_shortest_path(s, t, blocked, rng_inst):
            """BFS shortest path from s to t, internal vertices avoid blocked."""
            if s == t:
                return [s]
            parent = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                if curr == t:
                    break
                neighbors = list(adj[curr])
                rng_inst.shuffle(neighbors) # Randomize order to find different shortest paths
                for nb in neighbors:
                    if nb in blocked and nb != s and nb != t:
                        continue
                    if nb in parent:
                        continue
                    parent[nb] = curr
                    if nb == t:
                        break
                    q.append(nb)
            if t not in parent:
                return None
            path = []
            node = t
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
            
        def solve(ordering):
            chosen = [None] * len(ordering)
            
            def backtrack(idx, used_internal):
                if time_up():
                    return False
                if idx == len(ordering):
                    return True
                u, v = ordering[idx]
                blocked = used_internal | bs
                # Try to find a short path
                path = find_shortest_path(u, v, blocked, rng)
                if path is None:
                    return False
                internal = set(path[1:-1])
                if internal & used_internal or internal & bs:
                    return False
                
                chosen[idx] = path
                for vn in internal:
                    used_internal.add(vn)
                    
                if backtrack(idx + 1, used_internal):
                    return True
                    
                # Backtrack
                for vn in internal:
                    used_internal.discard(vn)
                chosen[idx] = None
                return False
                
            if backtrack(0, set()):
                # Reorder to standard pair order
                std_pairs = list(combinations(sorted_bs, 2))
                path_map = {}
                for i, pair in enumerate(ordering):
                    path_map[pair] = chosen[i]
                result = []
                for p in std_pairs:
                    if p in path_map:
                        result.append(path_map[p])
                    else:
                        return None
                return result
            return None
            
        for ordering in orderings:
            if time_up():
                break
            res = solve(ordering)
            if res is not None:
                return res
        return None
    
    best = None
    best_size = 2
    
    # Fallback
    u, v = edges[0]
    best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    # Try to find a set of size K
    # Start with K=3, go up to maybe 9
    # Use random sampling from top degree nodes
    
    max_k_to_try = 9
    pool_size = 15
    
    for k in range(3, max_k_to_try + 1):
        if time_up():
            break
            
        found = False
        # Try multiple random subsets of size k from the top pool
        num_tries = 20 if k <= 6 else 10
        
        for _ in range(num_tries):
            if time_up():
                break
            
            # Sample k distinct nodes from top pool
            subset = random.sample(deg_sorted[:pool_size], k)
            
            paths = find_paths_for_branches(subset)
            if paths is not None:
                best = {'branches': [int(x) for x in sorted(subset)], 'paths': [[int(x) for x in p] for p in paths]}
                best_size = k
                found = True
                # If we found a valid set of size k, we can break and try k+1
                break
                
        if not found and k > 3:
            # If we couldn't find size k, it's unlikely we find k+1
            break
            
    return best
