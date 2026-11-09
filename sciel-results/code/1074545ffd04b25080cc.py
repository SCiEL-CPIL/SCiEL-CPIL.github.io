import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 82.0


def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 2.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    def max_flow_disjoint_paths(branches):
        """
        Given a set of branch vertices, use max-flow to check if we can find
        vertex-disjoint paths (excluding branch vertices as internal nodes)
        connecting every pair of branch vertices.
        
        Returns list of paths if successful, None otherwise.
        """
        bs = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)
        
        # Build flow network:
        # - Each non-branch vertex v is split into v_in -> v_out with capacity 1
        # - Each branch vertex b is a super source/sink connection
        # - For each pair (u,v), we add a demand of 1 unit of flow from u to v
        # 
        # Actually, this is a multi-commodity flow problem, which is NP-hard.
        # Instead, we'll use a different approach:
        # 
        # We'll model it as: find vertex-disjoint paths for all pairs.
        # This can be checked by:
        # 1. For each pair, find a path
        # 2. Ensure all internal vertices are distinct
        #
        # A practical approach: use backtracking with pruning, but with
        # better heuristics and a time limit.
        
        # Alternative: Use the fact that for small branch sets, we can
        # try all possible path assignments with backtracking.
        
        # Let's use a smarter backtracking:
        # - Order pairs by "difficulty" (fewest available paths first)
        # - For each pair, find all simple paths (with length limit)
        # - Try each path, ensuring vertex disjointness
        
        # First, precompute all simple paths between each pair with length <= 4
        # (direct edge, length 2, length 3, length 4)
        
        all_paths = {}
        for u, v in pairs:
            paths = _find_short_paths(u, v, bs, max_len=4)
            all_paths[(u, v)] = paths
        
        # Check if any pair has no path
        for (u, v) in pairs:
            if not all_paths[(u, v)]:
                return None
        
        # Order pairs by number of available paths (most constrained first)
        ordered_pairs = sorted(pairs, key=lambda p: len(all_paths[p]))
        
        # Backtracking
        solution = {}
        used_internal = set()
        
        def bt(idx):
            if idx == len(ordered_pairs):
                return True
            
            u, v = ordered_pairs[idx]
            for path in all_paths[(u, v)]:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                
                # Check if path is valid (all consecutive vertices adjacent)
                valid = True
                for i in range(len(path) - 1):
                    if path[i+1] not in adj[path[i]]:
                        valid = False
                        break
                if not valid:
                    continue
                
                solution[(u, v)] = path
                used_internal.update(internal)
                
                if bt(idx + 1):
                    return True
                
                used_internal.difference_update(internal)
                del solution[(u, v)]
            
            return False
        
        if bt(0):
            # Reconstruct paths in standard order
            result = []
            for u, v in pairs:
                result.append(solution[(u, v)])
            return result
        
        return None

    def _find_short_paths(s, t, branch_set, max_len=4):
        """Find all simple paths from s to t with at most max_len edges,
        avoiding branch_set vertices as internal nodes."""
        paths = []
        if s == t:
            return [[s]]
        
        def dfs(curr, path, visited):
            if len(paths) >= 20:  # Limit to avoid explosion
                return
            if len(path) - 1 > max_len:
                return
            if curr == t:
                paths.append(list(path))
                return
            for nb in adj[curr]:
                if nb in visited:
                    continue
                if nb in branch_set and nb != s and nb != t:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
        
        dfs(s, [s], {s})
        return paths

    def local_search(branches, max_iterations=500):
        """
        Start with a valid branch set and try to improve by:
        1. Adding a new vertex
        2. Swapping a vertex with another
        """
        current = set(branches)
        current_paths = max_flow_disjoint_paths(list(current))
        if current_paths is None:
            return None, None
        
        best_set = set(current)
        best_paths = current_paths
        best_size = len(current)
        
        vertices = list(range(n))
        
        for iteration in range(max_iterations):
            if time_up():
                break
            
            improved = False
            
            # Try adding a vertex
            for v in vertices:
                if v in current:
                    continue
                new_set = current | {v}
                new_paths = max_flow_disjoint_paths(list(new_set))
                if new_paths is not None:
                    current = new_set
                    current_paths = new_paths
                    best_set = set(current)
                    best_paths = new_paths
                    best_size = len(current)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping a vertex
            for v_in in list(current):
                for v_out in vertices:
                    if v_out in current:
                        continue
                    new_set = (current - {v_in}) | {v_out}
                    new_paths = max_flow_disjoint_paths(list(new_set))
                    if new_paths is not None and len(new_set) > best_size:
                        current = new_set
                        current_paths = new_paths
                        best_set = set(current)
                        best_paths = new_paths
                        best_size = len(current)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_set, best_paths

    best_set = None
    best_paths = None
    best_size = 0
    
    # Start with a 2-branch solution
    u, v = edges[0]
    initial = {u, v}
    paths = max_flow_disjoint_paths([u, v])
    if paths is not None:
        best_set = initial
        best_paths = paths
        best_size = 2
    
    # Try to build up from high-degree vertices
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    # Try various initial sets and use local search
    random.seed(42)
    
    # Strategy 1: Start with top-k by degree, try different k
    for k in range(3, min(n, 15) + 1):
        if time_up():
            break
        if k <= best_size:
            continue
        
        # Try a few random subsets of size k from top candidates
        candidates = deg_sorted[:min(k * 2, n)]
        for attempt in range(10):
            if time_up():
                break
            if len(candidates) < k:
                break
            subset = set(random.sample(candidates, k))
            paths = max_flow_disjoint_paths(list(subset))
            if paths is not None:
                # Use local search to improve
                improved_set, improved_paths = local_search(list(subset), max_iterations=200)
                if improved_set is not None and len(improved_set) > best_size:
                    best_set = improved_set
                    best_paths = improved_paths
                    best_size = len(improved_set)
                elif len(subset) > best_size:
                    best_set = subset
                    best_paths = paths
                    best_size = len(subset)
    
    # Strategy 2: Try random subsets of various sizes
    for size in range(best_size + 1, min(n, 15) + 1):
        if time_up():
            break
        for attempt in range(20):
            if time_up():
                break
            subset = set(random.sample(range(n), size))
            paths = max_flow_disjoint_paths(list(subset))
            if paths is not None:
                improved_set, improved_paths = local_search(list(subset), max_iterations=100)
                if improved_set is not None and len(improved_set) > best_size:
                    best_set = improved_set
                    best_paths = improved_paths
                    best_size = len(improved_set)
    
    if best_set is None or best_paths is None:
        return fallback()
    
    # Format output
    branches_sorted = sorted(best_set)
    pairs = list(combinations(branches_sorted, 2))
    
    # Reorder paths to match standard combination order
    path_map = {}
    for i, (u, v) in enumerate(pairs):
        # Find the path for this pair
        for p in best_paths:
            if p[0] == u and p[-1] == v:
                path_map[(u, v)] = p
                break
            elif p[0] == v and p[-1] == u:
                path_map[(u, v)] = list(reversed(p))
                break
    
    final_paths = []
    for u, v in pairs:
        if (u, v) in path_map:
            final_paths.append(path_map[(u, v)])
        else:
            # Should not happen, but fallback
            return fallback()
    
    return {
        'branches': [int(x) for x in branches_sorted],
        'paths': [[int(x) for x in p] for p in final_paths]
    }
