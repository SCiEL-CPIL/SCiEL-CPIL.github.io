import time
import random
import sys
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    n = P['n']
    edge_order = P['edge_order']
    total_edges = len(edge_order)
    
    # Build position lookup: edge -> 1-indexed position
    pos = {}
    for i, (u, v) in enumerate(edge_order):
        if u > v:
            u, v = v, u
        pos[(u, v)] = i + 1
    
    def get_pos(u, v):
        if u > v:
            u, v = v, u
        return pos[(u, v)]
    
    def has_hamilton_cycle(k):
        """Check if subgraph with edges 1..k has a Hamilton cycle.
        Returns the cycle as a list of vertices if found, else None."""
        if time.time() > deadline:
            return None
        
        # Build adjacency for edges with position <= k
        adj = [set() for _ in range(n)]
        for i in range(k):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Check minimum degree
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        # Use bitmask backtracking with strong pruning
        full_mask = (1 << n) - 1
        
        # Precompute neighbor lists sorted by degree (MRV heuristic)
        adj_lists = [list(adj[v]) for v in range(n)]
        for v in range(n):
            adj_lists[v].sort(key=lambda x: len(adj[x]))
        
        # Connectivity check: verify all unvisited vertices are connected
        def is_connected(start, unvisited_mask):
            if unvisited_mask == 0:
                return True
            # BFS from start within unvisited vertices
            visited = 0
            stack = [start]
            while stack:
                v = stack.pop()
                if visited & (1 << v):
                    continue
                visited |= (1 << v)
                for nb in adj[v]:
                    if (unvisited_mask & (1 << nb)) and not (visited & (1 << nb)):
                        stack.append(nb)
            return visited == unvisited_mask
        
        def backtrack(v, mask, path, depth):
            if time.time() > deadline:
                return None
            
            if mask == full_mask:
                # Check if we can return to start (vertex 0)
                if 0 in adj[v]:
                    return path + [0]
                return None
            
            # Pruning: check connectivity of unvisited vertices
            unvisited = full_mask & ~mask
            if unvisited:
                # Find an unvisited vertex to check connectivity from
                check_v = (unvisited & -unvisited).bit_length() - 1
                if not is_connected(check_v, unvisited):
                    return None
            
            # Try neighbors sorted by degree (MRV)
            for nb in adj_lists[v]:
                if not (mask & (1 << nb)):
                    new_mask = mask | (1 << nb)
                    result = backtrack(nb, new_mask, path + [nb], depth + 1)
                    if result is not None:
                        return result
            
            return None
        
        # Try from vertex 0
        result = backtrack(0, 1, [0], 0)
        if result is not None:
            return result
        
        return None
    
    # Binary search for the minimum k where a Hamilton cycle exists
    def check_k(k):
        """Returns a Hamilton cycle using only edges with pos <= k, or None."""
        if time.time() > deadline:
            return None
        return has_hamilton_cycle(k)
    
    # Binary search
    lo, hi = 24, total_edges
    best_k = hi
    best_cycle = None
    
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = check_k(mid)
        if cycle is not None:
            hi = mid
            best_k = mid
            best_cycle = cycle
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Try the full range
        best_cycle = check_k(total_edges)
        if best_cycle is None:
            # Fallback: return identity
            return list(range(n))
    
    # Compute the actual max position of the found cycle
    def cycle_max_pos(cycle):
        max_p = 0
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            p = get_pos(u, v)
            if p > max_p:
                max_p = p
        return max_p
    
    current_best = cycle_max_pos(best_cycle)
    
    # Local search: try to reduce max edge position
    def try_improve(cycle):
        nonlocal current_best
        best_c = list(cycle)
        best_score = cycle_max_pos(best_c)
        
        # Try 2-opt style moves: reverse a segment
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(best_c) - 1):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 2, len(best_c)):
                    if j == len(best_c) - 1 and i == 0:
                        continue
                    new_c = best_c[:i+1] + best_c[i+1:j+1][::-1] + best_c[j+1:]
                    new_score = cycle_max_pos(new_c)
                    if new_score < best_score:
                        best_c = new_c
                        best_score = new_score
                        improved = True
                        if best_score < current_best:
                            current_best = best_score
        
        return best_c, best_score
    
    # Try to improve with local search
    if time.time() < deadline - 5:
        best_cycle, current_best = try_improve(best_cycle)
    
    # Final answer: return the cycle as a permutation (without the closing edge)
    result = best_cycle[:-1]
    
    # Verify it's a valid permutation
    assert len(result) == n
    assert set(result) == set(range(n))
    
    return result
# EVOLVE-BLOCK-END
