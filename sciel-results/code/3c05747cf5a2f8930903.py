import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
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
    
    # Precompute all edges with positions
    all_edges = []
    for i, (u, v) in enumerate(edge_order):
        all_edges.append((i + 1, u, v))
    
    def cycle_score(cycle):
        """Compute the max edge position in a Hamilton cycle."""
        max_p = 0
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            p = get_pos(u, v)
            if p > max_p:
                max_p = p
        # Closing edge
        u, v = cycle[-1], cycle[0]
        p = get_pos(u, v)
        if p > max_p:
            max_p = p
        return max_p
    
    def has_hamilton_cycle_fast(k):
        """Check if subgraph with edges 1..k has a Hamilton cycle using bitmask DP.
        Returns the cycle as a list of vertices if found, else None."""
        if time.time() > deadline:
            return None
        
        # Build adjacency bitmask for edges with position <= k
        adj_mask = [0] * n
        for i in range(k):
            u, v = edge_order[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Check minimum degree
        for v in range(n):
            if bin(adj_mask[v]).count('1') < 2:
                return None
        
        full_mask = (1 << n) - 1
        
        # Use iterative DFS with bitmask state
        # State: (current_vertex, visited_mask)
        # We fix vertex 0 as start to break symmetry
        
        # Backtracking with pruning
        # Use a stack-based approach to avoid recursion depth issues
        
        # Precompute neighbor lists sorted by degree for MRV heuristic
        deg = [bin(adj_mask[v]).count('1') for v in range(n)]
        neighbors = [[] for _ in range(n)]
        for v in range(n):
            nbrs = []
            m = adj_mask[v]
            while m:
                b = (m & (-m)).bit_length() - 1
                nbrs.append(b)
                m &= m - 1
            nbrs.sort(key=lambda x: deg[x])
            neighbors[v] = nbrs
        
        # DFS with memoization of failed states
        failed = set()
        
        def backtrack(v, mask, path):
            if time.time() > deadline:
                return None
            
            if mask == full_mask:
                # Check if we can return to start (vertex 0)
                if adj_mask[v] & 1:  # vertex 0 is in adj_mask[v]
                    return path + [0]
                return None
            
            # Try neighbors in MRV order
            for nb in neighbors[v]:
                if not (mask & (1 << nb)):
                    new_mask = mask | (1 << nb)
                    # Check if this state has been tried and failed
                    state = (nb, new_mask)
                    if state in failed:
                        continue
                    result = backtrack(nb, new_mask, path + [nb])
                    if result is not None:
                        return result
                    failed.add(state)
            
            return None
        
        result = backtrack(0, 1, [0])
        return result
    
    def local_search_2opt(cycle, max_iter=50000):
        """Improve cycle using 2-opt moves to reduce max edge position."""
        best_c = list(cycle)
        best_score = cycle_score(best_c)
        n = len(best_c)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try 2-opt: reverse a segment
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if i == 0 and j == n - 1:
                        continue  # same cycle
                    
                    # Compute new score efficiently
                    # Only edges affected are those at positions i-1, i, j-1, j
                    # (with wraparound)
                    
                    # Old edges: (best_c[i-1], best_c[i]) and (best_c[j-1], best_c[j])
                    # and closing if i==0 or j==n-1
                    # New edges: (best_c[i-1], best_c[j]) and (best_c[i], best_c[j-1])
                    
                    # For simplicity, just compute full score (n=24 is small)
                    new_c = best_c[:i+1] + best_c[i+1:j+1][::-1] + best_c[j+1:]
                    new_score = cycle_score(new_c)
                    
                    if new_score < best_score:
                        best_c = new_c
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_c, best_score
    
    def greedy_construct(k):
        """Greedy construction of a Hamilton cycle using edges with pos <= k.
        Returns cycle or None."""
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
        
        # Greedy: start at vertex 0, always go to unvisited neighbor with lowest position
        visited = set([0])
        path = [0]
        current = 0
        
        for _ in range(n - 1):
            # Find unvisited neighbors of current
            candidates = [v for v in adj[current] if v not in visited]
            if not candidates:
                return None
            
            # Choose the one with lowest edge position
            candidates.sort(key=lambda v: get_pos(current, v))
            next_v = candidates[0]
            
            # But we need to ensure the rest can be completed
            # Simple check: next_v must have at least 2 unvisited neighbors or be the last
            remaining = n - len(visited) - 1
            if remaining > 0:
                unvisited_nbrs = [v for v in adj[next_v] if v not in visited and v != next_v]
                if len(unvisited_nbrs) < 1:
                    # Try next candidate
                    found = False
                    for cand in candidates[1:]:
                        unv = [v for v in adj[cand] if v not in visited and v != cand]
                        if len(unv) >= 1:
                            next_v = cand
                            found = True
                            break
                    if not found:
                        return None
            
            path.append(next_v)
            visited.add(next_v)
            current = next_v
        
        # Check if we can close the cycle
        if 0 in adj[current]:
            return path
        return None
    
    # Strategy: Binary search for minimum k, then local search to improve
    best_cycle = None
    best_k = total_edges
    
    # First, try greedy for various k values to get a quick solution
    for k in range(24, total_edges + 1, 5):
        if time.time() > deadline - 10:
            break
        cycle = greedy_construct(k)
        if cycle is not None:
            best_cycle = cycle
            best_k = k
            break
    
    # If greedy didn't find one, use backtracking with binary search
    if best_cycle is None:
        lo, hi = 24, total_edges
        while lo < hi and time.time() < deadline - 5:
            mid = (lo + hi) // 2
            cycle = has_hamilton_cycle_fast(mid)
            if cycle is not None:
                hi = mid
                best_k = mid
                best_cycle = cycle
            else:
                lo = mid + 1
        
        if best_cycle is None:
            best_cycle = has_hamilton_cycle_fast(total_edges)
            if best_cycle is None:
                return list(range(n))
    
    # Now improve with local search
    if time.time() < deadline - 5:
        best_cycle, best_score = local_search_2opt(best_cycle, max_iter=100000)
    
    # Try additional local search passes with different starting points
    if time.time() < deadline - 3:
        # Try rotating the cycle and doing local search
        for shift in range(1, n):
            if time.time() > deadline - 2:
                break
            rotated = best_cycle[shift:] + best_cycle[:shift]
            improved_c, improved_s = local_search_2opt(rotated, max_iter=50000)
            if improved_s < best_score:
                best_cycle = improved_c
                best_score = improved_s
    
    # Final answer: return the cycle as a permutation (without the closing edge)
    # The cycle is [v0, v1, ..., v23, v0], so we return [v0, v1, ..., v23]
    if len(best_cycle) == n + 1:
        result = best_cycle[:-1]
    else:
        result = best_cycle
    
    # Verify it's a valid permutation
    assert len(result) == n
    assert set(result) == set(range(n))
    
    return result
# EVOLVE-BLOCK-END
