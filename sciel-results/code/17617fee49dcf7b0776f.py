import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
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
    
    def has_hamilton_cycle(k, timeout=None):
        """Check if subgraph with edges 1..k has a Hamilton cycle.
        Returns the cycle as a list of vertices if found, else None."""
        if timeout is not None:
            if time.time() > timeout:
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
        
        # Use bitmask backtracking
        # Fix vertex 0 as start to break symmetry
        # State: (current_vertex, visited_mask, path)
        # Use iterative DFS with pruning
        
        full_mask = (1 << n) - 1
        
        # Precompute: for each vertex, which neighbors have position <= k
        # and sort by position for greedy ordering
        
        # Try different starting vertices and use backtracking
        # Use a heuristic: try vertices with fewer options first (MRV)
        
        best_cycle = None
        best_max_pos = k + 1
        
        # We want to minimize the max position, so let's do a search
        # that tries to find cycles with smaller max positions
        
        # Actually, let's just check if any Hamilton cycle exists in the subgraph
        # and return it if found
        
        def backtrack(v, mask, path):
            if time.time() > deadline:
                return None
            
            if mask == full_mask:
                # Check if we can return to start (vertex 0)
                if 0 in adj[v]:
                    return path + [0]
                return None
            
            # Try neighbors
            # Sort neighbors by degree (fewest first) for pruning
            neighbors = list(adj[v])
            neighbors.sort(key=lambda x: len(adj[x]))
            
            for nb in neighbors:
                if not (mask & (1 << nb)):
                    # Pruning: check if nb can still reach all unvisited
                    # Simple connectivity check is expensive, skip for now
                    new_mask = mask | (1 << nb)
                    result = backtrack(nb, new_mask, path + [nb])
                    if result is not None:
                        return result
            
            return None
        
        # Try from vertex 0
        result = backtrack(0, 1, [0])
        if result is not None:
            return result
        
        return None
    
    # Binary search for the minimum k where a Hamilton cycle exists
    # But we also want to minimize the actual max edge position in the found cycle
    
    # Strategy: 
    # 1. Find the minimum k where a Hamilton cycle exists (binary search)
    # 2. Then search for a cycle that minimizes the max edge position
    
    # Actually, let's think differently. The score of a cycle is max(position of each edge).
    # We want to minimize this. So we can:
    # - For each k from low to high, check if there's a Hamilton cycle using only edges with pos <= k
    # - The first k where such a cycle exists gives us the optimal answer
    
    # But binary search is better. Let's find the min k.
    
    lo, hi = 24, total_edges  # need at least 24 edges for a Hamilton cycle
    best_k = hi
    best_cycle = None
    
    # First, let's try to find the minimum k with binary search
    # But we need to be careful: the property "has Hamilton cycle" is monotone in k
    
    # Let's do a linear scan from a reasonable lower bound
    # The minimum possible is 24 (a 24-cycle), but in practice it'll be higher
    
    # Let's try values around where we expect the answer
    # With 24 vertices and 276 edges, a random Hamilton cycle has max position ~ 276*24/25 ≈ 265
    # But we're searching for the best one, so it should be much lower
    
    # Let's try a range of k values
    # Start with a guess and refine
    
    # Actually, let's just do a binary search properly
    def check_k(k):
        """Returns a Hamilton cycle using only edges with pos <= k, or None."""
        if time.time() > deadline:
            return None
        return has_hamilton_cycle(k)
    
    # Binary search
    lo, hi = 24, total_edges
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
    
    # Now best_cycle is a Hamilton cycle using edges with pos <= best_k
    # But we can try to find a better one (lower max position) by searching more carefully
    
    # Let's compute the actual max position of the found cycle
    def cycle_max_pos(cycle):
        max_p = 0
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            p = get_pos(u, v)
            if p > max_p:
                max_p = p
        return max_p
    
    current_best = cycle_max_pos(best_cycle)
    
    # Try to improve: search for cycles with lower max position
    # We can try k values below best_k to see if we missed something
    # (binary search should have found the minimum, but let's verify)
    
    # Also try local search: swap vertices to reduce max position
    
    # Local search / hill climbing
    def try_improve(cycle, max_iterations=10000):
        nonlocal current_best
        best_c = list(cycle)
        best_score = cycle_max_pos(best_c)
        n_iter = 0
        
        while n_iter < max_iterations and time.time() < deadline:
            n_iter += 1
            improved = False
            
            # Try all pairs of consecutive edges and see if swapping helps
            # Or try 2-opt style moves
            
            # Simple: try swapping two non-adjacent vertices
            for i in range(len(best_c) - 1):
                for j in range(i + 2, len(best_c)):
                    if j == len(best_c) - 1 and i == 0:
                        continue  # would be same cycle
                    # Try reversing the segment
                    new_c = best_c[:i+1] + best_c[i+1:j+1][::-1] + best_c[j+1:]
                    # Check if all edges still have pos <= best_k (or lower)
                    new_score = cycle_max_pos(new_c)
                    if new_score < best_score:
                        best_c = new_c
                        best_score = new_score
                        improved = True
                        if best_score < current_best:
                            current_best = best_score
                            best_cycle = list(best_c)
            
            if not improved:
                break
        
        return best_c, best_score
    
    # Try to improve
    if time.time() < deadline - 5:
        best_cycle, current_best = try_improve(best_cycle)
    
    # Final answer: return the cycle as a permutation (without the closing edge)
    # The cycle is [v0, v1, ..., v23, v0], so we return [v0, v1, ..., v23]
    result = best_cycle[:-1]  # remove the last element which is the start vertex again
    
    # Verify it's a valid permutation
    assert len(result) == n
    assert set(result) == set(range(n))
    
    return result
# EVOLVE-BLOCK-END
