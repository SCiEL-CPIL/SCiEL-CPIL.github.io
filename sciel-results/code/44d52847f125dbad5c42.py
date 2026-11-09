"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    
    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i

    def find_hamiltonian_cycle(threshold):
        """
        Find a Hamiltonian cycle using only edges with position < threshold.
        Returns the cycle (list of vertices) or None if not found.
        Uses backtracking with pruning.
        """
        # Build adjacency list for edges with pos < threshold
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort adjacency lists by edge position to prioritize early edges (heuristic)
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
            
        # Use vertex 0 as the start to fix symmetry
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack():
            if time.time() - start_time > 85:
                return False
            if len(path) == n:
                # Check if we can close the cycle
                last = path[-1]
                if edge_pos.get((last, start), float('inf')) < threshold:
                    return True
                return False
            
            last = path[-1]
            # Heuristic: try neighbors with lowest edge positions first
            for next_v in adj[last]:
                if not visited[next_v]:
                    # Pruning: if next_v is the start and we haven't visited all nodes, skip
                    if next_v == start and len(path) < n - 1:
                        continue
                    path.append(next_v)
                    visited[next_v] = True
                    if backtrack():
                        return True
                    path.pop()
                    visited[next_v] = False
            return False
            
        if backtrack():
            return path
        return None

    # Binary search for the minimum threshold
    lo = n # Minimum possible is n (cycle of n edges)
    hi = num_edges
    best_cycle = None
    best_threshold = num_edges
    
    # First, ensure we have a valid cycle with the full set
    # This also serves as a fallback
    full_cycle = find_hamiltonian_cycle(num_edges)
    if full_cycle:
        best_cycle = full_cycle
        best_threshold = num_edges
    else:
        # Should not happen for complete graph, but safe fallback
        return list(range(n))
        
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        if time.time() - start_time > 80:
            break
        
        cycle = find_hamiltonian_cycle(mid)
        if cycle:
            best_cycle = cycle
            best_threshold = mid
            hi = mid
        else:
            lo = mid + 1
            
    # Refinement: Try to lower the threshold further if we have time
    # The binary search finds the *existence* threshold.
    # However, the score is the max edge index in the *specific* cycle found.
    # The cycle found at threshold T might use an edge with index T-1, or lower.
    # We want to minimize the max edge index of the cycle.
    
    # Let's try to optimize the specific cycle found.
    # We can try to find a cycle with a strictly lower max edge index.
    # The binary search gives us the smallest T such that *some* cycle exists in G_T.
    # That cycle's max edge index is <= T-1.
    # Is it possible to find a cycle with max edge index < T-1?
    # If yes, then a cycle would exist in G_{T-1}, which contradicts the minimality of T from binary search.
    # Wait, the binary search checks existence in G_mid (edges 0..mid-1).
    # If find_hamiltonian_cycle(mid) succeeds, it returns a cycle using only edges < mid.
    # So the max edge index is at most mid-1.
    # The score is max_index + 1. So score <= mid.
    # The binary search minimizes mid.
    # Thus, the score of the returned cycle is at most the optimal possible score.
    # Actually, if a cycle exists with max index K, then it exists in G_{K+1}.
    # Binary search finds min T such that cycle exists in G_T.
    # Then score <= T.
    # Could score be < T? If score is S, cycle exists in G_S.
    # Then binary search would have found T <= S.
    # So the minimum score is exactly the minimum T found.
    
    # The cycle returned by find_hamiltonian_cycle(best_threshold) uses edges < best_threshold.
    # Its max edge index is <= best_threshold - 1.
    # Score is max_index + 1 <= best_threshold.
    
    # We return the cycle found.
    if best_cycle:
        return best_cycle
    else:
        return list(range(n))

# EVOLVE-BLOCK-END
