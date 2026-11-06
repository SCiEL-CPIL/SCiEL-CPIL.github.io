# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _find_hamiltonian_cycle_optimized(adj, n):
    """
    Finds a Hamiltonian cycle using a DFS with Warnsdorff's rule and pruning.
    Returns the cycle as a list of vertices or None.
    """
    if n < 3:
        return None
    
    # Pre-compute static degrees for sorting? No, dynamic degrees are better for Warnsdorff.
    # But we can use a list of neighbors for faster iteration than sets.
    # Convert adj sets to lists for faster access if needed, but sets are fine for membership.
    
    start = 0
    visited_mask = 1 << start
    path = [start]
    
    # To speed up neighbor iteration, we can keep the adjacency lists.
    # adj is a list of sets.
    
    def dfs(current, visited_mask, path_len):
        if path_len == n:
            # Check if we can close the cycle
            if start in adj[current]:
                return True
            return False
        
        # Get unvisited neighbors
        # We want to sort them by their "future degree" (number of unvisited neighbors)
        # This is Warnsdorff's rule.
        
        # Collect candidates
        candidates = []
        for nb in adj[current]:
            if not (visited_mask & (1 << nb)):
                # Calculate number of unvisited neighbors for nb
                # This is O(deg(nb))
                count = 0
                for nnb in adj[nb]:
                    if not (visited_mask & (1 << nnb)):
                        count += 1
                candidates.append((count, nb))
        
        if not candidates:
            return False
            
        # Sort by count (ascending) - visit most constrained first
        candidates.sort(key=lambda x: x[0])
        
        for count, nb in candidates:
            new_visited = visited_mask | (1 << nb)
            
            # Pruning: Check if any unvisited vertex becomes isolated
            # A vertex u is isolated if all its neighbors are visited or u itself.
            # Since we just added nb to visited, we only need to check vertices that might be affected?
            # Actually, any unvisited vertex u such that all its neighbors are in new_visited is stuck.
            # However, u can still be visited later if it has a neighbor in the unvisited set.
            # If u has NO neighbors in the unvisited set (excluding itself), it's stuck.
            # Note: u's neighbors in the unvisited set are those v where v is unvisited and v in adj[u].
            # If count_unvisited_neighbors(u) == 0, it's stuck.
            # But u is unvisited, so it's not in new_visited.
            # So we check: for all u in unvisited, is there any v in unvisited such that v in adj[u]?
            # If not, u is isolated.
            
            # This check is O(N * AvgDeg). For N=24, it's cheap.
            # We can optimize by only checking if the graph of unvisited vertices is connected?
            # No, just checking for isolated vertices is a good first prune.
            
            # Let's do the isolation check.
            # Iterate over all vertices, if unvisited, check if it has any unvisited neighbor.
            # We can break early if found.
            
            # Optimization: We only need to check vertices that are neighbors of the newly visited node?
            # No, adding nb to visited removes nb from the "unvisited" pool for others.
            # So any vertex u that was connected ONLY to nb (and other visited nodes) is now isolated.
            # So we only need to check neighbors of nb?
            # If u is a neighbor of nb, and u is unvisited, and all other neighbors of u are visited, then u is isolated.
            # Yes! This is much faster.
            
            for u in adj[nb]:
                if u == current: # current is visited, so not in unvisited
                    continue
                if not (new_visited & (1 << u)):
                    # u is unvisited. Check if u has any other unvisited neighbors.
                    # We know nb is now visited.
                    # So we check if u has any neighbor v such that v is unvisited.
                    has_unvisited_neighbor = False
                    for v in adj[u]:
                        if not (new_visited & (1 << v)):
                            has_unvisited_neighbor = True
                            break
                    if not has_unvisited_neighbor:
                        # u is isolated
                        return False
            
            path.append(nb)
            if dfs(nb, new_visited, path_len + 1):
                return True
            path.pop()
            
        return False

    # Use a non-recursive approach or increase recursion limit?
    # N=24, recursion depth is fine.
    if dfs(start, visited_mask, 1):
        return path[:]
    return None

def construct():
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    
    # Map edge to position
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        key = (min(u, v), max(u, v))
        edge_pos[key] = i + 1
        
    # Binary search on the maximum edge index K
    # We want the smallest K such that a Hamiltonian cycle exists using edges with pos <= K.
    
    # Lower bound: At least N edges are needed for a cycle.
    # Upper bound: Total edges.
    
    lo = n
    hi = num_edges
    best_cycle = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        # Build graph with edges having position <= mid
        adj = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edge_order):
            if i + 1 <= mid:
                adj[u].add(v)
                adj[v].add(u)
            else:
                break # Since edge_order is the sequence, positions are 1..M.
                # Wait, edge_order is a list of edges. The position is the index in this list.
                # So we just iterate and add if index < mid.
                # But we broke? No, we should just iterate all or break if we know they are sorted by index?
                # The list `edge_order` IS the order. So edge at index `i` has position `i+1`.
                # So we can just take the first `mid` edges.
        
        # Re-implementing graph build for clarity/speed
        adj = [set() for _ in range(n)]
        for i in range(mid):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
            
        # Check for Hamiltonian cycle
        cycle = _find_hamiltonian_cycle_optimized(adj, n)
        if cycle:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle:
        return best_cycle
    else:
        # Fallback: The full graph must have a cycle.
        # This case should not happen for a complete graph, but just in case.
        adj = [set() for _ in range(n)]
        for u, v in edge_order:
            adj[u].add(v)
            adj[v].add(u)
        return _find_hamiltonian_cycle_optimized(adj, n) or list(range(n))
# EVOLVE-BLOCK-END
