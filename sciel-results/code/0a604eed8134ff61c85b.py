import time
import sys
import numpy as np

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def find_hamilton_cycle_fast(adj_masks, n, deadline):
    """
    Fast Hamilton cycle finder using bitmask backtracking with MRV heuristic.
    adj_masks: list of integers, where adj_masks[v] is a bitmask of neighbors of v.
    """
    # Quick degree check
    for v in range(n):
        if bin(adj_masks[v]).count('1') < 2:
            return None
    
    # Fix vertex 0 as start
    path = [0]
    visited_mask = 1 << 0
    
    # Precompute bit counts for speed (or use bin().count)
    # For N=24, bin().count is fast enough, but we can cache if needed.
    
    def popcount(x):
        return bin(x).count('1')

    def is_connected(unvisited_mask):
        """Check if the subgraph induced by unvisited_mask is connected."""
        if unvisited_mask == 0:
            return True
        # Find first set bit
        first = (unvisited_mask & -unvisited_mask).bit_length() - 1
        seen = 1 << first
        frontier = seen
        remaining = unvisited_mask
        
        while frontier:
            # Expand frontier
            # For each node in frontier, add its neighbors that are in remaining
            # We can do this by iterating through bits of frontier
            new_frontier = 0
            f = frontier
            while f:
                lsb = f & -f
                idx = lsb.bit_length() - 1
                # Add neighbors of idx that are in remaining and not seen
                new_frontier |= adj_masks[idx] & remaining & ~seen
                f -= lsb
            
            if not new_frontier:
                break
            
            seen |= new_frontier
            remaining &= ~seen
            frontier = new_frontier & remaining
            
        return remaining == 0

    def backtrack(depth, last):
        if time.time() > deadline:
            return None
        
        if depth == n:
            # Check if last connects to 0
            if (adj_masks[last] >> 0) & 1:
                return path[:]
            return None
        
        # Get candidates
        candidates_mask = adj_masks[last] & ~visited_mask
        if candidates_mask == 0:
            return None
            
        # MRV: Sort candidates by the number of available neighbors they have in the unvisited set
        # Note: The "unvisited set" for the next step will be visited_mask | (1 << nxt)
        # So we check popcount(adj_masks[nxt] & ~(visited_mask | (1 << nxt)))
        
        cand_list = []
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            # Degree of idx in the graph excluding visited and idx itself
            deg = popcount(adj_masks[idx] & ~(visited_mask | lsb))
            # If deg == 0 and depth < n-1, it's a dead end (unless it's the last one and connects to 0)
            # We just use deg for sorting. Lower deg is more constrained.
            cand_list.append((deg, idx))
            cm -= lsb
        
        # Sort by degree (ascending)
        cand_list.sort()
        
        for deg, nxt in cand_list:
            # Pruning: If this is the last vertex to be placed (depth == n-1)
            if depth == n - 1:
                # nxt must connect to 0
                if not ((adj_masks[nxt] >> 0) & 1):
                    continue
                # Also, the only unvisited vertex must be nxt (which is true by construction of candidates_mask if popcount(candidates_mask)==1)
                # Wait, candidates_mask might have multiple bits.
                # If depth == n-1, we are picking the (n-1)th vertex. There is 1 vertex left.
                # So candidates_mask should have exactly 1 bit set.
                # Let's verify: initially visited=1. depth=1.
                # If depth == n-1, visited has n-1 bits. unvisited has 1 bit.
                # So candidates_mask is a subset of unvisited.
                # If popcount(candidates_mask) > 1, then we are not at the last step?
                # No, depth is the number of vertices in the path.
                # If depth == n-1, path has n-1 vertices. visited has n-1 bits.
                # unvisited has 1 bit.
                # So candidates_mask must be that 1 bit.
                pass
            
            # Connectivity Pruning
            # Check if the remaining unvisited vertices (after adding nxt) are connected
            # And if they are connected to the rest of the path?
            # Actually, it's sufficient to check if the unvisited subgraph is connected.
            # If it's not, we can't form a cycle.
            
            new_visited = visited_mask | (1 << nxt)
            remaining_mask = ((1 << n) - 1) & ~new_visited
            
            if remaining_mask != 0:
                # Check connectivity of remaining subgraph
                if not is_connected(remaining_mask):
                    continue
                
                # Check if remaining subgraph is connected to the path (specifically, to 'last' or '0' eventually)
                # Since we are building a path, the remaining vertices must be reachable from 'last' or '0'.
                # Actually, if the remaining subgraph is connected, and we enter it from 'last', we just need to ensure
                # that at least one vertex in the remaining subgraph is adjacent to 'last'.
                # But 'last' is the current end. We are moving to 'nxt'.
                # So the new end is 'nxt'.
                # The remaining vertices must be reachable from 'nxt'.
                # So we check if any vertex in remaining_mask is adjacent to nxt.
                if not (adj_masks[nxt] & remaining_mask):
                    continue
            else:
                # No remaining vertices, just check connection to 0
                if not ((adj_masks[nxt] >> 0) & 1):
                    continue

            path.append(nxt)
            result = backtrack(depth + 1, nxt)
            if result is not None:
                return result
            path.pop()
        
        return None

    return backtrack(1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = N
    total_edges = n * (n - 1) // 2
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Binary search for minimum threshold T
    lo, hi = 24, total_edges
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj_masks = [0] * n
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        
        # Check for Hamilton cycle
        cycle = find_hamilton_cycle_fast(adj_masks, n, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: use full graph
    adj_masks_full = [0] * n
    for (u, v) in EDGE_ORDER:
        adj_masks_full[u] |= (1 << v)
        adj_masks_full[v] |= (1 << u)
    
    cycle = find_hamilton_cycle_fast(adj_masks_full, n, deadline)
    if cycle is not None:
        return cycle
    
    # Final fallback: identity permutation
    return list(range(n))
