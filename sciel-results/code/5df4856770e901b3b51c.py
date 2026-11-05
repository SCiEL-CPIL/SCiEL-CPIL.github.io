"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_index = {}
for idx, (u, v) in enumerate(P['edge_order'], 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = idx

def construct():
    start_time = time.time()
    deadline = start_time + 82
    
    max_idx = max(edge_index.values())
    
    def exists_hamiltonian_cycle(K):
        # Build adjacency as bitmasks for speed
        adj_mask = [0] * N
        for (u, v), idx in edge_index.items():
            if idx <= K:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Pre-check: Every vertex must have degree >= 2
        for i in range(N):
            if bin(adj_mask[i]).count('1') < 2:
                return None
        
        start = 0
        full_mask = (1 << N) - 1
        
        # Warnsdorff: sort neighbors by fewest onward options
        def get_sorted_neighbors(u, mask):
            neighbors = []
            m = adj_mask[u] & ~mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                # Count onward options for v
                onward = bin(adj_mask[v] & ~mask & ~(1 << u)).count('1')
                neighbors.append((onward, v))
            neighbors.sort()
            return [v for _, v in neighbors]
        
        # Strong pruning: check connectivity and degree constraints
        def can_complete(u, mask):
            if mask == full_mask:
                return (adj_mask[u] & (1 << start)) != 0
            
            unused_mask = full_mask & ~mask
            
            # Check: every unused vertex must have degree >= 2 in subgraph
            # induced by unused vertices + {u, start}
            relevant_mask = unused_mask | (1 << u) | (1 << start)
            
            for v in range(N):
                if not (unused_mask & (1 << v)):
                    continue
                # Count neighbors of v in relevant_mask
                deg = bin(adj_mask[v] & relevant_mask).count('1')
                if deg < 2:
                    return False
            
            # Check connectivity: unused vertices + {u, start} must be connected
            # BFS from u over relevant_mask
            visited = 1 << u
            frontier = 1 << u
            while frontier:
                new_frontier = 0
                f = frontier
                while f:
                    x = (f & -f).bit_length() - 1
                    f &= f - 1
                    # Expand to neighbors in relevant_mask not yet visited
                    nbrs = adj_mask[x] & relevant_mask & ~visited
                    new_frontier |= nbrs
                visited |= new_frontier
                frontier = new_frontier
            
            if visited != relevant_mask:
                return False
            
            return True
        
        # Iterative DFS with explicit stack to avoid recursion depth issues
        # and to allow better time management
        stack = [(start, 1 << start, [start])]
        best = None
        
        while stack:
            if time.time() > deadline:
                break
            
            u, mask, path = stack.pop()
            
            if not can_complete(u, mask):
                continue
            
            if mask == full_mask:
                if (adj_mask[u] & (1 << start)) != 0:
                    best = path[:]
                    break
            
            # Get neighbors sorted by Warnsdorff
            neighbors = get_sorted_neighbors(u, mask)
            # Push in reverse so first neighbor is popped first
            for v in reversed(neighbors):
                new_mask = mask | (1 << v)
                stack.append((v, new_mask, path + [v]))
        
        return best
    
    low = 0
    high = max_idx
    best_cycle = None
    
    while low <= high:
        mid = (low + high) // 2
        cycle = exists_hamiltonian_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        return list(range(N))
        
    return best_cycle

# EVOLVE-BLOCK-END
