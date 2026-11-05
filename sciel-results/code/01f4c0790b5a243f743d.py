import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_index = {}
for idx, (u, v) in enumerate(P['edge_order'], 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = idx

# Precompute edge list sorted by index for quick lookup
edges_by_index = []
for idx in range(1, 277):
    for (u, v), i in edge_index.items():
        if i == idx:
            edges_by_index.append((u, v))
            break

def popcount(x):
    """Count bits in x using Python's int.bit_count() (Python 3.8+)"""
    return x.bit_count()

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    max_idx = max(edge_index.values())
    
    def exists_hamiltonian_cycle(K, start_vertex=0):
        # Build adjacency masks for edges <= K
        adj_mask = [0] * N
        neighbors = [[] for _ in range(N)]
        
        for (u, v), idx in edge_index.items():
            if idx <= K:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                neighbors[u].append(v)
                neighbors[v].append(u)
                
        # Pre-check: Degree >= 2 for all
        for i in range(N):
            if popcount(adj_mask[i]) < 2:
                return None

        full_mask = (1 << N) - 1
        start = start_vertex
        
        def can_complete(u, mask):
            if mask == full_mask:
                return (adj_mask[u] & (1 << start)) != 0
            
            unused_mask = full_mask & ~mask
            relevant_mask = unused_mask | (1 << u) | (1 << start)
            
            # Degree check for unused vertices
            for v in range(N):
                if not (unused_mask & (1 << v)):
                    continue
                deg = popcount(adj_mask[v] & relevant_mask)
                if deg < 2:
                    return False
            
            # Connectivity check using BFS
            visited = 1 << u
            frontier = 1 << u
            while frontier:
                new_frontier = 0
                f = frontier
                while f:
                    x = (f & -f).bit_length() - 1
                    f -= f & -f
                    nbrs = adj_mask[x] & relevant_mask & ~visited
                    new_frontier |= nbrs
                visited |= new_frontier
                frontier = new_frontier
            
            if visited != relevant_mask:
                return False
            return True

        def get_sorted_neighbors(u, mask):
            """Get unvisited neighbors of u, sorted by onward degree (Warnsdorff)"""
            m = adj_mask[u] & ~mask
            res = []
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m -= lsb
                # Onward degree: neighbors of v not in mask, excluding u (u is in mask)
                onward = popcount(adj_mask[v] & ~mask)
                res.append((onward, v))
            res.sort()
            return [v for _, v in res]

        # Iterative DFS with backtracking
        stack = [(start, 1 << start, [start])]
        
        while stack:
            if time.time() > deadline:
                return None
                
            u, mask, path = stack.pop()
            
            if not can_complete(u, mask):
                continue
            
            if mask == full_mask:
                if (adj_mask[u] & (1 << start)) != 0:
                    return path[:]
            
            nbrs = get_sorted_neighbors(u, mask)
            
            # Push in reverse so that the best (lowest onward degree) is tried first
            for v in reversed(nbrs):
                new_mask = mask | (1 << v)
                stack.append((v, new_mask, path + [v]))
                
        return None

    def two_opt_improve(cycle, K):
        """Try to improve the cycle using 2-opt moves, keeping max edge index <= K"""
        if cycle is None:
            return None
        
        # Build a set of edges in the cycle for quick lookup
        n = len(cycle)
        cycle_edges = set()
        for i in range(n):
            u, v = cycle[i], cycle[(i+1) % n]
            a, b = (u, v) if u < v else (v, u)
            cycle_edges.add((a, b))
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try all 2-opt moves
            for i in range(n):
                for j in range(i+2, n):
                    if i == 0 and j == n-1:
                        continue  # This would be the closing edge
                    
                    # Current edges: (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                    u1, v1 = cycle[i], cycle[(i+1) % n]
                    u2, v2 = cycle[j], cycle[(j+1) % n]
                    
                    # New edges would be: (u1, u2) and (v1, v2)
                    # Check if these edges exist with index <= K
                    a1, b1 = (u1, u2) if u1 < u2 else (u2, u1)
                    a2, b2 = (v1, v2) if v1 < v2 else (v2, v1)
                    
                    if (a1, b1) in edge_index and edge_index[(a1, b1)] <= K:
                        if (a2, b2) in edge_index and edge_index[(a2, b2)] <= K:
                            # Check that we're not creating a 2-cycle or self-loop
                            if u1 != u2 and v1 != v2:
                                # Apply the move
                                segment = cycle[i+1:j+1]
                                segment.reverse()
                                new_cycle = cycle[:i+1] + segment + cycle[j+1:]
                                
                                # Verify it's still a valid Hamilton cycle
                                if len(new_cycle) == n and len(set(new_cycle)) == n:
                                    cycle = new_cycle
                                    improved = True
                                    break
                if improved:
                    break
        
        return cycle

    # Strategy: Try binary search, but also try multiple starting vertices
    low = 0
    high = max_idx
    best_cycle = None
    best_K = max_idx + 1
    
    # First, do a quick scan to find a reasonable upper bound
    # Try K values from high to low in steps, then binary search
    # Actually, let's just do binary search but with multiple starts
    
    # Try several starting vertices to increase chances of finding a solution
    start_vertices = [0, 1, 2, 3, 4, 5]
    
    while low <= high:
        if time.time() > deadline - 10:
            break
        mid = (low + high) // 2
        found = False
        for sv in start_vertices:
            cycle = exists_hamiltonian_cycle(mid, sv)
            if cycle is not None:
                # Try to improve with 2-opt
                cycle = two_opt_improve(cycle, mid)
                if cycle is not None:
                    best_cycle = cycle
                    best_K = mid
                    found = True
                    break
        if found:
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        # Fallback: try larger K values directly
        for K in range(max_idx, max_idx - 50, -1):
            if time.time() > deadline - 5:
                break
            for sv in start_vertices:
                cycle = exists_hamiltonian_cycle(K, sv)
                if cycle is not None:
                    best_cycle = cycle
                    best_K = K
                    break
            if best_cycle is not None:
                break
    
    if best_cycle is None:
        # Last resort: identity permutation
        return list(range(N))
        
    return best_cycle
