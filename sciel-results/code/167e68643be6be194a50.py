import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']
TOTAL_EDGES = N * (N - 1) // 2

# Precompute edge index lookup
edge_index = {}
for i, e in enumerate(EDGES):
    u, v = e
    if u > v:
        u, v = v, u
    edge_index[(u, v)] = i + 1

def build_adjacency(threshold):
    """Build adjacency bitsets for edges with index <= threshold."""
    adj = [0] * N
    for i in range(threshold):
        u, v = EDGES[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def find_hamilton_cycle_fast(adj, deadline):
    """
    Find Hamiltonian cycle in graph with adjacency bitsets using
    Warnsdorff's heuristic with forced moves and dead-end pruning.
    Returns path list or None.
    """
    start = 0
    full_mask = (1 << N) - 1
    
    # Precompute popcount for 24-bit integers using a lookup or builtin
    def popcount(x):
        return bin(x).count('1')
    
    # Iterative DFS with explicit stack to avoid recursion limit issues
    # State: (current, visited, path_tuple)
    # We'll use a stack of (current, visited, path_list, next_neighbor_index)
    
    # Try multiple starting vertices if the first fails quickly
    best_path = None
    
    for start_vertex in range(N):
        if time.time() > deadline:
            break
            
        full_mask = (1 << N) - 1
        start_mask = 1 << start_vertex
        
        # Stack-based DFS
        # Each entry: (node, visited_mask, path, neighbor_choices, choice_idx)
        stack = []
        
        # Initial: try to start from start_vertex
        # Neighbors of start_vertex that can lead to a Hamiltonian cycle
        init_neighbors = adj[start_vertex] & ~start_mask
        if init_neighbors == 0:
            continue
            
        # Warnsdorff: sort by number of unvisited neighbors
        init_list = []
        temp = init_neighbors
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # For the first move, count neighbors excluding start and v
            deg = popcount(adj[v] & ~(start_mask | (1 << v)))
            init_list.append((deg, v))
            temp ^= lsb
        init_list.sort()
        
        for deg, v in init_list:
            if time.time() > deadline:
                break
            new_visited = start_mask | (1 << v)
            # Check v isn't a dead end (unless it's the last node)
            if new_visited != full_mask:
                if (adj[v] & ~new_visited) == 0:
                    continue
            path = [start_vertex, v]
            stack.append((v, new_visited, path, None, 0))
            
            # Process stack
            while stack:
                if time.time() > deadline:
                    return best_path
                    
                node, visited, path, choices, cidx = stack[-1]
                
                if visited == full_mask:
                    # Check if node connects back to start
                    if adj[node] & start_mask:
                        return path
                    stack.pop()
                    continue
                
                # Get or compute neighbor choices
                if choices is None:
                    neighbors_mask = adj[node] & ~visited
                    if neighbors_mask == 0:
                        stack.pop()
                        continue
                    # Sort by Warnsdorff's rule
                    choices_list = []
                    temp = neighbors_mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        # Count unvisited neighbors of v (excluding current visited)
                        deg = popcount(adj[v] & ~visited)
                        choices_list.append((deg, v))
                        temp ^= lsb
                    choices_list.sort()
                    stack[-1] = (node, visited, path, choices_list, 0)
                    choices = choices_list
                    cidx = 0
                
                # Try next choice
                if cidx < len(choices):
                    deg, v = choices[cidx]
                    # Update stack with next index
                    stack[-1] = (node, visited, path, choices, cidx + 1)
                    
                    new_visited = visited | (1 << v)
                    # Dead-end check
                    if new_visited != full_mask:
                        if (adj[v] & ~new_visited) == 0:
                            continue
                    
                    new_path = path + [v]
                    stack.append((v, new_visited, new_path, None, 0))
                else:
                    # No more choices, backtrack
                    stack.pop()
                    
        # If we exhausted all starts for this start_vertex, continue to next
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Binary search for the minimum threshold T
    lo = N  # Minimum edges needed is N (for a cycle)
    hi = TOTAL_EDGES
    best_path = None
    best_T = hi
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        
        adj = build_adjacency(mid)
        path = find_hamilton_cycle_fast(adj, deadline)
        
        if path is not None:
            best_path = path
            best_T = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_path is None:
        # Fallback: return the trivial cycle
        return list(range(N))
        
    return best_path
# EVOLVE-BLOCK-END
