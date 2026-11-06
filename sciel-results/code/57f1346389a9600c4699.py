import random
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']

def find_hamiltonian_cycle(K):
    """
    Finds a Hamiltonian cycle in the graph formed by the first K edges.
    Returns the cycle as a list of vertices, or None if not found.
    """
    # Build adjacency list for the first K edges
    adj = {i: set() for i in range(N)}
    for i in range(K):
        u, v = EDGES[i]
        adj[u].add(v)
        adj[v].add(u)
    
    # Heuristic: Start from vertex 0
    # Path is a list of vertices
    # visited is a set of vertices in the path
    # We need to ensure the path can be closed to form a cycle
    
    def dfs(path, visited, last_vertex):
        if len(path) == N:
            # Check if last_vertex is connected to path[0]
            if path[0] in adj[last_vertex]:
                return path[:]
            return None
        
        # Pruning: Check if any unvisited vertex has no neighbors in the remaining graph
        # that could allow it to be included. This is complex.
        # Simpler pruning: If any unvisited vertex has degree 0 in the subgraph induced by unvisited + path[0] (for closing)?
        
        # Let's use a simple degree check:
        # If an unvisited vertex has no neighbors in the unvisited set and is not connected to path[0] (if it's the last one) 
        # or not connected to the current end, it might be stuck.
        
        # Better heuristic: Warnsdorff's rule - visit the neighbor with the fewest onward neighbors.
        candidates = []
        for neighbor in adj[last_vertex]:
            if neighbor not in visited:
                # Count onward neighbors for this neighbor
                onward = 0
                for n2 in adj[neighbor]:
                    if n2 not in visited:
                        onward += 1
                # Also, if neighbor is the last one to be added, it must connect to path[0]
                # We don't know if it's the last one yet.
                candidates.append((onward, neighbor))
        
        if not candidates:
            return None
            
        # Sort by onward degree
        candidates.sort()
        
        for _, next_v in candidates:
            # Special case: if adding next_v completes the cycle (len(path)+1 == N)
            # then next_v must connect to path[0]
            if len(path) + 1 == N:
                if path[0] not in adj[next_v]:
                    continue
            
            # Also, if we pick next_v, we must ensure we don't isolate any other unvisited vertex.
            # This is hard to check cheaply.
            
            new_path = path + [next_v]
            new_visited = visited | {next_v}
            
            # Additional Pruning: 
            # If there is an unvisited vertex that has no neighbors in (unvisited U {path[0]}) 
            # and is not the next_v (which is now visited), it might be impossible.
            # Actually, if an unvisited vertex has 0 neighbors in the unvisited set, 
            # it MUST connect to path[0] (if it's the last) or the current end? No.
            # It must connect to the path.
            
            result = dfs(new_path, new_visited, next_v)
            if result:
                return result
        
        return None

    # Start DFS
    # To improve performance, we can try starting from different vertices if 0 fails?
    # Or just use 0.
    result = dfs([0], {0}, 0)
    if result:
        return result
    return None

def construct():
    start_time = time.time()
    max_time = 55.0
    
    # Minimum edges for a cycle is N
    for K in range(N, len(EDGES) + 1):
        if time.time() - start_time > max_time:
            break
            
        cycle = find_hamiltonian_cycle(K)
        if cycle:
            return cycle
            
    # Fallback: Random cycle (should not happen if K goes high enough)
    return list(range(N))
# EVOLVE-BLOCK-END
