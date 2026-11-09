import time
import random

def construct():
    # Graph definition from seed
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    cycles = []
    current_adj = {i: set(adj[i]) for i in range(n)}
    
    def find_hamiltonian_cycle(adj_map, start_node=0):
        """
        Finds a Hamiltonian cycle in the graph defined by adj_map.
        Returns a list of vertices representing the cycle, or None if not found.
        """
        n = len(adj_map)
        path = [start_node]
        visited = {start_node}
        
        # Pre-check: all vertices must have degree >= 2
        for u in range(n):
            if len(adj_map[u]) < 2:
                return None
        
        def backtrack():
            if time.time() > deadline:
                return False
                
            current = path[-1]
            
            if len(path) == n:
                # Check if we can close the cycle
                if start_node in adj_map[current]:
                    # Verify the cycle is simple and valid
                    # Since we only add unvisited nodes, it is simple.
                    return True
                return False
            
            # Get neighbors
            # Heuristic: sort neighbors by degree in the remaining graph
            # to find bottlenecks early
            neighbors = list(adj_map[current])
            # Filter out visited nodes (except start_node if we are closing, but we handle closing separately)
            # Actually, we just look at unvisited neighbors
            unvisited_neighbors = [v for v in neighbors if v not in visited]
            
            if not unvisited_neighbors:
                return False
            
            # Sort by degree in the remaining graph (excluding path edges conceptually, 
            # but adj_map is static for this search instance, so we use static degree 
            # or dynamic degree if we were modifying it. Here adj_map is the remaining graph 
            # after removing previous cycles, but within this search, edges are "used" 
            # by the path. We don't remove them from adj_map yet.
            # So we should estimate the "remaining degree" for pruning.
            
            # Simple pruning: 
            # If any unvisited vertex has degree 1 in the subgraph of unvisited vertices + current, 
            # it must connect to its only neighbor.
            
            # For now, just try all
            # Randomize order to avoid getting stuck in local minima? 
            # Or deterministic? Deterministic is faster if it works.
            # Let's try deterministic first, then random if it fails.
            
            for next_node in unvisited_neighbors:
                # Pruning check:
                # If we go to next_node, does any vertex get stuck?
                # Specifically, if next_node is the last unvisited node, it must connect to start.
                if len(visited) + 1 == n:
                    if start_node not in adj_map[next_node]:
                        continue
                
                # Check if any other unvisited vertex now has degree < 2 
                # considering the edge (current, next_node) is "used".
                # This is expensive to check fully. 
                # A cheaper check: 
                # If next_node has only 1 unvisited neighbor (other than current), 
                # and that neighbor is not start (if it's the last step), etc.
                
                # Let's use a simple heuristic: 
                # If next_node has degree 1 in the remaining unvisited set + current, 
                # we must take that edge.
                
                # Just recurse
                visited.add(next_node)
                path.append(next_node)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(next_node)
                
            return False
        
        if backtrack():
            return path
        return None

    # Try to find 3 cycles
    # We might need to restart if we get stuck with < 3
    max_cycles = 3
    
    # Since the graph is symmetric, a specific order might work.
    # Let's just run the search.
    
    for i in range(max_cycles):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(current_adj)
        if cycle is None:
            break
        
        cycles.append(cycle)
        
        # Remove edges of this cycle from current_adj
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
        # If we removed edges, check if graph is still 2-connected enough
        # If any vertex has degree < 2, we can't form another cycle
        valid = True
        for u in range(n):
            if len(current_adj[u]) < 2:
                valid = False
                break
        if not valid:
            break
            
    # If we found fewer than 3, try randomizing the start node or edge order?
    # The prompt says "Maximize number of cycles".
    # If we only found 2, maybe a different first cycle allows 3?
    # Given the time limit and the fact that 3 is the max, and previous runs found 3,
    # a simple greedy approach might suffice. 
    # However, to be robust, we can try a few random restarts if we don't get 3.
    
    if len(cycles) < 3 and time.time() < deadline - 5.0:
        # Try a few random restarts
        for attempt in range(10):
            if time.time() > deadline - 5.0:
                break
            temp_adj = {i: set(adj[i]) for i in range(n)}
            temp_cycles = []
            success = True
            for _ in range(3):
                # Random start node
                start_node = random.randint(0, n-1)
                cycle = find_hamiltonian_cycle(temp_adj, start_node)
                if cycle is None:
                    success = False
                    break
                temp_cycles.append(cycle)
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    temp_adj[u].remove(v)
                    temp_adj[v].remove(u)
            
            if success and len(temp_cycles) > len(cycles):
                cycles = temp_cycles
                if len(cycles) == 3:
                    break

    # Return the best found
    # Ensure types are correct: list of lists of ints
    return [list(c) for c in cycles]
