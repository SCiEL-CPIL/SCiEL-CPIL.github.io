"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = P['n']
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # We know the max number of cycles is 3.
    # We will try to find 3 edge-disjoint Hamiltonian cycles.
    
    # State: available edges
    # Represent edges as a set of frozensets or sorted tuples
    avail_edges = set()
    for u, v in P['edges']:
        avail_edges.add((min(u, v), max(u, v)))
        
    found_cycles = []
    
    def find_hamiltonian_cycle():
        """
        Finds a Hamiltonian cycle in the graph defined by avail_edges.
        Returns the cycle as a list of vertices, or None if not found.
        """
        if time.time() > deadline:
            return None
            
        # Use backtracking DFS
        # Start at vertex 0
        path = [0]
        visited = {0}
        
        def dfs(current):
            if time.time() > deadline:
                return False
            if len(path) == n:
                # Check if we can close the cycle
                if (0, current) in avail_edges or (current, 0) in avail_edges:
                    # Verify edge exists in avail_edges
                    e = (min(0, current), max(0, current))
                    if e in avail_edges:
                        return True
                return False
            
            # Heuristic: sort neighbors by degree in remaining graph?
            # For simplicity, just iterate.
            # To speed up, we can try neighbors that are more constrained first.
            neighbors = adj[current]
            # Filter neighbors that are not visited and have an available edge
            candidates = []
            for nb in neighbors:
                if nb not in visited:
                    e = (min(current, nb), max(current, nb))
                    if e in avail_edges:
                        candidates.append(nb)
            
            # Sort candidates by degree in the remaining available graph
            # This is a simple heuristic (Warnsdorff's rule variant)
            def degree_in_avail(v):
                deg = 0
                for nb in adj[v]:
                    e = (min(v, nb), max(v, nb))
                    if e in avail_edges:
                        deg += 1
                return deg
                
            candidates.sort(key=lambda v: degree_in_avail(v))
            
            for nb in candidates:
                # Add edge
                e = (min(current, nb), max(current, nb))
                avail_edges.remove(e)
                path.append(nb)
                visited.add(nb)
                
                if dfs(nb):
                    return True
                    
                # Backtrack
                path.pop()
                visited.remove(nb)
                avail_edges.add(e)
                
            return False
            
        if dfs(0):
            return list(path)
        return None

    # Try to find 3 cycles
    for _ in range(3):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle()
        if cycle is None:
            break
        # Remove the closing edge (last to first) from avail_edges
        # The DFS already removed the edges in the path.
        # The closing edge was checked but not removed in the success case of dfs?
        # Wait, in dfs, when len(path)==n, it checks if edge (0, current) is in avail_edges.
        # It does NOT remove it.
        # So we must remove it here.
        last = cycle[-1]
        e_close = (min(0, last), max(0, last))
        if e_close in avail_edges:
            avail_edges.remove(e_close)
        else:
            # This shouldn't happen if logic is correct, but handle it
            pass
            
        found_cycles.append(cycle)
        
    if not found_cycles:
        # Fallback: return a single valid cycle if any, or empty?
        # The task requires nonempty list.
        # If we found 0, we failed. But we know 3 exist.
        # Let's try a hardcoded fallback if search fails due to timeout or bug.
        # Hardcoded cycles from previous analysis:
        # Step 1: 0,1,2...20,0
        # Step 2: 0,2,4...20,0 (check if edges exist)
        # Step 5: 0,5,10...
        
        # Let's just return the first one found if any, or a known valid one.
        # A known valid Hamiltonian cycle is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # Edges: (0,1), (1,2), ..., (19,20), (20,0).
        # (20,0) is in P['edges']? Yes, [0, 20] is there.
        # (19,20) is in P['edges']? Yes, [18, 20] and [19, 20] are there.
        # So this cycle is valid.
        
        return [[int(i) for i in range(n)]]
        
    return found_cycles

# EVOLVE-BLOCK-END
