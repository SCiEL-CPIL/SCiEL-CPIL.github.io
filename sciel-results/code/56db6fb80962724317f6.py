import random
import time

def construct():
    """
    Attempts to find a 5-coloring of K_n without monochromatic C4s.
    Uses randomized backtracking search.
    """
    def is_valid_addition(neighbors, u, v, c):
        """Check if adding edge (u,v) with color c creates a C4 in color c."""
        # A C4 is formed if u and v have at least 2 common neighbors in color c.
        # neighbors[c][u] is the set of neighbors of u in color c.
        # neighbors[c][v] is the set of neighbors of v in color c.
        common = neighbors[c][u] & neighbors[c][v]
        return len(common) < 2

    def solve(N, time_limit=100):
        """
        Randomized backtracking solver for K_N 5-coloring C4-free.
        """
        start_time = time.time()
        # neighbors[c][u] = set of v such that edge (u,v) has color c
        neighbors = [ [set() for _ in range(N)] for _ in range(5) ]
        
        # List of all edges (i, j) with i < j
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        # We will process edges in a dynamic order? 
        # For simplicity and speed of implementation, let's use a static order but randomize the initial order of edges?
        # Or better: Pick the edge with the fewest available colors (MRV heuristic).
        
        # To make MRV efficient, we might need to re-evaluate all edges, which is O(E^2).
        # For N=25, E=300. E^2 = 90000. This is fine.
        
        assigned = [False] * len(edges)
        colors = [-1] * len(edges)
        
        # Map edge index to (u, v)
        # We need to map (u, v) to edge index quickly?
        # Or just keep a list of unassigned edges.
        
        # Let's use a simpler approach: Recursive backtracking with random edge selection.
        # To avoid deep recursion issues, we use a stack.
        
        # Actually, a simple greedy with restarts is often faster to implement correctly.
        # Let's try Greedy with Backtracking.
        
        def backtrack(depth, unassigned_edges):
            if time.time() - start_time > time_limit:
                return None
            
            if not unassigned_edges:
                return colors[:] # Success
                
            # Heuristic: Pick an edge from unassigned_edges that has the fewest valid colors.
            # To avoid O(E^2) overhead every step, we can just pick a random edge from the remaining.
            # But MRV is much stronger. Let's implement a light MRV.
            
            # For N <= 25, E <= 300.
            # Checking all unassigned edges for available colors is O(E * 5 * (SetIntersect)).
            # SetIntersect is fast.
            
            best_idx = -1
            min_options = 5
            best_edge = None
            best_valid_colors = None
            
            # Sample a subset of unassigned edges to find a good candidate?
            # Or just iterate all?
            # Iterating all 300 edges is fast enough.
            
            for idx in range(len(unassigned_edges)):
                u, v = unassigned_edges[idx]
                valid_colors = []
                for c in range(5):
                    if is_valid_addition(neighbors, u, v, c):
                        valid_colors.append(c)
                        
                if len(valid_colors) < min_options:
                    min_options = len(valid_colors)
                    best_idx = idx
                    best_edge = (u, v)
                    best_valid_colors = valid_colors
                    
                    if min_options == 0:
                        # Dead end, backtrack immediately
                        return None
                    if min_options == 1:
                        # Can't get better than 1, break early
                        break
                        
            if best_idx == -1:
                # This shouldn't happen if unassigned_edges is not empty
                return None
                
            # Remove the chosen edge from unassigned
            # To make removal O(1), we can swap with last and pop
            u, v = best_edge
            unassigned_edges[best_idx] = unassigned_edges[-1]
            unassigned_edges.pop()
            
            # Try each valid color
            random.shuffle(best_valid_colors)
            
            for c in best_valid_colors:
                # Assign color c to (u, v)
                neighbors[c][u].add(v)
                neighbors[c][v].add(u)
                # We need to store which edge index this was to undo?
                # Since we are using a stack of states, we should save the state.
                # But saving full neighbor sets is expensive.
                # Instead, we can just recurse and undo on return.
                
                # Store the edge index in a local variable for undo?
                # We don't strictly need the index for undo, just (u, v) and c.
                
                res = backtrack(depth + 1, unassigned_edges)
                if res is not None:
                    return res
                
                # Undo
                neighbors[c][u].remove(v)
                neighbors[c][v].remove(u)
                
            return None

        # Initial setup
        initial_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(initial_edges)
        
        # Run multiple restarts
        max_restarts = 50
        for _ in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            # Reset state
            neighbors = [ [set() for _ in range(N)] for _ in range(5) ]
            colors = [-1] * (N*(N-1)//2)
            # Note: The colors array indexing needs to match the final output format.
            # The output format requires colors in lexicographic order of (i, j).
            # My backtracker uses a dynamic list of edges.
            # I need to map the final assignment back to the lexicographic order.
            
            # Let's store the assignment in a dict: {(u,v): color}
            assignment = {}
            
            def backtrack_dict(depth, unassigned_edges):
                if time.time() - start_time > time_limit:
                    return None
                if not unassigned_edges:
                    return assignment.copy()
                
                # MRV Heuristic
                best_idx = -1
                min_options = 5
                best_edge = None
                best_valid_colors = None
                
                for idx in range(len(unassigned_edges)):
                    u, v = unassigned_edges[idx]
                    valid_colors = []
                    for c in range(5):
                        if is_valid_addition(neighbors, u, v, c):
                            valid_colors.append(c)
                            
                    if len(valid_colors) < min_options:
                        min_options = len(valid_colors)
                        best_idx = idx
                        best_edge = (u, v)
                        best_valid_colors = valid_colors
                        
                        if min_options == 0:
                            return None
                        if min_options == 1:
                            break
                            
                if best_idx == -1:
                    return None
                    
                u, v = best_edge
                unassigned_edges[best_idx] = unassigned_edges[-1]
                unassigned_edges.pop()
                
                random.shuffle(best_valid_colors)
                
                for c in best_valid_colors:
                    neighbors[c][u].add(v)
                    neighbors[c][v].add(u)
                    assignment[(u,v)] = c
                    assignment[(v,u)] = c # Just for convenience, though we only use i<j
                    
                    res = backtrack_dict(depth + 1, unassigned_edges)
                    if res is not None:
                        return res
                        
                    # Undo
                    neighbors[c][u].remove(v)
                    neighbors[c][v].remove(u)
                    del assignment[(u,v)]
                    del assignment[(v,u)]
                    
                return None

            current_edges = initial_edges[:]
            # Shuffle again for each restart? Yes.
            random.shuffle(current_edges)
            
            result = backtrack_dict(0, current_edges)
            if result is not None:
                # Format result into the required list
                # Required: one color for each pair (i,j), i<j, in lexicographic order
                final_colors = []
                for i in range(N):
                    for j in range(i+1, N):
                        final_colors.append(result[(i,j)])
                return {'n': N, 'colors': final_colors}
                
        return None

    # Try to find a solution for N from 25 down to 11
    # We prioritize higher N.
    # To be safe and fast, we might limit the time per N.
    total_time_limit = 110
    start_total = time.time()
    
    for N in range(25, 10, -1):
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 5:
            break
            
        # Allocate more time for larger N? Or just a fixed chunk?
        # Let's give each N a share of the remaining time, but cap it.
        # If N=25 fails quickly, we move to 24.
        # If N=25 takes long, we might not have time for others.
        # Let's try a fixed 10s per N, but not exceeding total.
        alloc_time = min(10, time_remaining)
        
        res = solve(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback to seed solution
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
