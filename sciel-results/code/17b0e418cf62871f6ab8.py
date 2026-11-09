"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all edges as a set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    def find_hamiltonian_cycle(used_edges, start, order_seed):
        """Find a Hamiltonian cycle in the graph with edges not in used_edges.
        Returns list of vertices (cycle) or None."""
        # Build available adjacency (edges not used)
        avail = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail[u].add(v)
        
        # Check each vertex has at least 2 available edges
        for u in range(n):
            if len(avail[u]) < 2:
                return None
        
        # Start from given start vertex
        # We fix start to be the smallest vertex for canonical form
        # Use backtracking with Warnsdorff's rule
        
        path = [start]
        used_in_path = {start}
        
        # Degree of each vertex in the partial path (how many path edges incident)
        # For vertices in path (except start and current), they have degree 2
        # start has degree 1 so far, current has degree 1 so far
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            cur = path[-1]
            if len(path) == n:
                # Check if cur connects back to start
                if start in avail[cur]:
                    return list(path)
                return None
            
            # Get neighbors of cur not yet in path
            neighbors = [v for v in avail[cur] if v not in used_in_path]
            
            if not neighbors:
                return None
            
            # Pruning: for each vertex not in path, check if it can still be completed
            # A vertex v not in path needs at least 2 available neighbors that are
            # either not in path or will be in path
            # Simpler pruning: if any vertex not in path has 0 available neighbors
            # that are not in path (and not start unless it's the last), it's stuck
            # Actually: vertices not in path need >= 2 neighbors in (avail) that are
            # not already "closed" 
            # A vertex is "closed" if both its path-edges are determined
            # For a vertex not in path: its available degree must be >= 2, but some
            # of its available neighbors might be in path (giving it a path edge)
            # Let me think differently:
            # After we place all vertices, each vertex in path (not start, not cur) 
            # has exactly 2 path edges. Start has 1 (to first) + 1 (to last, pending).
            # Cur has 1 (to prev) + 1 (to next, pending).
            # For a vertex v NOT in path: it needs 2 edges in the final path.
            # These edges go to vertices that will be in path. All vertices will be
            # in path. So v needs at least 2 neighbors in avail[v] that are not
            # "locked out". A neighbor u of v is "locked out" for v if u is in path
            # and u already has 2 path edges (i.e., u is not start, not cur, and not
            # the first vertex after start... actually u is internal in path).
            # Internal vertices in path (index 1 to n-2) have 2 path edges, so they
            # can't connect to v. Start (index 0) has 1 path edge, can connect to v
            # only if v is the last vertex. Cur (index n-1) has 1 path edge, can
            # connect to v.
            # So for v not in path: available neighbors that can still connect to v
            # are: start (only if v would be last), cur, and other not-in-path vertices.
            # This is complex. Let's use a simpler but effective pruning:
            # For each vertex v not in path, count neighbors in avail[v] that are
            # not "saturated". A vertex u is saturated if it's in path and not
            # start and not cur (i.e., it has 2 path edges already).
            # v needs at least 2 non-saturated neighbors in avail[v].
            
            # Build set of saturated vertices
            # Saturated = vertices in path[1:-1] (internal, have 2 edges)
            # Start (path[0]) is not saturated (has 1 edge, can take 1 more)
            # Cur (path[-1]) is not saturated (has 1 edge, can take 1 more)
            saturated = set(path[1:-1]) if len(path) > 2 else set()
            
            for v in range(n):
                if v in used_in_path:
                    continue
                # Count non-saturated available neighbors of v
                count = 0
                for u in avail[v]:
                    if u not in saturated:
                        count += 1
                if count < 2:
                    return None
            
            # Order neighbors by Warnsdorff: prefer neighbors with fewer onward options
            def warnsdorff_score(v):
                if v == start:
                    # If we go to start, we close the cycle (only valid if len==n)
                    # But we only reach here if len < n, so start shouldn't be chosen
                    # unless it's the closing move. Actually start IS in used_in_path,
                    # so it won't be in neighbors list. Good.
                    return 0
                # Count available neighbors of v not in path
                return len([u for u in avail[v] if u not in used_in_path])
            
            neighbors.sort(key=lambda v: warnsdorff_score(v))
            
            for v in neighbors:
                path.append(v)
                used_in_path.add(v)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                used_in_path.discard(v)
            
            return None
        
        result = backtrack()
        return result
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    # Strategy: greedy with multiple random restarts
    
    best_cycles = []
    
    # First, try a deterministic approach: always pick start=0
    # Then try random orderings
    
    # We'll do multiple attempts
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        used_edges = set()
        cycles = []
        
        success = True
        for cycle_idx in range(3):
            # Choose start vertex (use 0 for first, then maybe different)
            start = 0
            
            # Try to find a cycle
            cycle = find_hamiltonian_cycle(used_edges, start, attempt)
            
            if cycle is None:
                # Try different start vertices
                found = False
                for alt_start in range(1, n):
                    if time.time() > deadline:
                        break
                    cycle = find_hamiltonian_cycle(used_edges, alt_start, attempt + alt_start)
                    if cycle is not None:
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            # Extract edges from cycle
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                cycle_edges.add(e)
            
            # Check no edge conflict
            if cycle_edges & used_edges:
                success = False
                break
            
            used_edges |= cycle_edges
            cycles.append(cycle)
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
        
        if len(best_cycles) == 3:
            break
    
    # Return the best set of cycles found
    # Convert to proper format: list of lists of ints
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
