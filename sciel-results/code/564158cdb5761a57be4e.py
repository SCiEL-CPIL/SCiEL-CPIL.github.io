import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 16
    
    # Q3 edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    # Vertices 0..7
    # Symmetry breaking: Fix Q3 vertex 0 to graph vertex g0.
    # We need to map Q3 vertices 1..7 to distinct graph vertices from {0..15} \ {g0}.
    
    # Pre-compute adjacency requirements for each Q3 vertex relative to vertex 0?
    # No, just check edges in the permutation.
    
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # To speed up, we can pre-compute for each Q3 vertex v, the list of Q3 neighbors.
    q3_neighbors = [[] for _ in range(8)]
    for u, v in q3_edges:
        q3_neighbors[u].append(v)
        q3_neighbors[v].append(u)
        
    # We will iterate g0 from 0 to 15.
    # For each g0, we need to assign Q3 vertices 1..7 to distinct graph vertices.
    # We can use backtracking for the assignment of 1..7, pruning if edges to already assigned vertices are missing.
    # This is much faster than iterating all 7! permutations if we prune.
    
    def has_q3(adj):
        """
        adj: list of 16 ints, bit i set in adj[j] means edge (i,j) exists.
        Returns True if Q3 subgraph exists.
        """
        # Fix Q3 vertex 0 to graph vertex g0
        for g0 in range(n):
            # Assignments: qv -> gv
            # qv 0 is fixed to g0
            assign = {0: g0}
            
            # Backtrack to assign qv 1..7
            # Order: 1, 2, 3, 4, 5, 6, 7
            # Heuristic: Assign the Q3 vertex with most connections to already assigned vertices first?
            # Or just fixed order 1..7.
            
            def backtrack(qv):
                if qv == 8:
                    return True
                
                # Try all graph vertices gv not in assign.values()
                # To speed up, we can pre-compute available vertices?
                # Or just loop 0..15 and check if used.
                
                # Optimization: Check constraints with already assigned neighbors
                # Neighbors of qv in Q3 that are already assigned
                # Let's just try all gv
                
                used = set(assign.values())
                
                for gv in range(n):
                    if gv in used:
                        continue
                    
                    # Check if gv is compatible with already assigned neighbors of qv
                    ok = True
                    for nq in q3_neighbors[qv]:
                        if nq in assign:
                            gn = assign[nq]
                            # Check edge (gv, gn)
                            if not (adj[gv] >> gn) & 1:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    assign[qv] = gv
                    if backtrack(qv + 1):
                        return True
                    del assign[qv]
                
                return False
            
            if backtrack(1):
                return True
                
        return False

    best_edges_count = 0
    best_adj = None
    
    # Run multiple trials
    # Each trial: Random init -> Clean -> Improve
    trials = 0
    while time.time() < deadline - 5:
        trials += 1
        random.seed(trials)
        
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Phase 1: Remove edges to make it Q3-free
        # Greedy removal: While has_q3, remove a random edge from a found Q3?
        # Simpler: Just remove random edges until valid.
        # Better: Find a Q3, remove a random edge from it.
        # But finding the specific Q3 is expensive if we just return True.
        # Let's modify has_q3 to return the edges of one found Q3?
        # Or just remove random edges. It might be slow.
        
        # Alternative: Start with a known Q3-free structure?
        # Or just use the fast check to do a "repair" phase.
        
        # Let's try:
        # 1. Random init
        # 2. While has_q3(adj):
        #      Remove a random edge.
        #      (This is slow if many Q3s)
        
        # Better approach:
        # Start with empty graph.
        # Try to add edges greedily.
        
        adj = [0] * n
        edge_count = 0
        
        # Greedy Addition
        # Try adding all possible edges in random order
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 2:
                break
            
            # Try adding (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3(adj):
                # Remove it
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                edge_count += 1
        
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_adj = adj[:]
            
        # If we have time, try to improve the best found so far
        # But let's just do many random restarts with greedy addition.
        # The greedy addition is deterministic given the order. Randomizing the order helps.
        
        # We can also try a "local search" on the best_adj found so far?
        # For now, rely on many restarts.
        
        if trials > 50:
            break

    if best_adj is None:
        # Fallback: Empty graph
        return []
        
    # Convert adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

construct()
