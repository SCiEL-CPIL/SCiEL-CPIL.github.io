import random
import math
import time

def construct():
    # We will search for the best bipartite graph.
    # Q3 is a subgraph of K4,4 minus a matching.
    # A bipartite graph contains Q3 iff there exist 4 vertices in U and 4 in V
    # such that the induced subgraph has min degree >= 3.
    
    def count_bad_subgraphs(adj, a, b):
        """
        adj: list of lists, adj[i][j] is 1 if edge exists.
        a: size of U
        b: size of V
        Returns number of bad 4x4 submatrices.
        """
        if a < 4 or b < 4:
            return 0
        
        bad_count = 0
        # Precompute row sums for speed? 
        # We need sums within subsets.
        
        # Iterate all 4-subsets of U
        # Since a is small (<=8), we can just iterate combinations.
        # Using itertools.combinations is fine.
        
        # To optimize, we can precompute bitmasks or just use loops.
        # Given small size, simple loops are okay.
        
        # Let's use a helper to get 4-combinations
        # For efficiency in Python, we might want to avoid itertools if called millions of times.
        # But we only call it in the SA loop.
        
        # Actually, we can check condition more efficiently.
        # A submatrix is bad if min row sum >= 3 AND min col sum >= 3.
        
        # Let's just count them.
        
        # Optimization: If any row in U has total degree < 3, it can't be part of a bad subgraph?
        # No, it needs degree >= 3 *within* the subset.
        
        # We'll just iterate.
        
        # Generate all 4-combinations of range(a)
        # And range(b)
        
        # To speed up, we can precompute the list of 4-combs.
        # But a and b change.
        
        pass

    def has_bad_subgraph(adj, a, b):
        """Returns a bad subgraph (indices) or None."""
        if a < 4 or b < 4:
            return None
        
        # Generate combinations
        # For a=8, b=8, 4900 checks.
        
        # We can short-circuit.
        
        # Iterate u_comb
        # We'll use a simple generator for 4-combs
        def get_combinations(n, k):
            # Simple recursive or iterative
            # For small n, k, this is fine.
            # Using itertools.combinations is implemented in C, so it's fast.
            import itertools
            return itertools.combinations(range(n), k)
            
        import itertools
        
        for u_set in itertools.combinations(range(a), 4):
            # Precompute row sums for this u_set?
            # We need col sums too.
            
            # Let's just check all v_sets
            for v_set in itertools.combinations(range(b), 4):
                # Check min row sum
                min_row = 5
                min_col = 5
                
                # Row sums
                for i in u_set:
                    s = 0
                    for j in v_set:
                        s += adj[i][j]
                    if s < min_row:
                        min_row = s
                    if min_row < 3:
                        break
                
                if min_row < 3:
                    continue
                
                # Col sums
                for j in v_set:
                    s = 0
                    for i in u_set:
                        s += adj[i][j]
                    if s < min_col:
                        min_col = s
                    if min_col < 3:
                        break
                
                if min_col < 3:
                    continue
                
                # Both min row and min col >= 3
                return (u_set, v_set)
                
        return None

    def get_edge_count(adj, a, b):
        return sum(sum(row) for row in adj[:a])

    best_edges = 0
    best_graph = None
    
    # Try different partitions
    # a + b = 16
    # a from 4 to 8 (symmetric to b from 12 to 8)
    # We can just try a=4..8
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    for a in range(4, 9):
        b = 16 - a
        
        # Initialize with a random graph or dense graph
        # Dense graph is K_{a,b}
        adj = [[1 for _ in range(b)] for _ in range(a)]
        
        # Remove edges to make it Q3-free
        # While there is a bad subgraph, remove an edge
        # To be smart, remove an edge that appears in many bad subgraphs?
        # For now, random removal.
        
        # We will use Simulated Annealing to maximize edges subject to no bad subgraphs.
        
        # State: adj
        # Energy: 0 if no bad subgraph, 1 if any (or count)
        # We want Energy=0.
        
        # Start by removing edges until valid.
        while True:
            bad = has_bad_subgraph(adj, a, b)
            if bad is None:
                break
            
            u_set, v_set = bad
            # Pick a random edge in this subgraph to remove
            # Edges are (i, j) where i in u_set, j in v_set, adj[i][j]=1
            possible_edges = [(i, j) for i in u_set for j in v_set if adj[i][j] == 1]
            if not possible_edges:
                # Should not happen
                break
            i, j = random.choice(possible_edges)
            adj[i][j] = 0
            
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break
        
        # Now we have a valid graph.
        # Try to add edges (Simulated Annealing)
        current_edges = get_edge_count(adj, a, b)
        
        # SA Loop
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        # Precompute list of all potential edges (i, j)
        all_edges = [(i, j) for i in range(a) for j in range(b)]
        
        while T > min_T:
            # Pick a random edge
            i, j = random.choice(all_edges)
            
            if adj[i][j] == 1:
                # Try to remove
                adj[i][j] = 0
                new_edges = current_edges - 1
                # Check validity
                # Removing an edge cannot create a bad subgraph.
                # So new state is always valid.
                # But we want to maximize edges, so we only accept if it helps?
                # No, removing an edge decreases score. We only do it to allow adding others?
                # In SA, we might remove to escape local minima.
                # But here, constraint is "no bad subgraphs".
                # Removing an edge relaxes constraint.
                # So we can always remove.
                # But we want to ADD edges.
                # So we should bias towards adding.
                
                # Let's just try to add edges if possible.
                adj[i][j] = 1 # Revert
                continue # Skip removal for now, focus on addition
                
            else:
                # Try to add
                adj[i][j] = 1
                new_edges = current_edges + 1
                
                # Check if this creates a bad subgraph
                # Only need to check subgraphs containing (i, j)
                # This is complex to optimize.
                # Just check all?
                # With a=8, b=8, 4900 checks.
                # If we do this for every move, it might be slow.
                # But 120s is generous.
                
                bad = has_bad_subgraph(adj, a, b)
                
                if bad is None:
                    # Accept
                    current_edges = new_edges
                else:
                    # Reject
                    adj[i][j] = 0
                    # In SA, we might accept with prob exp(-1/T)
                    # But we want to stay valid.
                    # So we reject.
                    
            T *= cooling
            
            if time.time() - start_time > time_limit:
                break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = [adj[i][j] for i in range(a) for j in range(b) if adj[i][j]==1]
            # Store full graph structure
            best_a = a
            best_b = b
            best_adj = [row[:] for row in adj]
            
        if time.time() - start_time > time_limit:
            break
            
    # Convert best_adj to edge list
    if best_graph is not None:
        edges = []
        for i in range(best_a):
            for j in range(best_b):
                if best_adj[i][j] == 1:
                    # Map back to 0..15
                    # U is 0..a-1, V is a..15
                    u = i
                    v = a + j
                    edges.append([u, v])
        return edges
    else:
        # Fallback to seed
        return [[a, b] for a in range(3) for b in range(3, 16)]

