import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Aims for n=20.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_n = 5
    best_edges = []
    
    # Helper to check if a set of vertices induces a regular subgraph
    def is_regular(vertices, adj):
        """
        vertices: list of vertex indices
        adj: 2D list or numpy array of adjacency
        Returns True if the induced subgraph on 'vertices' is regular and len(vertices) >= 6
        """
        n_v = len(vertices)
        if n_v < 6:
            return False
        
        # Calculate degrees within the induced subgraph
        degs = []
        for v in vertices:
            deg = 0
            for u in vertices:
                if u != v and adj[v][u]:
                    deg += 1
            degs.append(deg)
        
        # Check if all degrees are equal
        first_deg = degs[0]
        for d in degs[1:]:
            if d != first_deg:
                return False
        return True

    # Helper to check if adding a new vertex with a specific edge set creates a violation
    def check_violation(new_vertex_idx, new_edges, current_adj, current_n):
        """
        current_adj: list of lists for 0..current_n-1
        new_vertex_idx: current_n (the index of the new vertex)
        new_edges: list of indices in 0..current_n-1 that are connected to new_vertex_idx
        
        Checks all subsets of size >= 6 that include the new vertex.
        Note: Subsets not including the new vertex were already valid.
        """
        # We only need to check subsets that include the new vertex
        # Because subsets without it were already checked in previous steps.
        
        # To be efficient, we can iterate over subsets of the existing vertices
        # and check if adding the new vertex makes it regular.
        
        # However, a regular subgraph of size k including the new vertex
        # means the new vertex has degree d, and all other k-1 vertices have degree d.
        
        # Let's just build the full adjacency matrix for the candidate graph
        # and check all subsets of size >= 6? That's too slow for n=20 (2^20 subsets).
        
        # Optimization:
        # If we maintain the property that no regular subgraph of size >= 6 exists,
        # we only need to check subsets that include the new vertex.
        # Number of such subsets: sum_{k=6}^{n+1} C(n, k-1)
        # For n=19, k=6..20. This is still large.
        
        # Better approach:
        # Check if the new vertex completes a regular subgraph.
        # A regular subgraph of size m including v_new implies:
        # 1. v_new has degree d in this subgraph.
        # 2. The other m-1 vertices form a subgraph where each has degree d (if d>0) 
        #    or d=0 (empty graph on m vertices, but m>=6 so regular empty is degree 0).
        #    Actually, if d=0, it's an independent set of size m.
        #    If d>0, the other m-1 vertices must have exactly d neighbors within the set.
        
        # Let's implement a specific check:
        # For each subset S of existing vertices of size m-1 (where m-1 >= 5):
        #   Let d_new be the number of neighbors of v_new in S.
        #   The induced subgraph on S U {v_new} is regular if:
        #     1. deg(v_new) = d_new
        #     2. For all u in S, deg(u) in (S U {v_new}) = d_new
        
        # This is still expensive if we iterate all subsets.
        # However, n is small (max 20).
        # C(19, 5) = 11628. C(19, 6) = 27132. Sum is manageable if optimized?
        # 120s budget.
        
        # Let's try a simpler, potentially slower but robust check for small n:
        # We will maintain the adjacency matrix as a list of bitmasks or just lists.
        # Given the small n (<=20), we can afford to check subsets if we prune.
        
        # Actually, let's just check all subsets of the NEW graph that include the new vertex.
        # To speed up, we can precompute degrees or use bit operations.
        
        # Let's use a simpler heuristic for the check:
        # If the new vertex has degree 0 in the new graph, we must ensure no independent set of size >= 6 exists.
        # If the new vertex has degree 1, etc.
        
        # Let's just do the direct check for subsets containing the new vertex.
        # To make it fast, we represent the graph with bitmasks.
        
        pass # Placeholder for logic below

    # Re-implementing with bitmasks for speed
    def check_addition(adj_bits, new_vertex_idx, new_edges_mask):
        """
        adj_bits: list of ints, bitmask of neighbors for each vertex 0..n-1
        new_vertex_idx: n
        new_edges_mask: bitmask of neighbors for the new vertex (subset of 0..n-1)
        Returns True if valid (no regular induced subgraph of size >= 6 including new vertex)
        """
        n = new_vertex_idx
        # The new vertex's neighbors are new_edges_mask
        # We need to check all subsets S of {0..n-1} such that |S U {new_vertex_idx}| >= 6
        # i.e., |S| >= 5
        
        # For a subset S of existing vertices:
        # Let m = |S| + 1
        # The induced subgraph on S U {v_new} is regular if:
        # deg(v_new) = popcount(new_edges_mask & S)
        # AND for all u in S, deg(u in S U {v_new}) = deg(v_new)
        
        # We can iterate over all subsets of size >= 5.
        # This is 2^n subsets. For n=19, 2^19 = 524,288.
        # Doing this for every new vertex (20 times) and every random trial (say 1000) is 20 * 1000 * 500k = 10^10. Too slow.
        
        # We need a smarter check or fewer trials.
        # Observation: Regular subgraphs are rare.
        # We can prune: 
        # If d = deg(v_new), then every u in S must have exactly d neighbors in S (plus 1 if u is connected to v_new).
        # So deg_S(u) + (1 if u in new_edges_mask else 0) must equal d.
        
        # This is a strong constraint.
        # For a fixed d, the set S must consist of vertices that have specific degrees in S.
        
        # Let's iterate by possible degree d of the new vertex.
        # d can be 0..n.
        # For each d, we look for a subset S of size >= 5 such that:
        # 1. popcount(new_edges_mask & S) == d  (Wait, deg(v_new) is the number of neighbors in S)
        #    So d = popcount(new_edges_mask & S)
        # 2. For all u in S:
        #    deg(u in S) + (1 if u in new_edges_mask else 0) == d
        
        # This allows us to filter vertices in S.
        # If d=0:
        #   popcount(new_edges_mask & S) == 0 => S is disjoint from new_edges_mask.
        #   For all u in S: deg(u in S) + 0 == 0 => deg(u in S) == 0.
        #   So S must be an independent set in the induced subgraph of (V \ new_edges_mask).
        #   We just need to check if there is an independent set of size >= 5 in V \ new_edges_mask.
        
        # If d=1:
        #   popcount(new_edges_mask & S) == 1 => S contains exactly 1 neighbor of v_new.
        #   For u in S:
        #     If u is the neighbor: deg(u in S) + 1 == 1 => deg(u in S) == 0.
        #     If u is not the neighbor: deg(u in S) + 0 == 1 => deg(u in S) == 1.
        #   So S contains 1 vertex with 0 neighbors in S, and |S|-1 vertices with 1 neighbor in S.
        
        # This structure check is much faster than checking all subsets blindly.
        
        # However, implementing this for all d is complex.
        # Given the time limit, let's use a randomized local search or a simpler greedy that accepts "mostly" valid graphs and fixes?
        # No, the constraint is hard.
        
        # Alternative: Use the fact that n=20 is known to exist.
        # We can try to construct it using a specific known structure or a very targeted search.
        
        # Let's try a simpler check: 
        # We will maintain the list of "forbidden" regular subgraphs found so far? No.
        
        # Let's go back to the bitmask subset iteration but optimize it.
        # We only check subsets of size exactly 6 first? No, if size 6 is regular, size 7 might not be, but the task says "No set of 6 OR MORE".
        # So if ANY subset of size >= 6 is regular, it's invalid.
        
        # If we ensure no subset of size 6 is regular, does that imply no larger subset is regular?
        # No. A 7-vertex regular graph contains 6-vertex regular subgraphs?
        # If a 7-vertex graph is regular of degree d, does it contain a 6-vertex regular subgraph?
        # Not necessarily.
        
        # However, if we find a regular subgraph of size K > 6, it might not contain a regular subgraph of size 6.
        # But checking all sizes is hard.
        
        # Let's assume the "bottleneck" is finding the graph.
        # The observed solution used "Randomized greedy search".
        
        # Let's implement the check for ALL subsets of size >= 6 including the new vertex, but using the degree constraint pruning.
        
        # To keep it simple and fast enough for 120s:
        # We will limit the number of random restarts/trials.
        
        def is_valid_addition(adj_bits, n, new_edges_mask):
            # n is current number of vertices (index of new vertex is n)
            # new_edges_mask is the bitmask of neighbors of the new vertex among 0..n-1
            
            # Check for regular induced subgraphs of size >= 6 including the new vertex.
            # We iterate over subsets S of {0..n-1} with |S| >= 5.
            # To speed up, we can iterate by size.
            
            # Optimization: 
            # If d = deg(new vertex in S), then for all u in S, deg(u in S) + (1 if u in new_edges_mask else 0) == d.
            
            # Let's try to find such an S.
            # We can do a DFS or just iterate combinations if n is small.
            # n <= 19.
            # Combinations of 19 items taken 5 at a time: 11628.
            # Taken 6: 27132.
            # Total subsets of size >= 5: ~ 100,000.
            # For each subset, checking the degree condition takes O(|S|^2) or O(|S|) with bit counts.
            # 100,000 * 20 ops = 2,000,000 ops per check.
            # If we do 1000 checks, that's 2 * 10^9 ops. Might be too slow in Python.
            
            # We need to prune.
            # Most subsets will fail the degree condition quickly.
            
            # Let's implement a fast check using bitmasks.
            
            # Precompute for each vertex u in 0..n-1:
            # adj_bits[u] is the bitmask of neighbors in 0..n-1.
            
            # For a subset S (bitmask):
            # d = popcount(new_edges_mask & S)
            # For each u in S:
            #   deg_u_in_S = popcount(adj_bits[u] & S)
            #   conn_to_new = 1 if (new_edges_mask >> u) & 1 else 0
            #   if deg_u_in_S + conn_to_new != d: return False (not regular)
            
            # If all pass, return True (regular found -> invalid)
            
            # We want to return False if NO such S exists.
            
            # Iterating all S is the bottleneck.
            
            # Can we restrict to S that are "likely" to be regular?
            # Regular graphs are rare.
            
            # Let's just do the full check but with a hard time limit on the search.
            # If the check is too slow, we might timeout.
            
            # Alternative: Only check subsets of size 6.
            # Is it possible to have a regular induced subgraph of size 7 but no size 6?
            # Yes.
            # But maybe in the context of "maximizing n", the critical obstruction is size 6?
            # The problem states "No set of 6 or more".
            
            # Let's assume checking size 6 is sufficient for a heuristic, or we check size 6 and 7.
            
            # Let's check size 6 only first. If we find a valid graph with no size-6 regular subgraphs, it might still have a size-7 one.
            # But the probability is low?
            
            # Actually, let's just check ALL sizes but break early if found.
            
            # To make it faster, we can randomize the order of checking subsets? No, must be exhaustive.
            
            # Let's try to optimize the inner loop.
            
            # If n is small (e.g., < 15), it's fast.
            # As n grows, it slows down.
            
            # We will use a simplified check: 
            # Check subsets of size 6.
            # If none found, assume valid (heuristic).
            # This might produce an invalid graph if a size 7+ regular subgraph exists.
            # But the task requires validity.
            
            # So we MUST check all sizes.
            
            # How to speed up?
            # Use the fact that if S is regular, its subsets of size 6 might not be.
            
            # Let's stick to the full check but optimize the Python code.
            
            # We will iterate over combinations of indices.
            
            from itertools import combinations
            
            # We only care about subsets S of {0..n-1} with |S| >= 5.
            # Let's group by size.
            
            # To avoid generating all combinations every time, we can precompute?
            # No, n changes.
            
            # Let's just generate them.
            
            # For performance, we'll check size 6 first, then 7, etc.
            
            for size in range(6, n + 1):
                # S is a subset of {0..n-1} of size size-1
                # Let k = size - 1
                k = size - 1
                if k < 5: continue
                
                # Generate combinations of k vertices from 0..n-1
                # This is the expensive part.
                
                # Optimization: 
                # If we find one, return True (invalid).
                
                # We can prune combinations based on degrees?
                
                # Let's just run it.
                
                # Note: This might be too slow for n=20.
                # C(19, 5) = 11628
                # C(19, 6) = 27132
                # C(19, 7) = 50388
                # C(19, 8) = 75582
                # ...
                # Sum is ~ 200,000 combinations.
                # For each, we do popcounts.
                # 200,000 * 20 = 4,000,000 ops.
                # In Python, 4M ops is ~ 0.1s.
                # So one check is ~ 0.1s.
                # If we do 1000 checks, that's 100s.
                # We have 115s budget.
                # So we can do ~ 1000 checks total across all vertices.
                # This is tight.
                
                # We need to be smarter.
                
                # Let's only check if the new vertex has a "plausible" degree.
                
                # Actually, let's just try to construct the graph with fewer random trials.
                
                # We will do 10 trials.
                
                pass # Logic in main loop

            return False # Placeholder

        # Main Construction Loop
        # We will try to build a graph with 20 vertices.
        
        for trial in range(50): # 50 trials
            if time.time() > deadline:
                break
                
            n = 0
            adj_bits = []
            valid = True
            
            for i in range(20):
                if time.time() > deadline:
                    valid = False
                    break
                
                # Try to add vertex i
                # Randomly choose edges
                found_valid_edges = False
                attempts = 0
                max_attempts = 500
                
                while attempts < max_attempts and not found_valid_edges:
                    attempts += 1
                    
                    # Random mask
                    # Probability of edge: 0.5
                    mask = 0
                    for j in range(n):
                        if random.random() < 0.5:
                            mask |= (1 << j)
                    
                    # Check if adding vertex i with this mask is valid
                    # We need to check subsets of {0..n-1} of size >= 5
                    # that form a regular subgraph with i.
                    
                    # Optimized Check
                    # d = popcount(mask & S)
                    # For all u in S: popcount(adj_bits[u] & S) + (1 if mask>>u & 1 else 0) == d
                    
                    # We iterate over subsets S of {0..n-1} of size k (k>=5)
                    # To speed up, we can check if there is ANY such S.
                    
                    # Let's use a helper that returns True if INVALID (regular found)
                    
                    def is_invalid_addition(adj_bits, n, mask):
                        # Check subsets of size k=5..n-1
                        # Note: The subgraph size is k+1.
                        # We need k+1 >= 6 => k >= 5.
                        
                        # We can prune: 
                        # If d = popcount(mask & S), then d must be between 0 and k.
                        
                        # Let's just iterate.
                        
                        # To make it faster, we can check specific structures?
                        # No, let's just iterate combinations.
                        
                        # We will use itertools.combinations
                        # This is slow in Python for large n.
                        
                        # Alternative: 
                        # We can precompute the list of all subsets of size >= 5? No.
                        
                        # Let's try to limit the search space.
                        # If n > 15, this might be too slow.
                        
                        # Let's assume n is small enough or we use a heuristic.
                        
                        # Heuristic: Only check subsets of size 5 (subgraph size 6).
                        # If we miss a size 7 regular subgraph, we might return an invalid graph.
                        # But the task requires validity.
                        
                        # Let's check size 6 and 7.
                        
                        from itertools import combinations
                        
                        # Size 6 subgraph => k=5
                        for S in combinations(range(n), 5):
                            S_mask = 0
                            for u in S:
                                S_mask |= (1 << u)
                            
                            d = bin(mask & S_mask).count('1')
                            
                            # Check degrees
                            valid_reg = True
                            for u in S:
                                deg_in_S = bin(adj_bits[u] & S_mask).count('1')
                                conn_to_new = 1 if (mask >> u) & 1 else 0
                                if deg_in_S + conn_to_new != d:
                                    valid_reg = False
                                    break
                            
                            if valid_reg:
                                return True # Found regular subgraph, so invalid addition
                        
                        # Size 7 subgraph => k=6
                        for S in combinations(range(n), 6):
                            S_mask = 0
                            for u in S:
                                S_mask |= (1 << u)
                            
                            d = bin(mask & S_mask).count('1')
                            
                            valid_reg = True
                            for u in S:
                                deg_in_S = bin(adj_bits[u] & S_mask).count('1')
                                conn_to_new = 1 if (mask >> u) & 1 else 0
                                if deg_in_S + conn_to_new != d:
                                    valid_reg = False
                                    break
                            
                            if valid_reg:
                                return True
                        
                        # We skip larger sizes for speed, assuming they are rare or implied?
                        # Risky. But let's try.
                        
                        return False
                    
                    if not is_invalid_addition(adj_bits, n, mask):
                        # Valid
                        adj_bits.append(mask)
                        found_valid_edges = True
                        n += 1
                        break
                else:
                    # Failed to add vertex
                    valid = False
                    break
            
            if valid and n == 20:
                # Construct edges list
                edges = []
                for i in range(20):
                    mask = adj_bits[i]
                    for j in range(i):
                        if (mask >> j) & 1:
                            edges.append([j, i])
                
                # Final verification (optional, but good for safety)
                # We might have missed size > 7.
                # Let's do a full check on the final graph?
                # Full check on 20 vertices: 2^20 subsets. Too slow.
                # But we only need to check subsets of size >= 6.
                # C(20, 6) = 38760. C(20, 7) = 77520.
                # Total ~ 100k subsets.
                # Check each: O(20). 2M ops. ~ 0.1s.
                # So we CAN do a full check on the final graph.
                
                def is_valid_final_graph(edges, n):
                    # Build adj matrix
                    adj = [[0]*n for _ in range(n)]
                    for u, v in edges:
                        adj[u][v] = 1
                        adj[v][u] = 1
                    
                    # Check all subsets of size >= 6
                    from itertools import combinations
                    for size in range(6, n+1):
                        for S in combinations(range(n), size):
                            # Check if regular
                            degs = []
                            for u in S:
                                deg = 0
                                for v in S:
                                    if u != v and adj[u][v]:
                                        deg += 1
                                degs.append(deg)
                            if all(d == degs[0] for d in degs):
                                return False
                    return True
                
                if is_valid_final_graph(edges, 20):
                    return {'n': 20, 'edges': edges}
                else:
                    # Invalid, try next trial
                    continue
            else:
                # Didn't reach 20
                # Return best found? 
                # We can return a smaller valid graph.
                # But we need to ensure it's valid.
                # We only checked up to size 7 during construction.
                # So we must verify the final smaller graph too.
                
                if n >= 6:
                    edges = []
                    for i in range(n):
                        mask = adj_bits[i]
                        for j in range(i):
                            if (mask >> j) & 1:
                                edges.append([j, i])
                    
                    if is_valid_final_graph(edges, n):
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                elif n > best_n:
                    best_n = n
                    best_edges = []
                
                # Update best
                if n > best_n:
                    best_n = n
                    # Reconstruct edges for best_n
                    edges = []
                    for i in range(best_n):
                        mask = adj_bits[i]
                        for j in range(i):
                            if (mask >> j) & 1:
                                edges.append([j, i])
                    best_edges = edges

    # Fallback
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
