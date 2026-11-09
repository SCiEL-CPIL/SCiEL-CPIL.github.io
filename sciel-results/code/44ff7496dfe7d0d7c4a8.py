import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: 
    1. Try to construct a graph on n=21 using a modular arithmetic definition 
       (e.g., quadratic residues or similar) which often avoids symmetric structures.
    2. Verify the graph using a vectorized check for regular induced subgraphs.
    3. If invalid or time runs out, fallback to a known valid smaller graph or 
       a repaired version.
    
    A graph G is regular induced on a subset S if all vertices in S have the same 
    degree within the induced subgraph G[S].
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Helper to check if a graph has no regular induced subgraph of size >= 6
    def is_valid_graph(n, adj_matrix):
        """
        Checks if the graph defined by adj_matrix (n x n boolean) has no 
        regular induced subgraph on 6 or more vertices.
        """
        # For each subset of size k >= 6, check if it induces a regular graph.
        # A graph on k vertices is regular if all vertices have the same degree.
        # Note: The empty graph (k=0,1) and single vertex are trivially regular? 
        # Usually "regular graph" implies k >= 2 or specific context. 
        # The problem says "No set of 6 or more vertices may induce a regular subgraph".
        # A subgraph is regular if every vertex has the same degree within the subgraph.
        
        # Optimization: 
        # For a subset S of size k, let d_i be the degree of vertex i in G[S].
        # G[S] is regular if min(d_i) == max(d_i) for all i in S.
        
        # We can precompute degrees in the whole graph, but induced degrees depend on S.
        # Efficient check:
        # Iterate over combinations? Too slow for n=21 (C(21,6) ~ 54k, C(21,10) ~ 352k, etc.)
        # Total subsets of size >= 6 is large.
        # C(21, 6) = 54264
        # C(21, 7) = 116280
        # C(21, 8) = 203490
        # ...
        # C(21, 10) = 352716
        # Sum is roughly 2^21 = 2M subsets. 
        # Checking each subset by summing degrees is O(k^2) or O(k).
        # With NumPy, we can vectorize the degree calculation for a batch of subsets?
        # Or just iterate and use bit operations?
        
        # Let's use a bitset representation for the adjacency matrix.
        # adj_bits[i] is an integer where bit j is 1 if edge (i,j) exists.
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj_matrix[i, j]:
                    mask |= (1 << j)
            adj_bits[i] = mask

        # We need to check all subsets S with |S| >= 6.
        # For a subset S, the degree of vertex u in S is popcount(adj_bits[u] & mask_S).
        # We need min_deg == max_deg.
        
        # To speed this up, we can iterate by size k from 6 to n.
        # If we find one regular induced subgraph, it's invalid.
        
        # We can prune: if a subset S is regular, any superset isn't necessarily regular, 
        # but we just need to find ONE.
        
        # Let's try a randomized verification first? No, must be certain.
        
        # Efficient iteration of combinations using itertools is in C, so it's fast.
        # The bottleneck is the degree calculation.
        # popcount in Python 3.10+ is int.bit_count().
        
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Calculate degrees for vertices in combo
                # We only need to check if they are all equal.
                # We can compute degrees for the first vertex, then check others.
                first_v = combo[0]
                d0 = (adj_bits[first_v] & mask).bit_count()
                
                is_reg = True
                for v in combo[1:]:
                    dv = (adj_bits[v] & mask).bit_count()
                    if dv != d0:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    # Helper to generate edges from adjacency matrix
    def get_edges(adj_matrix):
        edges = []
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j]:
                    edges.append([i, j])
        return edges

    # Strategy: Try n=21 with a specific construction.
    # Paley graph on q vertices (q ≡ 1 mod 4 prime power) is a strongly regular graph.
    # Paley(21) doesn't exist (21 is not prime power).
    # Paley(19) exists. Paley(25) exists.
    # Let's try a random graph with density 0.5 and then repair?
    # Or try specific modular definitions.
    
    # Let's try a "modular" graph on n=21:
    # Edge (u, v) if (v - u) mod 21 is in a specific set D.
    # If D is symmetric and doesn't contain 0, it's a circulant graph.
    # Circulant graphs often have many regular induced subgraphs (e.g., cycles).
    # So modular arithmetic might be bad.
    
    # Better: Use a known construction or random search with verification.
    # Since C(21, k) is manageable, we can verify quickly.
    
    # Let's try to find a graph on n=21.
    # Start with a random graph and use local search to remove regular subgraphs.
    # Or just try several random graphs.
    
    # Verification cost for n=21:
    # ~2M subsets. Each check is O(k) bit counts. 
    # 2M * 10 ops = 20M ops. In Python, this might take ~5-10 seconds.
    # So we can try ~10-20 random graphs in 120s? 
    # Actually, itertools.combinations overhead is significant.
    # Let's optimize the check.
    
    def fast_check(n, adj_bits):
        # Precompute popcounts for all masks? No, too many.
        # Just iterate.
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Optimization: Check degrees
                # We can stop early if we find a mismatch.
                # Order vertices in combo such that we check high-degree vertices first?
                # No, just check.
                
                d0 = (adj_bits[combo[0]] & mask).bit_count()
                valid = True
                for v in combo[1:]:
                    if (adj_bits[v] & mask).bit_count() != d0:
                        valid = False
                        break
                if valid:
                    return False
        return True

    # Try n=21
    n_target = 21
    np.random.seed(42)
    
    best_graph = None
    best_n = 5
    best_edges = []
    
    # Try a few random graphs on n=21
    # Density 0.5 is usually good for avoiding regular subgraphs?
    # Actually, regular subgraphs are more likely in dense or sparse graphs?
    # A 5-regular graph on 21 vertices is impossible (21*5 is odd).
    # So any 21-vertex graph cannot be 5-regular.
    # But induced subgraphs can be regular.
    
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        # Generate random graph
        # Use numpy for speed
        adj = (np.random.rand(n_target, n_target) < 0.5).astype(int)
        np.fill_diagonal(adj, 0)
        adj = (adj + adj.T) // 2 # Symmetrize, though random < 0.5 is already symmetric if we take upper triangle?
        # Actually np.random.rand is independent. 
        # Better:
        adj = np.zeros((n_target, n_target), dtype=int)
        for i in range(n_target):
            for j in range(i+1, n_target):
                if np.random.rand() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Convert to bit lists
        adj_bits = [0] * n_target
        for i in range(n_target):
            mask = 0
            for j in range(n_target):
                if adj[i, j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
            
        if fast_check(n_target, adj_bits):
            # Valid graph found!
            edges = get_edges(adj)
            return {'n': n_target, 'edges': edges}
        
        # If not found, maybe try n=20?
        # The task says graphs with 20 vertices are known.
        # If we can't find 21 quickly, we should return a valid 20 to score > 1.0 (since seed is 5).
        # Score = n / 5. So n=20 gives score 4.0. n=21 gives 4.2.
        # We must ensure we return a VALID graph.
        
        # Fallback: Try to construct a valid graph on n=20 using similar method or known structure.
        # Let's try n=20 random graphs if n=21 fails quickly.
        if attempts == 5:
            # Switch to n=20
            n_target = 20
            # Reset attempts counter logic? Just keep trying n=20.
            pass
        elif attempts == 10:
            # If n=20 also fails, try n=19?
            # But we know 20 exists. Random search should find it.
            pass
            
    # If loop finishes without finding a valid graph (unlikely for n<=20 with random search),
    # return a trivial valid graph to avoid 0 score.
    # Empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
