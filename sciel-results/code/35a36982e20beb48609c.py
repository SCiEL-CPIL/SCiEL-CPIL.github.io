import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to check if a graph is valid
    def is_valid(n, adj):
        # adj is a list of integers representing adjacency bitsets
        # Check all subsets of size 6
        # We can also check larger subsets, but if no size 6 is regular, 
        # does it imply no larger is regular? No.
        # The condition is "No set of 6 or more vertices may induce a regular subgraph".
        # So we must check ALL subsets of size >= 6.
        # This is computationally expensive for n=20.
        # Number of subsets of size 6: C(20,6) = 38760
        # Size 7: C(20,7) = 77520
        # Size 8: C(20,8) = 125970
        # ...
        # Total subsets of size >= 6 is ~ 1/2 of 2^20 = 500,000.
        # Checking each one might be too slow in pure Python if done naively.
        
        # Optimization: 
        # If a subset S is regular, any superset T is NOT necessarily regular.
        # But if a subset S is NOT regular, it doesn't help.
        # Actually, we can just check all subsets of size 6.
        # Wait, the condition is "No set of 6 or more".
        # If a set of size 7 is regular, it contains subsets of size 6.
        # Does a regular graph of size 7 contain a regular induced subgraph of size 6?
        # Not necessarily. E.g., a 7-cycle is regular (degree 2). 
        # Remove one vertex: path of length 6. Degrees: 1,2,2,2,2,1. Not regular.
        # So checking size 6 is NOT sufficient. We must check all sizes >= 6.
        
        # This makes the problem much harder.
        # However, random graphs are very unlikely to have ANY regular induced subgraph of size >= 6.
        # The probability drops exponentially with size.
        # So checking size 6 might be a good heuristic filter, but for correctness we must check all.
        
        # Given the time limit, checking all subsets of size >= 6 for n=20 is ~500k subsets.
        # For each subset, we need to compute degrees.
        # Using bitsets, we can compute degrees quickly.
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                # Check if induced subgraph on subset is regular
                # Get adjacency within subset
                # Count degrees
                # If all degrees equal, return False
                # Optimization: break early if time is running out
                if time.time() > deadline - 5:
                    return False # Assume invalid to be safe, or handle timeout differently
                
                # Efficient degree check
                # For each v in subset, count neighbors in subset
                # We can use bit operations if we represent subset as a mask
                # But combinations gives tuples.
                
                # Let's try a simpler check first.
                # If k is large, regularity is very rare.
                # Most subsets will be rejected quickly.
                
                # Implementation detail:
                # We need to be fast.
                # Let's use a precomputed list of neighbors for each vertex.
                # But bitsets are faster.
                
                # Let's stick to a simpler approach for the first pass:
                # Generate random graph, check size 6 subsets.
                # If any size 6 is regular, reject.
                # This is a necessary condition? No.
                # But if we find a graph where NO size 6 is regular, it's a strong candidate.
                # We will then verify full validity for the best candidate.
                
                pass
        
        # Re-evaluating strategy:
        # The task says "Graphs with 20 vertices are known".
        # This implies such graphs exist and can be found.
        # If I just return a hardcoded 20-vertex graph, I get 4.0.
        # Do I know one?
        # I can try to construct one using a known method: 
        # "Cubic graphs" or "High girth graphs"?
        # Actually, a random graph with p=0.5 is the most likely to work.
        
        # Let's implement the random search with full verification.
        # To speed up verification, we can use NumPy.
        
        pass

    # Strategy:
    # 1. Try to find a valid graph for n=21.
    # 2. If not found, try n=20.
    # 3. Return the best one.
    
    best_n = 5
    best_edges = []
    
    def check_validity(n, adj_bits):
        """
        adj_bits: list of ints, bit i set if edge (v, i) exists.
        Returns True if no induced subgraph of size >= 6 is regular.
        """
        # We check all subsets of size k from 6 to n.
        # To optimize, we can stop early if we find a regular one.
        
        # Precompute neighbor masks for each vertex?
        # adj_bits[v] already is the mask.
        
        for k in range(6, n + 1):
            # Iterate through all combinations of k vertices
            # Using itertools.combinations is slow for 500k iterations if the inner loop is heavy.
            # We need a fast inner loop.
            
            # Alternative: 
            # A graph is regular if all degrees are equal.
            # For a subset S, deg(v) = popcount(adj_bits[v] & mask_S).
            
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check degrees
                degs = []
                for v in subset:
                    deg = bin(adj_bits[v] & mask).count('1')
                    degs.append(deg)
                
                if len(set(degs)) == 1:
                    return False
        return True

    # This is too slow in pure Python for n=20.
    # C(20, 10) is 184k. Total subsets ~ 500k.
    # 500k * 10 (avg k) * 1 (popcount) = 5M operations.
    # Python can do ~10M-50M simple ops/sec.
    # So ~0.1-0.5 sec per check.
    # We can do ~100-1000 checks in 120s.
    # Random search might find one.
    
    # Let's optimize the check using NumPy or bit tricks.
    # Actually, `bin(x).count('1')` is slow.
    # Use `int.bit_count()` (Python 3.8+).
    
    def check_validity_fast(n, adj_bits):
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if all degrees are equal
                # Pick first vertex
                v0 = subset[0]
                d0 = (adj_bits[v0] & mask).bit_count()
                
                for v in subset[1:]:
                    if (adj_bits[v] & mask).bit_count() != d0:
                        break
                else:
                    # All degrees equal
                    return False
        return True

    # Let's try random search
    for n_target in [21, 20, 19]:
        if time.time() > deadline - 20:
            break
            
        attempts = 0
        while time.time() < deadline - 20:
            attempts += 1
            # Generate random graph
            adj_bits = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            if check_validity_fast(n_target, adj_bits):
                # Convert to edges
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if (adj_bits[i] >> j) & 1:
                            edges.append([i, j])
                
                return {'n': n_target, 'edges': edges}
        
    # If no random graph found, return the seed
    return {'n': 5, 'edges': []}
