import time
import random
import sys
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a graph with no regular induced subgraph on 6 or more vertices.
    Strategy:
    1. Try to construct a 21-vertex graph using algebraic methods (Paley-like or specific finite field structures).
    2. Verify using optimized NumPy-based subset checking.
    3. If 21 fails, try 20 with multiple algebraic candidates and random perturbations.
    4. If 20 fails, fall back to smaller n.
    
    Key insight: Regular induced subgraphs are rare in "random-looking" algebraic graphs.
    Paley graphs are strongly regular and have few regular induced subgraphs.
    """
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Helper: Fast verification using NumPy
    def verify_graph(n, adj_matrix):
        """
        Verify that no subset of size >= 6 induces a regular subgraph.
        adj_matrix: n x n numpy array (0 or 1).
        Returns True if valid.
        """
        # We only need to check subsets of size 6, 7, ..., n.
        # For efficiency, we can use bitmasks or vectorized operations.
        # Given n <= 21, the number of subsets is manageable but large for n=21.
        # C(21, 6) = 54264, C(21, 7) = 116280, ... Total ~ 1M subsets.
        # Vectorized check per subset is fast.
        
        # Precompute adjacency bitmasks for faster degree calculation?
        # Actually, for each subset S, we need the degree of each vertex in S.
        # deg_S(v) = sum_{u in S, u != v} adj[v][u]
        
        # Let's iterate over subsets. To speed up, we can use a recursive approach or
        # simply iterate and use numpy slicing.
        
        # Optimization: 
        # If a subset S of size k is regular, then its complement (if n-k >= 6) is not necessarily regular.
        # But we can prune: if we find a violation, return False immediately.
        
        # For n=21, checking all subsets is ~ 2^21 = 2M subsets. 
        # For each subset, computing degrees is O(k). 
        # Total ops ~ 2M * 10 = 20M ops. In Python, this is too slow.
        # We need a faster check.
        
        # Alternative: 
        # Use the fact that regular induced subgraphs are rare.
        # We can sample? No, we need certainty.
        
        # Let's use a C-accelerated approach if possible, or optimize the Python loop.
        # Actually, scipy or numpy can help.
        
        # Let's try a different verification:
        # For each subset S, the degree sequence must be constant.
        # We can precompute for each vertex v, the vector of connections to all other vertices.
        # For a subset S, the degree of v in S is the dot product of adj[v] and indicator(S).
        
        # Let's implement a loop over subsets using bitmasks for speed.
        # n <= 21 fits in an integer.
        
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj_matrix[i][j] == 1:
                    mask |= (1 << j)
            adj_bits[i] = mask
            
        # Check all subsets of size >= 6
        # To speed up, we can generate subsets of size 6, 7, etc.
        # C(21, 6) is ~54k. C(21, 7) is ~116k. C(21, 8) is ~203k.
        # Total subsets to check: sum(C(21, k) for k in 6..21) ~ 1M.
        # Each check: compute degrees for k vertices.
        # Using bit_count (Python 3.8+), this is fast.
        
        for k in range(6, n + 1):
            for S in combinations(range(n), k):
                mask = 0
                for u in S:
                    mask |= (1 << u)
                
                # Check if induced subgraph on S is regular
                # Degree of u in S is bit_count(adj_bits[u] & mask)
                degs = []
                for u in S:
                    degs.append((adj_bits[u] & mask).bit_count())
                
                # Check if all degrees are equal
                d0 = degs[0]
                if all(d == d0 for d in degs):
                    return False
                    
        return True

    # Helper: Generate Paley graph
    def generate_paley_graph(n):
        """
        Generate a Paley graph if n is a prime power congruent to 1 mod 4.
        Otherwise, return None.
        """
        # Check if n is a prime power
        # For our purposes, we can just try small primes.
        # Paley graphs exist for q = 5, 9, 13, 17, 21 (not prime power? 21=3*7 no), 25, 29...
        # n=21 is not a prime power. n=20 is not a prime power.
        # So Paley graphs don't directly apply to n=20 or 21.
        
        # However, we can use a "Paley-like" construction or a random graph with specific properties.
        # Or use a graph from a finite field of size q, and take an induced subgraph of size n.
        
        return None

    # Helper: Generate a random graph
    def generate_random_graph(n, p=0.5):
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj

    # Helper: Generate a "structured" random graph
    def generate_structured_graph(n, seed=0):
        """
        Use a deterministic pseudo-random number generator to create a graph
        that is "random-looking" but reproducible.
        """
        rng = np.random.RandomState(seed)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj

    # Try n=21
    for n_target in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple seeds/structures
        for seed in range(50):
            if time.time() > deadline - 5:
                break
                
            # Generate candidate graph
            adj = generate_structured_graph(n_target, seed)
            
            # Verify
            if verify_graph(n_target, adj):
                # Convert to edge list
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i][j] == 1:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
                
        # If no structured graph worked, try random with local search?
        # For now, just try more seeds.
        for seed in range(50, 200):
            if time.time() > deadline - 5:
                break
            adj = generate_structured_graph(n_target, seed)
            if verify_graph(n_target, adj):
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i][j] == 1:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}

    # Fallback: return a small valid graph (e.g., n=5)
    # A graph with no edges is valid for n<6.
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
