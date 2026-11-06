<<<<<<< SEARCH
"""Rewritten program: C6-free graph on 40 vertices, maximizing edges."""
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a C6-free graph on 40 vertices with many edges.
    
    Strategy:
    1. Initialize with a random regular-ish graph (or 8 K5s).
    2. Use bitset-based local search to add edges and perform swaps.
    
    Returns a list of edges [u, v].
    """
    n = 40
    rng = random.Random(42)
    
    # Initialize adjacency as bitsets (integers)
    adj = [0] * n
    edges = []
    
    # Step 1: Initialize with 8 disjoint K5s
    # This gives 8 * 10 = 80 edges.
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
=======
"""Rewritten program: C6-free graph on 40 vertices, maximizing edges."""
import random
import numpy as np
import time

def construct():
    """
    Construct a C6-free graph on 40 vertices with many edges.
    
    Strategy:
    1. Initialize with a random 4-regular graph.
    2. Compute spectral vector (eigenvector of second largest eigenvalue).
    3. Sort non-edges by spectral weight product.
    4. Greedily add edges in that order if C6-free.
    5. Local search with swaps if stuck.
    
    Returns a list of edges [u, v].
    """
    n = 40
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for overhead
    
    # Initialize adjacency as bitsets (integers)
    adj = [0] * n
    edges = []
    
    # Step 1: Initialize with a random 4-regular graph
    # Use a simple random regular graph generator
    def generate_random_regular(n, d, rng):
        # Create a list of stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        rng.shuffle(stubs)
        
        adj_bits = [0] * n
        edge_set = set()
        
        # Pair up stubs
        i = 0
        while i < len(stubs):
            u = stubs[i]
            v = stubs[i+1]
            i += 2
            if u == v:
                # Self-loop, swap with next
                if i < len(stubs):
                    v = stubs[i]
                    i += 1
                    if u == v:
                        continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                # Multi-edge, try to fix by swapping
                # For simplicity, just skip or restart. 
                # With n=40, d=4, this is rare.
                # We'll just restart the generation if this happens often.
                # But for now, let's just add it and hope, or better, use a different method.
                # Actually, let's use a different initialization: 
                # Random permutation based.
                pass
            
        # Simpler initialization: Random graph G(n, p) and prune to regular?
        # Or just use the K5 method but randomize the partition?
        # Let's stick to a simple random graph with p=0.2 and then check C6s.
        # Or better: Start with a random 4-regular graph using a known algorithm.
        
        # Let's implement a simple random regular graph generator properly.
        while True:
            adj_bits = [0] * n
            edge_list = []
            # Fisher-Yates shuffle on stubs
            stubs = []
            for i in range(n):
                stubs.extend([i] * d)
            rng.shuffle(stubs)
            
            valid = True
            for i in range(0, n * d, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if adj_bits[u] & (1 << v):
                    valid = False
                    break
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                if u > v:
                    u, v = v, u
                edge_list.append((u, v))
            
            if valid:
                return adj_bits, edge_list

    adj, initial_edges = generate_random_regular(n, 4, rng)
    edges = [[u, v] for u, v in initial_edges]
    
    # Step 2: Compute spectral vector
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
        
    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # The largest eigenvalue corresponds to the principal eigenvector (all positive).
    # We want the second largest to guide the structure (bipartite-like or community structure).
    # eigh returns eigenvalues in ascending order.
    # So the largest is eigvals[-1], second largest is eigvals[-2].
    # Let's take the eigenvector corresponding to the second largest eigenvalue.
    if n > 1:
        idx = -2
    else:
        idx = -1
    v_spec = eigvecs[:, idx]
    
    # Squared components for probability weighting
    w = np.square(v_spec)
    
>>>>>>> REPLACE

Now we need to replace the local search logic to use this spectral ordering and a faster C6 check.

