import numpy as np
import random
import time

def construct():
    # Try to find a valid graph using Circulant Graph Search
    # This is a different approach from the random local search in the seed.
    
    # Try n from 35 down to 1
    for n in range(35, 0, -1):
        result = _find_circulant_graph(n)
        if result is not None:
            return result
            
    # Fallback: Paley graph of order 17
    # This is a known K4-free graph with alpha=5
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _find_circulant_graph(n):
    # For a circulant graph on n vertices, edges are determined by a set of differences S
    # S is a subset of {1, ..., floor(n/2)}
    # The graph is symmetric: if d in S, then -d in S (mod n)
    
    max_diff = n // 2
    possible_diffs = list(range(1, max_diff + 1))
    
    # We will try random subsets of possible_diffs
    # Number of tries depends on n to balance speed and coverage
    num_tries = 1000 if n <= 30 else 5000
    
    start_time = time.time()
    timeout = 10.0 # Allow up to 10s for each n
    
    for _ in range(num_tries):
        if time.time() - start_time > timeout:
            break
        
        # Generate a random subset S
        # Bias towards certain densities? Random is fine for start.
        p = 0.5
        S = set()
        for d in possible_diffs:
            if random.random() < p:
                S.add(d)
        
        # Construct adjacency matrix
        adj = [0] * n
        for i in range(n):
            for d in S:
                j1 = (i + d) % n
                j2 = (i - d) % n
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        
        # Check K4
        if _has_k4(adj, n):
            continue
            
        # Check Independence Number
        if _has_indep6(adj, n):
            continue
            
        # Valid graph found
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    return None

def _has_k4(adj, n):
    # Check for K4 using bitsets
    # For each pair (u, v), check if they have 2 common neighbors
    # If yes, we need to ensure those neighbors are connected?
    # No, K4 is 4 vertices all connected.
    # If u, v are connected, and w, x are common neighbors, and w-x is connected, then {u,v,w,x} is K4.
    
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            # Remove u and v from common
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            # Iterate through common neighbors
            # We need to find two common neighbors w, x such that w-x is an edge
            # This is equivalent to checking if the induced subgraph on common neighbors has an edge
            # Or simpler: check if any pair in common is connected
            
            # To speed up, we can just check if the number of common neighbors is small?
            # No, just iterate.
            c = common
            while c:
                lsb_w = c & -c
                w = lsb_w.bit_length() - 1
                c ^= lsb_w
                
                # Check if w is connected to any other common neighbor
                # We can check adj[w] & common (excluding w itself)
                # But we must exclude u and v? They are already removed from common.
                # We must exclude w itself.
                neighbors_in_common = adj[w] & common
                # Remove w
                neighbors_in_common &= ~(1 << w)
                
                if neighbors_in_common:
                    return True
                    
    return False

def _has_indep6(adj, n):
    # Check if there is an independent set of size 6
    # Equivalent to K6 in the complement graph
    # We can use a simple backtracking search for a clique of size 6 in the complement
    
    # Build complement adjacency
    comp_adj = [0] * n
    mask_all = (1 << n) - 1
    for i in range(n):
        # Complement of adj[i] excluding i itself
        comp_adj[i] = mask_all ^ (1 << i) ^ adj[i]
        
    def find_clique(mask, k):
        if k == 0:
            return True
        if k == 1:
            return bool(mask)
            
        # Heuristic: pick a vertex with high degree in the current subgraph?
        # Or just iterate
        c = mask
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # Next candidates must be neighbors of v in the complement
            next_mask = mask & comp_adj[v] & (mask_all << (v + 1)) # Only consider vertices > v to avoid duplicates?
            # Actually, for clique finding, we just need to ensure we don't revisit.
            # Standard Bron-Kerbosch or simple recursion.
            # Since k=6 is small, simple recursion is fine.
            
            if next_mask:
                if find_clique(next_mask, k - 1):
                    return True
                    
        return False

    # We can start with any vertex
    # To speed up, we can try to prune.
    # If a vertex has degree < 5 in complement, it can't be part of a K6.
    # But we just need to find one.
    
    # Simple approach: iterate through all subsets? No.
    # Use the recursive function starting with the full mask.
    # But we need to be careful about complexity.
    # For n=35, this might be slow if called many times.
    
    # Optimization: If the graph is dense in complement (sparse in original), 
    # independent sets are more likely.
    
    # Let's try a simple greedy check first?
    # No, must be exact.
    
    # We will use the recursive function.
    # To avoid deep recursion overhead, we can use a bitset-based approach.
    
    return find_clique(mask_all, 6)
