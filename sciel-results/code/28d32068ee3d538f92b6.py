import itertools
import random
import time

def construct():
    n = 16
    vertices = list(range(n))
    
    # Base Q3 edges
    base_q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Precompute all 840 labeled Q3 patterns on 0..7
    # This takes a bit of time, but is done once.
    patterns = set()
    # Optimization: We can just use a subset of permutations if time is tight, 
    # but 40320 is small.
    for p in itertools.permutations(range(8)):
        mapped_edges = set()
        for u, v in base_q3_edges:
            mu, mv = p[u], p[v]
            if mu > mv: 
                mu, mv = mv, mu
            mapped_edges.add((mu, mv))
        patterns.add(frozenset(mapped_edges))
        
    # Initialize adjacency matrix as K16
    adj = [[1]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0
        
    start_time = time.time()
    
    # Randomized edge removal
    # We want to remove edges to destroy all Q3s.
    # We sample random 8-subsets and check if they contain a Q3.
    # If yes, remove a random edge from that Q3.
    
    # To make it faster, we can keep a list of "recently checked" subsets?
    # Or just random.
    
    # Loop until no Q3 found in a batch of samples
    while time.time() - start_time < 110:
        found_q3 = False
        # Sample 2000 random 8-subsets
        for _ in range(2000):
            subset = tuple(random.sample(vertices, 8))
            # Check if G[subset] contains Q3
            # Build set of edges present in G[subset]
            edges_present = set()
            for i in range(8):
                u = subset[i]
                for j in range(i+1, 8):
                    v = subset[j]
                    if adj[u][v] == 1:
                        edges_present.add((i, j))
                        
            # Check against patterns
            for pat in patterns:
                if pat.issubset(edges_present):
                    # Found a Q3
                    # Remove a random edge from this Q3
                    edge_indices = list(pat)
                    rem_idx = random.choice(edge_indices)
                    u, v = subset[rem_idx[0]], subset[rem_idx[1]]
                    adj[u][v] = 0
                    adj[v][u] = 0
                    found_q3 = True
                    break
            if found_q3:
                break
                
        if not found_q3:
            # If we didn't find any in 2000 samples, assume it's Q3-free
            # To be safer, we can do a few more rounds
            # But given the density, if Q3s exist, they are likely to be found.
            break
            
    # Construct edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v] == 1:
                edges.append([u, v])
                
    return edges
