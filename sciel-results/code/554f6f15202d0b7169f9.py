import numpy as np
import random
import itertools

def construct():
    # Set a seed for reproducibility? No, random is fine, but maybe fixed seed for stability?
    # The task says "Maximize the number of edges".
    # We will run a local search.
    
    n = 16
    # Start with a dense random graph
    p = 0.85
    # Generate random graph
    # Ensure no self loops
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Precompute 8-subsets
    subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute bipartitions of 8 items into two sets of 4
    # Represented as tuples of indices in the 8-set
    # We can just iterate k from 0 to 70 or use combinations
    # Actually, we can just check all 4-subsets of the 8-set as the "left" side.
    # There are C(8,4) = 70 such subsets.
    
    # To make it faster, we can just iterate over the 70 masks.
    # But simple loop is fine.
    
    max_iterations = 500
    for _ in range(max_iterations):
        found = False
        # Shuffle subsets to avoid bias?
        # random.shuffle(subsets)
        
        for s in subsets:
            # Extract subgraph
            # indices in s are the vertices
            # Get the 8x8 adjacency submatrix
            sub_adj = adj[np.ix_(s, s)]
            
            # Check all 4-subsets of the 8 positions
            # Positions are 0..7
            # We need to pick 4 positions for the left side
            # The other 4 are the right side
            # We can iterate over combinations of range(8), 4
            
            # Optimization: Only check if the subgraph has enough edges?
            # Q3 has 12 edges.
            # If subgraph has < 12 edges, skip.
            if np.sum(sub_adj) < 24: # sum is 2 * edges
                continue
                
            for left_idx in itertools.combinations(range(8), 4):
                left_set = set(left_idx)
                right_set = set(range(8)) - left_set
                
                # Get the bipartite adjacency matrix
                # Rows: left_idx, Cols: right_idx
                # But we need to be careful with mapping.
                # sub_adj[i, j] is edge between s[i] and s[j].
                
                # Extract the 4x4 block
                # left_indices_in_sub: list of indices in 0..7
                # right_indices_in_sub: list of indices in 0..7
                
                left_list = list(left_idx)
                right_list = list(right_set)
                
                # The block is sub_adj[np.ix_(left_list, right_list)]
                block = sub_adj[np.ix_(left_list, right_list)]
                
                # Check if it is 3-regular
                # Row sums should be 3
                # Col sums should be 3
                if np.all(block.sum(axis=1) == 3) and np.all(block.sum(axis=0) == 3):
                    # Found a Q3
                    # Remove a random edge from this Q3
                    # The edges are the 1s in the block
                    # Map back to original graph indices
                    # left_list[i] corresponds to s[left_list[i]]
                    # right_list[j] corresponds to s[right_list[j]]
                    
                    # Get list of (i, j) where block[i, j] == 1
                    edges_in_q3 = [(s[i], s[j]) for i in left_list for j in right_list if block[i, j] == 1]
                    
                    # Pick a random edge
                    u, v = random.choice(edges_in_q3)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    
                    found = True
                    break
            if found:
                break
        
        if not found:
            break
            
    # Convert adjacency matrix to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
                
    return edges
