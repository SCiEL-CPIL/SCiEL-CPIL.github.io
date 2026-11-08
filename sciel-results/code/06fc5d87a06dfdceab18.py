"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # safety margin from 150s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors

    # Build adjacency list for conflict graph
    # Conflict if dot product > 2
    # For weight-4 vectors, this happens if they share 3 or 4 positions with same signs.
    # We can compute this efficiently or just use the Gram matrix for N=5280.
    # G = 5280x5280 matrix of ints. 5280^2 * 8 bytes = 224 MB. Might be tight but OK.
    # Let's try to be memory efficient.
    
    # Actually, let's just build the adjacency list directly to save memory and time in SA.
    # But computing all pairs is O(N^2 * D) = 5280^2 * 11 ~ 300M ops. 
    # In Python, this is slow. 
    # Using NumPy: G = vectors @ vectors.T is fast (BLAS).
    # 5280x5280 matrix is 224MB. 
    # Let's use the Gram matrix approach as it's vectorized.
    
    G = vectors @ vectors.T
    # Conflict if G[i,j] > 2
    # We need adjacency list for SA speed.
    # Convert boolean mask to adjacency list.
    
    conflict_mask = G > 2
    np.fill_diagonal(conflict_mask, False)
    
    # Build adjacency list
    # This might take a few seconds.
    adj = [np.where(conflict_mask[i])[0] for i in range(n)]
    # Free memory
    del G
    del conflict_mask
    
    # Simulated Annealing
    # State: x (bool array)
    # Energy: -sum(x) + lambda * sum_conflicts(x)
    
    best_x = np.zeros(n, dtype=bool)
    best_energy = np.inf
    
    np.random.seed(42)
    
    # Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.999
    lambda_pen = 10.0 # Penalty weight for conflicts
    
    # Initialize x randomly with some density
    # Expected independent set size ~ N/(d+1) = 92.
    # Density ~ 92/5280 ~ 0.017
    # Start with a greedy solution to have a good starting point.
    
    # Greedy Initialization
    x = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    # Random order greedy
    order = np.random.permutation(n)
    for idx in order:
        if not remaining[idx]:
            continue
        x[idx] = True
        remaining[adj[idx]] = False
        remaining[idx] = False
    
    current_x = x.copy()
    # Calculate current energy
    # Conflicts: sum of x_i x_j for edges
    # We can compute this by iterating over edges or using matrix.
    # Since we have adj list, we can sum for each i in set: count neighbors in set.
    # E_conf = 0.5 * sum_{i in set} (number of neighbors in set)
    
    def calc_energy(x_vec):
        # Returns -sum(x_vec) + lambda * conflicts
        # Conflicts = 0.5 * sum_{i where x_i=1} (sum_{j in adj[i]} x_j)
        # This is O(N * d)
        if not x_vec.any():
            return 0.0
        idxs = np.where(x_vec)[0]
        # For each idx, count neighbors that are also in x_vec
        # We can do this by checking adj[idx] against x_vec
        # Vectorized: 
        # For each i in idxs, sum(x_vec[adj[i]])
        # This is a loop in Python, which is slow if N is large.
        # But number of 1s is small (~100).
        # So it's fast.
        conf_count = 0
        for i in idxs:
            conf_count += x_vec[adj[i]].sum()
        # Each conflict is counted twice (i,j and j,i)
        conf_count //= 2
        return -len(idxs) + lambda_pen * conf_count

    current_energy = calc_energy(current_x)
    best_energy = current_energy
    best_x = current_x.copy()
    
    T = T_start
    # Number of steps per temperature
    # N=5280. Let's do 1000 steps per T.
    # Total steps ~ 1000 * log(T_start/T_end)/log(1/alpha)
    # log(1000)/log(1.001) ~ 6.9/0.001 ~ 6900 temps.
    # 6900 * 1000 = 6.9M steps.
    # Each step is O(d) = O(56).
    # 6.9M * 56 ~ 387M ops.
    # In Python, this might be too slow if not optimized.
    # Let's reduce steps or optimize.
    
    # Optimization: Precompute adj arrays as lists of lists?
    # adj is list of numpy arrays.
    
    # Let's try a faster loop.
    # We can use a list of active vertices to update?
    # Or just flip random vertices.
    
    steps_per_T = 500
    # Adjust alpha to finish in time
    # Time per step: ~10-20 us (numpy overhead)
    # 500 steps * 6900 T = 3.45M steps.
    # 3.45M * 20us = 69 seconds.
    # This fits in 140s.
    
    while T > T_end:
        for _ in range(steps_per_T):
            if time.time() > deadline - 10:
                break
            
            # Pick random vertex
            i = np.random.randint(n)
            
            # Calculate delta E
            # If x[i] is 0 -> 1:
            #   delta = -1 + lambda * (num neighbors in set)
            # If x[i] is 1 -> 0:
            #   delta = 1 - lambda * (num neighbors in set)
            
            neighbors_in_set = current_x[adj[i]].sum()
            
            if not current_x[i]:
                delta_E = -1.0 + lambda_pen * neighbors_in_set
            else:
                delta_E = 1.0 - lambda_pen * neighbors_in_set
            
            # Accept if delta_E < 0 or with prob exp(-delta_E/T)
            if delta_E < 0:
                current_x[i] = not current_x[i]
                current_energy += delta_E
            else:
                # Metropolis criterion
                # If T is very small, exp(-delta/T) is 0
                # Avoid overflow
                if T > 0:
                    # delta_E >= 0
                    # If delta_E is large, exp is 0
                    if delta_E < 100 * T: # Approximation
                        prob = np.exp(-delta_E / T)
                        if np.random.random() < prob:
                            current_x[i] = not current_x[i]
                            current_energy += delta_E
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = current_x.copy()
            
        T *= alpha
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # Final local search on best_x
    # Try to add vertices that don't conflict
    # And remove vertices if it allows adding more?
    # Simple add-back
    in_set = best_x.copy()
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        # Find candidates not in set
        not_in_set = ~in_set
        # For each candidate, check if it conflicts with in_set
        # We can do this by checking adj
        added = False
        # Iterate through candidates
        # To speed up, we can just check a random subset or all
        # N=5280, checking all is 5280 * 56 ~ 300k ops. Fast.
        candidates = np.where(not_in_set)[0]
        for v in candidates:
            if in_set[adj[v]].any():
                continue
            # No conflict, add it
            in_set[v] = True
            added = True
        if not added:
            break
            
    result = vectors[in_set]
    return result
# EVOLVE-BLOCK-END
