import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    D = 11
    W = 4  # Weight of vectors
    
    # 1. Generate all supports (4-subsets)
    supports = list(combinations(range(D), W))
    N_sup = len(supports) # 330
    
    # 2. Build incompatibility graph
    # incompat[i, j] is True if supports[i] and supports[j] intersect in >= 3 elements
    # We want an Independent Set in this graph.
    # This is equivalent to a Clique in the complement graph.
    
    sup_sets = [set(s) for s in supports]
    
    # Precompute incompatibility matrix
    # A[i, j] = 1 if incompatible
    A = np.zeros((N_sup, N_sup), dtype=np.int8)
    
    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            inter_size = len(sup_sets[i] & sup_sets[j])
            if inter_size >= 3:
                A[i, j] = 1
                A[j, i] = 1

    # 3. Simulated Annealing to find Max Independent Set
    # Objective: Maximize sum(x) subject to no edges in A among selected nodes.
    # Energy: E = -sum(x) + lambda * sum(A[i,j] x[i] x[j])
    # We minimize E.
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_x = None
    
    # Parameters for SA
    lambda_penalty = 100.0  # High penalty to enforce independence
    T_start = 10.0
    T_end = 0.01
    n_iterations = 5000  # Steps per run
    n_runs = 20         # Number of independent SA runs
    
    for run in range(n_runs):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize x
        # Start with a random sparse set or greedy
        # Greedy init: pick random order, add if no conflict
        order = rng.permutation(N_sup)
        x = np.zeros(N_sup, dtype=np.int8)
        for idx in order:
            # Check conflicts
            # Conflicts are A[idx] & x
            if not np.any(A[idx] & x):
                x[idx] = 1
        
        current_size = np.sum(x)
        
        # SA Loop
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
        
        for step in range(n_iterations):
            if time.time() > deadline - 10.0:
                break
            
            # Pick random node
            i = rng.integers(N_sup)
            
            # Calculate Delta E for flipping x[i]
            # Current contribution of i to penalty: lambda * sum(A[i, j] x[j])
            # Let conflict_count = np.sum(A[i] * x)
            conflict_count = np.sum(A[i] * x)
            
            if x[i] == 0:
                # Flip 0 -> 1
                # Delta E = -1 (gain from sum) + lambda * conflict_count (new penalty)
                delta_E = -1.0 + lambda_penalty * conflict_count
            else:
                # Flip 1 -> 0
                # Delta E = +1 (loss from sum) - lambda * conflict_count (penalty removed)
                delta_E = 1.0 - lambda_penalty * conflict_count
            
            # Acceptance
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                x[i] = 1 - x[i]
            
            T *= cooling_rate
        
        # Cleanup: Ensure we have a valid Independent Set
        # Remove nodes involved in conflicts
        # Greedy removal: while conflicts exist, remove one
        changed = True
        while changed:
            changed = False
            # Find all selected nodes
            selected_indices = np.where(x == 1)[0]
            if len(selected_indices) == 0:
                break
            
            # Check for conflicts
            # For each selected node, count conflicts with other selected nodes
            # We can just iterate and remove the first conflict found
            # To be efficient, find a node with high degree in the induced subgraph
            # Simpler: iterate through selected, if it has conflict, remove it.
            # This might be slow if many conflicts, but N=330 is small.
            
            # More efficient cleanup:
            # Create subgraph of selected nodes
            # If any edge exists, remove one endpoint.
            
            # Let's just check if the set is valid.
            # If not, pick a random selected node involved in a conflict and remove it.
            
            # Get conflict counts for selected nodes
            # conflict_counts[i] = sum(A[i] * x) for i in selected
            # If any > 0, we have a conflict.
            
            # Vectorized check:
            # We want to know if there are any edges in A[x==1, x==1]
            # This is equivalent to checking if (A * x) has any non-zero entries where x is 1.
            
            # Let's just do a simple loop for cleanup, it's fast enough for N=330
            for idx in selected_indices:
                if x[idx] == 1:
                    # Check if idx conflicts with any other selected node
                    # A[idx] is row of incompatibilities
                    # We need A[idx][j] == 1 and x[j] == 1 for j != idx
                    # Since A is symmetric and A[idx][idx]=0, sum(A[idx] * x) counts conflicts
                    if np.sum(A[idx] * x) > 0:
                        x[idx] = 0
                        changed = True
                        break # Restart check after removal
        
        # Fill: Try to add more nodes
        # Repeat until no more can be added
        changed = True
        while changed:
            changed = False
            for i in range(N_sup):
                if x[i] == 0:
                    if np.sum(A[i] * x) == 0:
                        x[i] = 1
                        changed = True
        
        size = np.sum(x)
        if size > best_size:
            best_size = size
            best_x = x.copy()
            
        # Optional: Local Search / Hill Climbing on the best solution found
        # Try swapping pairs? Or just simple greedy removal/addition is often enough.
        
    if best_x is None:
        return np.array([[]], dtype=np.int8)
        
    # 4. Generate Vectors
    vectors = []
    for i in range(N_sup):
        if best_x[i] == 1:
            sup = supports[i]
            # Generate all 16 sign variations
            for signs in product([1, -1], repeat=W):
                v = np.zeros(D, dtype=np.int8)
                for idx, s in zip(sup, signs):
                    v[idx] = s
                vectors.append(v)
                
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
