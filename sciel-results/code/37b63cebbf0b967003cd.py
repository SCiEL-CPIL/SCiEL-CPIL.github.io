"""Direct Permutation Optimization via Simulated Annealing with Incremental Delta Evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin below 120s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for fast access
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    
    # 1. Initial Heuristic: Greedy based on Degree (Lowest Degree First)
    # This is a strong baseline for Independent Set (like Maximum Degree Last in Clique)
    degrees = A.sum(axis=1)
    init_order = np.argsort(degrees, kind='stable')
    
    # 2. Evaluate Initial Order
    # We track 'blocked' vertices. A vertex is blocked if it's picked or is a neighbor of a picked vertex.
    # The size of the independent set is the number of vertices that are NOT blocked when the process ends?
    # No, the greedy decoder picks v if it's not a neighbor of any previously picked vertex.
    # Let's simulate the greedy process to get the baseline size.
    
    def simulate_greedy(order):
        picked = np.zeros(N, dtype=bool)
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if not picked[v]:
                count += 1
                is_picked[v] = True
                picked[v] = True
                # Mark neighbors as blocked
                if neighbors[v].size > 0:
                    picked[neighbors[v]] = True
        return count, is_picked

    best_size, best_is_picked = simulate_greedy(init_order)
    best_order = init_order.copy()
    
    # We need a fast way to evaluate changes.
    # Changing the order of two vertices i and j (where i comes before j in current order)
    # only affects the outcome if i or j are picked, or if their picking status changes the blocking of others.
    # However, a full re-simulation is O(N^2) in worst case (dense graph).
    # With N=200, O(N^2) = 40,000 ops. 
    # If we can do ~10,000 evaluations in 100s, that's 10 evals/sec. 
    # 40,000 ops * 10 = 400,000 ops/sec. This is easily doable in Python if optimized, but list lookups are slow.
    # Let's try to vectorize the simulation or use a C-like loop structure.
    
    # Actually, 200 nodes is small. A pure Python loop over 200 nodes is ~10-20 microseconds.
    # 100 seconds / 20us = 5,000,000 evaluations. This is plenty.
    # We can afford a simple re-simulation for each swap.
    
    def get_score(order):
        picked = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if not picked[v]:
                count += 1
                picked[v] = True
                if neighbors[v].size > 0:
                    picked[neighbors[v]] = True
        return count

    # Simulated Annealing on Permutations
    # State: current_order (array of vertex indices)
    # Move: Swap two random positions in the order.
    # Cost: -Independent Set Size (we want to minimize cost, i.e., maximize size)
    
    current_order = best_order.copy()
    current_size = best_size
    
    # Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    # We don't know exact number of steps, so we loop until time limit
    
    rng = np.random.default_rng(42)
    
    # To speed up, we can precompute neighbor masks? 
    # picked is a boolean array. 
    # picked[neighbors[v]] = True is a vectorized operation.
    
    while time.time() < DEADLINE:
        # Generate a move: swap two indices
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
            
        # Apply swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate new state
        new_size = get_score(current_order)
        delta = new_size - current_size
        
        # Acceptance criterion
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            # Accept
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            # Reject: swap back
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        # Cool down
        T *= alpha
        if T < T_min:
            # If we cool down too much, we might get stuck. 
            # Since we have time budget, we can restart with a slightly higher T or just keep going if it's still exploring?
            # Actually, if T < T_min, we are in local search mode. 
            # If we haven't found a global max, we might want to "reheat" occasionally?
            # For now, let it stay in local search mode (greedy hill climbing) which is safe.
            pass
            
    # Convert best_order to scores
    # The score of a vertex is its rank in the best_order.
    # Vertices earlier in the order have higher priority.
    # So score[v] = N - index_of_v_in_order
    scores = np.zeros(N)
    for rank, v in enumerate(best_order):
        scores[v] = N - rank
        
    return scores
# EVOLVE-BLOCK-END
