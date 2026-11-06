"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Combinatorial Ordering Search (Simulated Annealing on Permutations).
    1. Initialize the vertex order using a strong structural heuristic (Min-Weighted Degree).
    2. Treat the ordering as a discrete state space.
    3. Use Simulated Annealing to swap vertices in the ordering, accepting worse solutions
       with a decaying probability to escape local optima of the greedy decoder.
    4. Return the scores corresponding to the best ordering found (inverse rank).
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    neighbor_degree_sum = adj_bool @ degree
    
    # Fast Greedy Simulator
    # Returns the size of the independent set produced by the greedy decoder 
    # given a specific ordering of vertices.
    def simulate_greedy(order):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            # Mark neighbors as unavailable
            # Using boolean mask update
            available[adj_bool[v]] = False
            
            # Early exit if no vertices available
            if not available.any():
                break
                
        return selected

    # Initial Heuristic: 
    # We want to select low-degree vertices first.
    # To break ties and add diversity, we use a small random component.
    # Score for sorting: Lower is better for selection priority.
    # Let's sort by (degree, random_noise) ascending.
    
    def get_initial_order():
        # Stronger heuristic: degree / (1 + neighbor_degree_sum) might be better?
        # Let's stick to degree + small noise for robustness, but maybe weighted.
        # Actually, "Min-Weighted Degree" is a classic.
        # Let's try: degree + 0.1 * (sum of degrees of neighbors)
        sort_key = degree + 0.1 * neighbor_degree_sum + np.random.normal(0, 0.1, n)
        return np.argsort(sort_key, kind='stable')

    # Main Search Loop: Simulated Annealing on the Permutation
    
    # Initial state
    current_order = get_initial_order()
    current_size = simulate_greedy(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # Annealing Parameters
    # We need to run many iterations. 
    # Each iteration involves a swap and a simulation.
    # Simulation is O(n^2) worst case (n=200 -> 40,000 ops).
    # In Python, this is roughly 1-5 ms.
    # Budget: 118 seconds.
    # We can likely run 10,000 - 50,000 iterations.
    
    # Let's estimate speed first to calibrate T0 and cooling rate.
    # But we must finish within deadline.
    
    T = 1.0  # Initial temperature
    cooling = 0.9995  # Cooling factor
    
    # To ensure we don't waste time, we do a quick calibration.
    t_start = time.time()
    test_order = current_order.copy()
    simulate_greedy(test_order)
    t_end = time.time()
    avg_time_per_iter = (t_end - t_start) / 1.0
    
    # Target iterations: aim to use ~80% of remaining time
    remaining_time = deadline - time.time()
    target_iters = int(remaining_time * 0.8 / max(avg_time_per_iter, 1e-6))
    
    # If target_iters is too high, we might timeout. Cap it.
    # If too low, we might not explore enough.
    # Let's just run until deadline, but with a strict check.
    
    # Better approach: Fixed number of iterations based on time, with early break.
    # Let's set a max number of iterations to prevent infinite loops if time.time() is buggy.
    max_iters = 200000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate Neighbor: Swap two random positions in the order
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        
        if idx1 == idx2:
            continue
            
        new_order = current_order.copy()
        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
        
        new_size = simulate_greedy(new_order)
        
        delta = new_size - current_size
        
        # Acceptance Criteria
        if delta > 0:
            # Always accept improvement
            current_order = new_order
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            # Accept worse move with probability exp(delta / T)
            # Since delta is integer (negative), and T is float.
            # To avoid underflow issues with very negative delta/T, we can check log.
            # But for n=200, sizes are around 100-150. Delta is small.
            if np.random.random() < np.exp(delta / T):
                current_order = new_order
                current_size = new_size
        
        # Cool down
        T *= cooling
        
        # If T is very small, we can stop or restart? 
        # Restarting (quench) can help escape local minima.
        # If T < 0.01, we might be stuck. Let's try a restart mechanism.
        if T < 0.01:
            # Probability to restart from best found or a new random heuristic
            if np.random.random() < 0.1:
                # Restart from best
                current_order = best_order.copy()
                current_size = best_size
                T = 0.5 # Reheat slightly
            elif np.random.random() < 0.1:
                # Restart from new heuristic
                current_order = get_initial_order()
                current_size = simulate_greedy(current_order)
                T = 0.5

    # Convert best_order to scores
    # The greedy decoder sorts by descending score.
    # So, vertex at position 0 in best_order should have the highest score.
    # Vertex at position n-1 should have the lowest score.
    
    scores = np.zeros(n)
    # Assign scores such that np.argsort(-scores) == best_order
    # If best_order[0] is v0, score[v0] should be n-1 (or high)
    # If best_order[1] is v1, score[v1] should be n-2
    # ...
    # best_order[k] gets score n-1-k
    
    # Vectorized assignment:
    # ranks: 0 for best, 1 for second best, etc.
    # We want score = -rank (or n-1-rank)
    # Let's just use -rank to be safe with "descending" order.
    # Actually, the problem says "descending score order".
    # So highest score is picked first.
    # If order is [v0, v1, ...], v0 is picked first.
    # So score[v0] > score[v1] > ...
    
    # Create an array of ranks 0..n-1
    ranks = np.arange(n)
    # The vertex at best_order[i] should have rank i (0 is best)
    # We want score to be inversely proportional to rank.
    # score[best_order[i]] = n - i
    
    scores[best_order] = n - np.arange(n)
    
    return scores
# EVOLVE-BLOCK-END
