"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute initial scores based on structural heuristics (degree, neighborhood density).
    2. Use Simulated Annealing to perturb the scores and find a better ordering for the greedy decoder.
    3. The objective is to maximize the size of the independent set produced by the greedy algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Precompute neighbor lists for fast simulation
    # Using a list of arrays for faster access
    neighbor_lists = [np.where(adj_bool[i])[0] for i in range(n)]
    
    # Initial scores: Low degree is generally good for MIS (Luby's rule intuition)
    # But we want to optimize the specific ordering.
    # Start with a score that favors low degree, but randomize slightly to break ties or explore.
    # Using -degree as a baseline is a strong heuristic.
    scores = -degree.copy()
    
    # Helper to simulate greedy MIS given scores
    def simulate_greedy_fast(scores):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        # Get vertices in order of descending score
        # We need to be careful with ties. Stable sort helps reproducibility but might get stuck.
        # Adding a tiny random noise to scores can help break ties dynamically in SA, 
        # but for evaluation we just sort.
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            neighbors = neighbor_lists[v]
            if len(neighbors) > 0:
                available[neighbors] = False
            
            # Early exit
            if not available.any():
                break
        
        return selected

    # Initial evaluation
    best_size = simulate_greedy_fast(scores)
    best_scores = scores.copy()
    current_scores = scores.copy()
    
    # Simulated Annealing Parameters
    # We have ~110 seconds. 
    # n=200. Simulation is O(n log n + n * avg_deg). 
    # Let's estimate speed.
    
    # Adaptive temperature schedule
    T_start = 1.0
    T_end = 0.01
    # Number of iterations depends on time.
    # Let's run as many iterations as possible within the time budget.
    
    # To make SA work, we need to perturb scores.
    # If we just add Gaussian noise, we might drift away from the -degree heuristic.
    # Better: Perturb specific vertices or small sets.
    # Or: Re-score a subset of vertices using random linear combinations of features.
    
    # Features:
    # 1. Degree
    # 2. Neighbor Degree Sum
    # 3. Random noise (to escape local minima)
    
    neighbor_degree_sum = (adj_bool @ degree).astype(float)
    
    # Normalize features for mixing
    max_deg = degree.max() if degree.max() > 0 else 1.0
    max_nds = neighbor_degree_sum.max() if neighbor_degree_sum.max() > 0 else 1.0
    
    # We will maintain a "base" score and a "noise" score.
    # base = -degree / max_deg
    # noise = random vector
    # current_score = base + alpha * noise
    
    # Actually, a simpler approach for SA on orderings:
    # Swap positions of two vertices in the sorted order? No, the decoder sorts by score.
    # So we are optimizing the score vector.
    
    # Let's try optimizing the score vector directly.
    # Perturbation: Add Gaussian noise to a random subset of vertices.
    
    # Track best
    best_score_val = best_size
    
    # Time-based loop
    # Estimate time per iteration.
    # First iteration took some time.
    t_iter_start = time.time()
    
    # Let's do a quick benchmark of 10 iterations to estimate cost
    test_scores = current_scores + np.random.normal(0, 0.1, n)
    for _ in range(10):
        simulate_greedy_fast(test_scores)
    t_iter_end = time.time()
    time_per_iter = (t_iter_end - t_iter_start) / 10.0
    
    # Calculate how many iterations we can do
    # Reserve 10s for final cleanup/return
    available_time = deadline - time.time()
    max_iters = int(available_time / (time_per_iter * 1.1)) # 10% buffer
    
    # Cap max_iters to avoid running forever if it's too fast, but we want to use the time.
    # SA usually benefits from many iterations.
    
    # Temperature schedule
    # We want to cool down over max_iters.
    # T(t) = T_start * (T_end / T_start) ** (t / max_iters)
    
    # Current temperature
    T = T_start
    
    # To allow escape from local minima, we might want to restart or increase noise occasionally.
    # Let's stick to standard SA first.
    
    # Features for potential re-initialization or guided search
    # If we get stuck, we can re-initialize with a different heuristic.
    
    # Heuristics to mix:
    # H1: -degree
    # H2: -neighbor_degree_sum
    # H3: -degree - 0.5*neighbor_degree_sum
    # H4: Random
    
    # Let's use a weighted combination of features as the "base" and optimize weights?
    # That's a different optimization problem (weight optimization).
    # The current approach optimizes the score vector directly.
    
    # Let's stick to direct score vector optimization with Gaussian perturbations.
    # To prevent drift, we can occasionally pull back towards the -degree heuristic.
    
    # Parameters
    sigma = 0.1  # Initial noise std dev
    sigma_decay = 0.999
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate candidate scores
        # Perturb current scores
        # To avoid losing the good structure of -degree, we perturb around current, 
        # but also have a small chance to "reset" a vertex to its -degree value + noise.
        
        # Simple Gaussian perturbation
        noise = np.random.normal(0, sigma, n)
        candidate_scores = current_scores + noise
        
        # Evaluate candidate
        cand_size = simulate_greedy_fast(candidate_scores)
        
        # Acceptance criterion
        if cand_size > best_score_val:
            # Better than best
            best_score_val = cand_size
            best_scores = candidate_scores.copy()
            current_scores = candidate_scores
            # Maybe reduce sigma slightly when improving?
            sigma *= 0.999
        else:
            # Worse or equal
            # Calculate delta
            delta = cand_size - best_score_val
            # Note: delta is negative or zero
            # Accept with probability exp(delta / T)
            # Since delta is integer (size), and T is float.
            # If T is small, exp(delta/T) is tiny for negative delta.
            if np.random.random() < np.exp(delta / T):
                current_scores = candidate_scores
        
        # Update temperature
        # Geometric cooling
        T = T_start * (T_end / T_start) ** ((i + 1) / max_iters)
        
        # Optional: Restart mechanism
        # If stuck for a long time, re-initialize with a different heuristic?
        # Let's add a simple restart if no improvement for a while.
        # We can track 'steps_since_improvement'.
        
        # For simplicity and robustness, let's just let SA run.
        # However, direct score optimization can drift far from sensible values.
        # Let's add a "pull back" force.
        # Every 1000 steps, blend current_scores with -degree heuristic.
        if i % 1000 == 0 and i > 0:
            # Blend 10% towards -degree
            # -degree is the "anchor"
            anchor = -degree
            current_scores = 0.9 * current_scores + 0.1 * anchor
            # Re-evaluate to ensure we didn't break it? No, just update current.
            # The next perturbation will be based on this.
            # We should update best if this blend is better?
            # Let's evaluate the blend.
            blend_size = simulate_greedy_fast(current_scores)
            if blend_size > best_score_val:
                best_score_val = blend_size
                best_scores = current_scores.copy()
    
    return best_scores

# EVOLVE-BLOCK-END
