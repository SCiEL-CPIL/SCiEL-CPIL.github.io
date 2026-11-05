import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Stochastic Game Value Estimation + Simulated Annealing
    1. Estimate each vertex's expected contribution to the independent set size
       via Monte Carlo sampling of random priority orders.
    2. Use the empirical selection probability as the initial score.
    3. Refine with simulated annealing to escape local optima.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    def simulate_greedy(order):
        """Run greedy decoder on a given vertex order, return selected count and selected mask."""
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count
    
    def simulate_greedy_with_selection(order):
        """Run greedy decoder, return (count, selected_mask)."""
        active = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                selected[v] = True
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count, selected
    
    # === Phase 1: Monte Carlo estimation of vertex selection probability ===
    # Sample random priority orders and estimate P(vertex v is selected)
    # This captures non-linear interference: a vertex's value depends on who else
    # is around it and the global structure.
    
    rng = np.random.default_rng(12345)
    
    # Number of MC samples - balance between accuracy and time
    # Each sample is O(n * avg_degree) which is fast for n=200
    mc_samples = 2000
    selection_count = np.zeros(n, dtype=float)
    
    # Use time budget: spend ~30s on MC
    mc_deadline = start_time + 30.0
    samples_done = 0
    
    for s in range(mc_samples):
        if time.time() > mc_deadline:
            break
        
        # Generate a random priority order
        # Use random scores and sort
        random_scores = rng.random(n)
        order = np.argsort(-random_scores, kind='stable')
        
        # Run greedy
        active = np.ones(n, dtype=bool)
        for v in order:
            if active[v]:
                selection_count[v] += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        
        samples_done += 1
    
    # Normalize to get probability
    if samples_done > 0:
        selection_prob = selection_count / samples_done
    else:
        selection_prob = np.ones(n, dtype=float) / n
    
    # Initial score: vertices with higher selection probability should be prioritized
    # But we also want to consider degree as a secondary signal
    deg = adj.sum(axis=1).astype(float)
    
    # Score = selection probability (primary) + small degree term (tie-breaking)
    scores = selection_prob.copy()
    
    # === Phase 2: Simulated Annealing refinement ===
    # Directly optimize the greedy decoder output by perturbing scores
    # with simulated annealing to escape local optima
    
    def simulate_fast(s):
        """Fast greedy simulation using the score ordering."""
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count
    
    # Get baseline
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    current_scores = scores.copy()
    current_val = best_val
    
    # Simulated annealing parameters
    max_time = 115.0  # Safety margin before 120s limit
    T_initial = 0.5
    T_min = 0.001
    cooling_rate = 0.9995  # Slow cooling for thorough search
    
    T = T_initial
    sa_rng = np.random.default_rng(67890)
    
    while time.time() < max_time and T > T_min:
        # Propose a perturbation: add noise to a random subset of scores
        # This allows larger moves than single pair swaps
        num_perturb = sa_rng.integers(1, max(2, n // 10) + 1)
        idx = sa_rng.integers(0, n, size=num_perturb)
        
        # Save original values
        old_vals = current_scores[idx].copy()
        
        # Perturb
        noise = sa_rng.normal(0, T * 2, size=num_perturb)
        current_scores[idx] = old_vals + noise
        
        new_val = simulate_fast(current_scores)
        delta = new_val - current_val
        
        if delta > 0:
            # Accept improvement
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_scores = current_scores.copy()
        else:
            # Accept worse with probability exp(delta / T)
            if delta < 0 and sa_rng.random() < np.exp(delta / T):
                current_val = new_val
            else:
                # Reject: revert
                current_scores[idx] = old_vals
        
        T *= cooling_rate
    
    return best_scores
