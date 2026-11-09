import numpy as np
import time

def heuristics(prizes, weights):
    # 1. Input processing
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are (5, 100)
    if prizes.ndim == 1:
        # Fallback if 1D, though contract says (5, 100)
        P = prizes
        w = weights.reshape(5, -1) if weights.size == prizes.size * 5 else weights
    else:
        P = prizes.sum(axis=0) # Total prize per item (100,)
        w = weights             # (5, 100)
        
    n = P.shape[0]
    m = w.shape[0]
    capacity = 25.0
    
    # 2. Time Guard
    t_start = time.time()
    deadline = t_start + 280.0  # 20s safety margin
    
    def check_time():
        return time.time() > deadline
        
    # 3. Feature Engineering
    # We want to generate a diverse set of candidate heuristics.
    # Base features:
    # P: (100,) total prize
    # w: (5, 100) weights
    # wsum: (100,) sum of weights
    # wmax: (100,) max weight
    # wmin: (100,) min weight
    # wprod: (100,) product of weights (proxy for volume)
    
    wsum = w.sum(axis=0)
    wmax = w.max(axis=0)
    wmin = w.min(axis=0)
    # Log volume to avoid underflow/overflow and emphasize large items
    wlogvol = np.log(w + 1e-9).sum(axis=0)
    
    # Normalized features for stability
    P_norm = P / (P.max() + 1e-9)
    wsum_norm = wsum / (wsum.max() + 1e-9)
    wmax_norm = wmax / (wmax.max() + 1e-9)
    
    # 4. Candidate Generation
    # Instead of a few lambdas, we generate a matrix of candidates.
    # Each column is a candidate heuristic vector (100,).
    # We will use a mix of:
    # a) Simple ratios: P / (1 + k * w_metric)
    # b) Exponential decay: P * exp(-k * w_metric)
    # c) Power laws: (P / (1 + w_metric))^k
    # d) Linear combinations: alpha * P_norm + beta * (1 - wsum_norm)
    
    candidates = []
    
    # Group A: Ratio based
    for k in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        candidates.append(P / (1.0 + k * wsum))
        candidates.append(P / (1.0 + k * wmax))
        
    # Group B: Exponential
    for k in [0.5, 1.0, 2.0, 5.0]:
        candidates.append(P * np.exp(-k * wsum_norm))
        candidates.append(P * np.exp(-k * wmax_norm))
        
    # Group C: Power
    for p in [0.5, 1.0, 2.0, 3.0]:
        candidates.append((P / (1.0 + wsum)) ** p)
        candidates.append((P / (1.0 + wmax)) ** p)
        
    # Group D: Linear Mixes (Greedy heuristic approximations)
    # High prize, low weight
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        candidates.append(alpha * P_norm + (1.0 - wsum_norm))
        candidates.append(alpha * P_norm + (1.0 - wmax_norm))
        
    # Group E: Entropy/Spread based
    # Items that are "balanced" across constraints might be safer?
    # Or items that are "specialized"?
    # Let's try penalizing variance of weights
    w_var = np.var(w, axis=0)
    for k in [0.5, 1.0]:
        candidates.append(P * np.exp(-k * w_var))
        
    # Group F: Rank based (Robust to outliers)
    P_rank = np.argsort(np.argsort(P)).astype(float) / n
    wsum_rank = np.argsort(np.argsort(wsum)).astype(float) / n
    for alpha in [1.0, 2.0, 5.0]:
        candidates.append(alpha * P_rank + (1.0 - wsum_rank))
        
    # Ensure all candidates are positive and finite
    processed_candidates = []
    for c in candidates:
        c = np.asarray(c, dtype=float)
        # Sanitize: non-finite or <= 0 become 1e-9
        c = np.where(np.isfinite(c) & (c > 0), c, 1e-9)
        c = np.clip(c, 1e-9, 1e6)
        processed_candidates.append(c)
        
    candidates = np.array(processed_candidates).T # Shape: (n, num_candidates)
    
    # 5. Fast Vectorized ACO Simulation
    # We need to simulate the ACO for all candidates in parallel.
    # State:
    # ph: (n,) pheromone, starts at 1.0
    # h: (n, C) heuristics for C candidates
    # We run 30 iterations, 20 ants.
    # To simulate 20 ants in parallel for C candidates:
    # This is tricky because each ant's selection depends on previous selections in that ant.
    # However, we can simulate the *average* behavior or just run a single "representative" ant per candidate?
    # No, the scoring is "mean over instances of the highest total prize found by any ant".
    # The ACO dynamics are stochastic.
    # Running 20 ants * 30 iters * C candidates * 100 items in pure Python is too slow.
    # We need a vectorized approximation or a very fast loop.
    
    # Strategy:
    # Since we have a time limit, we can't run full stochastic ACO for 50+ candidates.
    # However, the heuristic is fixed. The pheromone update depends on the *best* prize of the iteration.
    # If we assume the ants pick the best available item (greedy limit of high pheromone/heuristic),
    # we can approximate the colony's performance by simulating a few greedy-like paths.
    # BUT, the task says "probability proportional to...".
    # Let's try to vectorize the *selection* step for all candidates at once for a single ant?
    # No, ants are independent.
    
    # Alternative:
    # Use a small number of iterations/ants for evaluation, but vectorize over candidates.
    # Let's simulate 10 iterations, 5 ants.
    # For each ant, we select items sequentially.
    # We can vectorize the *probability calculation* across all candidates for a given ant state.
    # But the ant state (used items, remaining capacity) differs per candidate?
    # No, the state depends on the *items selected*. The items selected depend on the heuristic.
    # So states diverge immediately.
    
    # Therefore, we cannot fully vectorize the sequential selection across candidates without massive memory overhead or complex masking.
    
    # Revised Strategy:
    # We will evaluate candidates in batches.
    # For each batch, we run a *fast* ACO simulation.
    # To make it fast:
    # 1. Use Numba? No, standard lib only.
    # 2. Use NumPy vectorization where possible.
    # 3. Reduce simulation fidelity: 10 iterations, 5 ants.
    # 4. Early stopping: if a candidate is clearly worse, stop?
    
    # Let's optimize the simulation loop itself.
    # The bottleneck is `rng.choice` and the `while` loop for each ant.
    
    # We will implement a simplified ACO simulation that is vectorized over the *items* for a single ant,
    # and loop over candidates. To save time, we will only evaluate a subset of candidates
    # or use a very fast heuristic evaluation.
    
    # Actually, let's look at the structure.
    # If we run the simulation for Candidate 1, then Candidate 2, etc.
    # 50 candidates * 10 iters * 5 ants * (avg 10 items * 100 items check) = 50 * 10 * 5 * 1000 = 2.5M ops.
    # This is doable in Python if optimized, but `rng.choice` is slow.
    
    # Let's try to vectorize the *evaluation* of all candidates for a single "virtual" ant?
    # No, the state diverges.
    
    # Let's stick to a fast loop but reduce the search space.
    # We will use a tournament selection.
    # Evaluate 10 candidates at a time? No, just loop.
    
    # Optimization:
    # Precompute `fits` logic?
    # `fits = np.all(w <= rem[:, None], axis=0)` is O(5*100) per step.
    # 10 iters * 5 ants * 10 steps * 500 ops = 250k ops per candidate.
    # 50 candidates = 12.5M ops.
    # Python overhead is ~100ns per op? 1.25s. This is very fast!
    # The main cost is `rng.choice` and list manipulations.
    
    def simulate_aco_fast(h, P, w, n, m, cap, iters=10, ants=5, seed=42):
        rng = np.random.default_rng(seed)
        ph = np.ones(n)
        global_best = 0.0
        
        # Pre-allocate for speed?
        for _ in range(iters):
            iter_best = 0.0
            for _ in range(ants):
                rem = np.full(m, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                # Greedy-like selection loop
                # To speed up, we can break early if no items fit
                while True:
                    # Vectorized fit check
                    # w is (m, n), rem is (m,)
                    # w <= rem[:, None] -> (m, n) bool
                    fits = np.all(w <= rem[:, None], axis=0)
                    eligible = fits & ~used
                    
                    if not np.any(eligible):
                        break
                    
                    # Probabilities
                    p = ph * h
                    p = np.where(eligible, p, 0.0)
                    s = p.sum()
                    if s <= 0:
                        break
                    
                    # Selection
                    # rng.choice is slow. 
                    # We can use cumulative sum and searchsorted for faster selection if needed,
                    # but rng.choice is okay for n=100.
                    idx = rng.choice(n, p=p / s)
                    
                    used[idx] = True
                    rem -= w[:, idx]
                    total += P[idx]
                    
                    if check_time():
                        return -1.0 # Timeout indicator
                
            if total > iter_best:
                iter_best = total
            if total > global_best:
                global_best = total
                
            # Pheromone update
            ph = ph * 0.9 + iter_best / 10.0
            
        return global_best

    # Evaluate candidates
    best_score = -1.0
    best_h = None
    
    # Shuffle candidates to ensure we don't just pick the first one if they are similar
    # Or just iterate.
    for i in range(candidates.shape[1]):
        if check_time():
            break
            
        h = candidates[:, i]
        score = simulate_aco_fast(h, P, w, n, m, capacity, iters=10, ants=5, seed=123+i)
        
        if score > best_score:
            best_score = score
            best_h = h
            
    if best_h is None:
        # Fallback to a simple heuristic if all failed or timed out immediately
        best_h = P / (1.0 + wsum)
        best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
        
    return best_h
