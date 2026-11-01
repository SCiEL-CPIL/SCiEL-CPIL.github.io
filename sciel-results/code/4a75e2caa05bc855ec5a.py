import numpy as np
import time

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # Compute autocorrelation
    # np.correlate(h, h, 'full') gives lags from -(N-1) to N-1.
    # Index N-1 corresponds to lag 0.
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_h = None
    best_score = float('inf')
    
    # Initial random guesses
    num_initial = 50
    for i in range(num_initial):
        if time.time() > deadline - 5:
            break
        h = np.random.choice([-1, 1], size=N)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        return np.ones(N, dtype=int)

    h = best_h
    current_score = best_score
    
    # Phase 1: Greedy Hill Climbing (Fast local refinement)
    # We do a few passes of greedy flipping to settle into a local minimum.
    # This is fast and establishes a good baseline.
    for pass_idx in range(5):
        if time.time() > deadline - 20:
            break
        improved = False
        # Shuffle order to avoid bias
        indices = np.random.permutation(N)
        for j in indices:
            if time.time() > deadline - 20:
                break
            old_val = h[j]
            h[j] = -h[j]
            new_score = _calc_score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                h[j] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        if not improved:
            # Local minimum reached, break early to save time for SA
            break

    # Phase 2: Simulated Annealing (Global escape mechanism)
    # Structural change: Instead of random restarts or perturbations that might be too small or too large,
    # use SA with a cooling schedule to systematically explore the landscape.
    
    # Initialize SA state from current best
    h_sa = best_h.copy()
    score_sa = best_score
    
    # SA Parameters
    # Temperature T controls the probability of accepting worse solutions.
    # P(accept) = exp(-delta_E / T)
    # We want to start with high T (accept many worse moves) and cool down.
    
    # Estimate typical delta_E. 
    # Flipping a bit changes autocorrelation values.
    # Let's just pick a heuristic start temperature.
    # Score is around 0.01 - 0.05. Delta score can be small.
    # Let's try T_start = 0.01, T_end = 1e-6
    
    T_start = 0.01
    T_end = 1e-6
    
    # Number of iterations for SA
    # We have ~90 seconds left.
    # Each iteration: 1 flip, 1 score calc.
    # Score calc ~ 1ms. So ~1000 iters/sec.
    # 90s -> 90,000 iters.
    
    # We will run SA until time limit.
    # Cooling schedule: Geometric cooling T = T_start * (T_end/T_start)^(i/total_iters)
    # But total_iters is unknown. Let's use time-based cooling or just a large fixed number.
    # Let's use a fixed number of iterations and adjust T based on progress.
    
    # To be safe, let's estimate iterations we can do in remaining time.
    # 1 iteration ~ 1ms + overhead. Let's say 2ms.
    # 90s / 2ms = 45,000 iters.
    
    # Let's just run a loop until deadline.
    # We need a temperature schedule.
    # Let's use T = T_start * exp(-alpha * t_elapsed)
    # Or simpler: T = T_start * (1 - t_elapsed / t_total)^2 ?
    
    # Let's use a simple geometric cooling based on iteration count, assuming we do ~50k iterations.
    # If we finish early, T will be higher than ideal, which is fine (more exploration).
    # If we run long, T will be lower.
    
    total_planned_iters = 50000
    i = 0
    current_T = T_start
    
    # To avoid getting stuck in a deep local min immediately, we can randomize the start slightly?
    # No, starting from the greedy result is good.
    
    while time.time() < deadline - 5:
        # Update temperature
        # If we haven't done enough iterations yet, keep T high?
        # Geometric cooling: T_i = T_start * (T_end/T_start)^(i / total_planned_iters)
        if i >= total_planned_iters:
            # If we exceed planned, keep T at T_end or lower?
            # Let's just keep T at T_end to act like greedy hill climbing with occasional noise.
            current_T = T_end
        else:
            current_T = T_start * (T_end / T_start) ** (i / total_planned_iters)
        
        # Pick a random index to flip
        j = np.random.randint(N)
        
        old_val = h_sa[j]
        h_sa[j] = -h_sa[j]
        
        new_score = _calc_score(h_sa)
        delta = new_score - score_sa
        
        if delta < 0:
            # Improvement
            score_sa = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h_sa.copy()
        else:
            # Worse, accept with probability exp(-delta / T)
            if current_T > 0:
                prob = np.exp(-delta / current_T)
                if np.random.random() < prob:
                    score_sa = new_score
                else:
                    # Revert
                    h_sa[j] = old_val
            else:
                # T is effectively 0, revert
                h_sa[j] = old_val
        
        i += 1
        # Throttle if too fast? No, np.correlate is fast.
        # But Python loop overhead might be significant.
        # 50k iterations in 90s is < 1000/s. Should be fine.
        
        # Optional: If we found a very good score, we might want to cool faster?
        # No, standard schedule is fine.

    return best_h
