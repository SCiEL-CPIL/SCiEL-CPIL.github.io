import numpy as np
import random
import time

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    # Helper to compute representation counts R[k] for k in targets
    # Returns array of length 129 (indices 128..256)
    def get_rep_counts(x):
        conv = np.convolve(x, x, mode='full')
        conv_vals = conv[targets]
        
        even_mask = (targets % 2 == 0)
        mid_indices = targets[even_mask] // 2
        mid_vals = x[mid_indices]
        
        R = np.zeros(len(targets), dtype=np.int32)
        
        odd_mask = ~even_mask
        # conv_vals for odd are even, so /2 is int
        R[odd_mask] = conv_vals[odd_mask] // 2
        
        # R = (conv + mid) / 2
        R[even_mask] = (conv_vals[even_mask] + mid_vals) // 2
        
        return R

    # Greedy Construction
    A = set()
    # Start with a few random points to break symmetry? Or just empty?
    # Let's start with empty and add greedily.
    
    # Precompute contribution of each candidate a to each target t
    # contrib[a][t] = 1 if a + (t-a) is a valid pair? No.
    # When we add 'a', the new pairs are (a, a) and (a, b) for b in A.
    # This dynamic update is complex. 
    # Simpler: Just recompute R for each candidate? 
    # 129 candidates * 129 targets * Conv(129) is too slow for 129 iterations?
    # 129 * 129 * 129 ~ 2M ops. It's fine.
    
    # Actually, we can be smarter.
    # Let's just use a heuristic initial set first to speed up, then refine.
    # Heuristic: Arithmetic progression?
    # Or just random set of size 40-50 and then fix coverage?
    
    # Let's stick to Greedy Max Coverage for initialization.
    # State: x (binary array)
    x = np.zeros(n, dtype=np.int8)
    
    # To speed up greedy, we track current R.
    R = get_rep_counts(x)
    
    # We need to know which targets are uncovered.
    # Uncovered if R[t] == 0.
    
    # Greedy Step:
    # For each candidate a not in A:
    #   Temporarily add a.
    #   Compute new R.
    #   Count how many previously 0 targets become > 0.
    #   Pick a with max new coverage.
    
    # This is O(N * N * Cost(R)).
    # Cost(R) is O(N) with conv.
    # Total O(N^3) ~ 2M. Very fast.
    
    candidates = list(range(n))
    random.shuffle(candidates) # Tie breaker
    
    while True:
        if np.all(R > 0):
            break
            
        best_a = -1
        best_gain = -1
        best_new_R = None
        best_new_peak = 0
        
        for a in candidates:
            if x[a] == 1:
                continue
                
            x[a] = 1
            new_R = get_rep_counts(x)
            
            # Gain: number of targets that were 0 and are now > 0
            gain = np.sum((R == 0) & (new_R > 0))
            
            new_peak = np.max(new_R)
            
            if gain > best_gain or (gain == best_gain and new_peak < best_new_peak):
                best_gain = gain
                best_a = a
                best_new_R = new_R
                best_new_peak = new_peak
            
            x[a] = 0
        
        if best_a == -1:
            break # Should not happen if coverage is possible
            
        x[best_a] = 1
        R = best_new_R
        
    # Now we have a valid set.
    # Refinement: Local Search to minimize peak.
    
    current_peak = np.max(R)
    
    # Hill Climbing
    # We want to minimize peak.
    # Moves: Flip a bit.
    # Accept if:
    #   1. New Peak < Current Peak
    #   2. New Peak == Current Peak AND (New Size < Current Size OR (Random and small delta))
    
    # To avoid getting stuck, we can use a simple simulated annealing 
    # but only on the "valid" states or with a strong penalty for invalidity.
    # Given the greedy start is valid, we can try to remove elements first.
    
    # Phase 1: Try to remove elements.
    # If removing an element keeps all R > 0, do it.
    # This reduces size, which usually reduces peak.
    
    improved = True
    while improved:
        improved = False
        for a in range(n):
            if x[a] == 1:
                x[a] = 0
                new_R = get_rep_counts(x)
                if np.all(new_R > 0):
                    # Check if peak decreased or stayed same
                    new_peak = np.max(new_R)
                    if new_peak <= current_peak:
                        R = new_R
                        current_peak = new_peak
                        improved = True
                    else:
                        x[a] = 1 # Revert
                else:
                    x[a] = 1 # Revert
                    
    # Phase 2: Swap elements to reduce peak.
    # Try swapping a in A with b not in A.
    # If new_peak < current_peak, accept.
    # If new_peak == current_peak, accept with some probability to explore.
    
    # We will run a short simulated annealing phase now, starting from this good valid state.
    # This combines the strengths: good start + optimization.
    
    start_temp = 10.0
    end_temp = 0.1
    steps = 2000 # Fast enough
    
    for i in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (i / steps)
        
        # Pick a random move: flip a bit
        idx = random.randint(0, n - 1)
        x[idx] = 1 - x[idx]
        
        new_R = get_rep_counts(x)
        
        # Check validity
        if np.any(new_R == 0):
            # Invalid state. Strong penalty or revert?
            # Let's revert if invalid to stay in feasible region mostly,
            # or accept with very low prob to jump out?
            # For this problem, staying feasible is better for local search.
            # But sometimes we need to break feasibility to find better peaks?
            # Let's try: if invalid, revert.
            x[idx] = 1 - x[idx]
            continue
            
        new_peak = np.max(new_R)
        new_size = np.sum(x)
        curr_size = np.sum(x)
        
        # Cost function: Primary: Peak. Secondary: Size.
        # We want to minimize Peak.
        # If Peak decreases, accept.
        # If Peak same, prefer smaller size.
        
        if new_peak < current_peak:
            R = new_R
            current_peak = new_peak
        elif new_peak == current_peak:
            if new_size < curr_size:
                R = new_R
                # current_peak stays same
            else:
                # Accept with small probability to allow exploration
                if random.random() < 0.1:
                    R = new_R
                else:
                    x[idx] = 1 - x[idx]
        else:
            # Peak increased.
            # Accept with probability exp(-delta / temp)
            delta = new_peak - current_peak
            if random.random() < np.exp(-delta / temp):
                R = new_R
                current_peak = new_peak
            else:
                x[idx] = 1 - x[idx]
                
    result = np.nonzero(x)[0].tolist()
    
    if not result:
        return [64]
        
    return result
