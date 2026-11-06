import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    random.seed(42)
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    # Precompute: for fast evaluation using indicator vector + convolution
    # For unordered pairs with equal summands allowed:
    # count(t) = (1/2) * (sum_{a in A} I(a+b=t) + I(2a=t))
    # = (1/2) * (conv(A,A)[t] + I(2a=t for a in A))
    # But we need integer counts. Let's use:
    # For each t, count = number of (a,b) with a<=b, a,b in A, a+b=t
    # Using FFT/convolution: full_conv = np.convolve(ind, ind) gives ordered counts
    # unordered_count(t) = (full_conv[t] + (1 if t even and t//2 in A else 0)) // 2
    
    def evaluate_fast(A_set):
        """Fast evaluation using indicator vector and convolution.
        A_set is a set of integers. Returns (peak, reps_array)."""
        # Create indicator vector of length 257 (indices 0..256)
        ind = np.zeros(257, dtype=np.int32)
        for a in A_set:
            ind[a] = 1
        
        # Convolution: conv[t] = number of ordered pairs (a,b) in A x A with a+b=t
        conv = np.convolve(ind, ind, mode='full')  # length 513, index t gives count for sum=t
        
        # Extract target range [128, 256]
        # conv index t corresponds to sum t
        target_conv = conv[target_low:target_high+1]  # length 129
        
        # Convert ordered to unordered:
        # unordered_count(t) = (ordered_count(t) + diagonal(t)) // 2
        # where diagonal(t) = 1 if t is even and t//2 in A, else 0
        # But ordered_count(t) already includes (a,a) pairs once when a+a=t
        # Actually: ordered_count(t) = sum_{a in A} I(a in A and t-a in A)
        # This counts (a,b) and (b,a) separately for a!=b, and (a,a) once
        # Unordered count = for a<b: 1 per pair, for a=b: 1
        # = (ordered_count(t) - diag(t)) / 2 + diag(t)
        # = (ordered_count(t) + diag(t)) / 2
        # where diag(t) = 1 if t even and t//2 in A
        
        # Compute diagonal contributions
        targets = np.arange(target_low, target_high+1)
        diag = np.zeros(num_targets, dtype=np.int32)
        for i, t in enumerate(targets):
            if t % 2 == 0:
                half = t // 2
                if half in A_set:
                    diag[i] = 1
        
        # Unordered counts
        reps = (target_conv + diag) // 2
        
        # Check coverage
        if np.any(reps == 0):
            return float('inf'), reps
        
        peak = int(np.max(reps))
        return peak, reps
    
    best_A = None
    best_peak = float('inf')
    
    # Strategy: Start with structured initial candidates, then local search
    # Key: A should be "spread out" to cover [128,256] efficiently
    
    # Initial structured candidates
    # Idea: A = {0, 1, ..., k} union {m, m+1, ..., n} for some k, m
    # Or: evenly spaced points
    
    # Try different sizes with multiple restarts
    for size in [30, 35, 40, 45, 50, 55, 60, 65]:
        if time.time() > deadline:
            break
        
        max_restarts = 8 if size <= 45 else 4
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Structured: low cluster + high cluster
                half_size = size // 2
                A_list = list(range(half_size)) + list(range(n - (size - half_size), n + 1))
                A_list = sorted(set(A_list))
            elif restart == 1:
                # Evenly spaced
                A_list = sorted(set(int(x) for x in np.linspace(0, n, size)))
            elif restart == 2:
                # Two clusters
                cluster1_size = size // 2
                cluster2_size = size - cluster1_size
                c1_start = random.randint(0, max(0, 20 - cluster1_size + 1))
                # Ensure c2 cluster fits in [0, 128]
                max_c2_start = 128 - cluster2_size + 1
                min_c2_start = 100
                if max_c2_start < min_c2_start:
                    c2_start = max_c2_start
                else:
                    c2_start = random.randint(min_c2_start, max_c2_start)
                A_list = list(range(c1_start, c1_start + cluster1_size)) + \
                         list(range(c2_start, c2_start + cluster2_size))
                A_list = sorted(set(a for a in A_list if 0 <= a <= n))
            else:
                # Random
                A_list = sorted(random.sample(range(0, n + 1), size))
            
            # Ensure exactly 'size' elements
            while len(A_list) < size:
                candidates = [x for x in range(n+1) if x not in set(A_list)]
                if not candidates:
                    break
                A_list.append(random.choice(candidates))
                A_list = sorted(A_list)
            
            A_list = sorted(A_list[:size])
            A_set = set(A_list)
            
            peak, reps = evaluate_fast(A_set)
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_list)
            
            # Local search with swap operations
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                # Random swap
                old_val = random.choice(A_list)
                new_val = random.randint(0, n)
                
                if new_val == old_val or new_val in A_set:
                    continue
                
                # Try the swap
                A_new_set = A_set.copy()
                A_new_set.remove(old_val)
                A_new_set.add(new_val)
                
                new_peak, _ = evaluate_fast(A_new_set)
                
                if new_peak < peak:
                    peak = new_peak
                    A_set = A_new_set
                    A_list = sorted(A_set)
                    if peak < best_peak:
                        best_peak = peak
                        best_A = list(A_list)
                elif new_peak == peak and random.random() < 0.1:
                    A_set = A_new_set
                    A_list = sorted(A_set)
    
    # Refinement phase: try to improve best_A further
    if best_A is not None and time.time() < deadline - 5:
        A_set = set(best_A)
        peak, reps = evaluate_fast(A_set)
        
        # Try targeted improvements: for each target with high rep count, 
        # try to remove an element that contributes to it
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Find target with maximum rep count
            if peak == float('inf'):
                break
            max_idx = int(np.argmax(reps))
            max_target = target_low + max_idx
            
            # Find elements that contribute to this target
            contributing = []
            for a in A_set:
                b = max_target - a
                if a <= b and b in A_set:
                    contributing.append(a)
            
            if contributing:
                # Try removing a random contributing element
                old_val = random.choice(contributing)
                new_val = random.randint(0, n)
                if new_val != old_val and new_val not in A_set:
                    A_new_set = A_set.copy()
                    A_new_set.remove(old_val)
                    A_new_set.add(new_val)
                    new_peak, _ = evaluate_fast(A_new_set)
                    if new_peak <= peak:
                        peak = new_peak
                        A_set = A_new_set
                        if peak < best_peak:
                            best_peak = peak
                            best_A = sorted(A_set)
    
    # Try to reduce set size
    if best_A is not None and time.time() < deadline - 3:
        A_set = set(best_A)
        peak, _ = evaluate_fast(A_set)
        
        for attempt in range(300):
            if time.time() > deadline:
                break
            if len(A_set) <= 25:
                break
            
            # Try removing a random element
            trial_set = A_set.copy()
            elem_to_remove = random.choice(list(trial_set))
            trial_set.remove(elem_to_remove)
            
            new_peak, _ = evaluate_fast(trial_set)
            if new_peak <= peak:
                peak = new_peak
                A_set = trial_set
                if peak < best_peak:
                    best_peak = peak
                    best_A = sorted(A_set)
    
    if best_A is None:
        # Fallback: guaranteed valid set
        return list(range(64, 129))
    
    return best_A
