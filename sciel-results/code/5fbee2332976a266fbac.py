import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    random.seed(42)
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    def compute_reps_fast(A_arr):
        """Compute representation counts for all targets [128,256] using numpy.
        A_arr is a numpy array of the set elements.
        Returns reps array of length 129 (index 0 = target 128, index 128 = target 256)."""
        # For each pair (i,j) with i<=j, compute sum and count
        # Use broadcasting but that's O(|A|^2) memory which is fine for |A|~50-70
        # Actually, let's do it more efficiently
        m = len(A_arr)
        # Compute all pair sums where i <= j
        # We can use a 2D approach but for m~50-70, m^2/2 ~ 1750-2450 pairs, very fast
        reps = np.zeros(num_targets, dtype=np.int32)
        for i in range(m):
            for j in range(i, m):
                s = A_arr[i] + A_arr[j]
                if target_low <= s <= target_high:
                    reps[s - target_low] += 1
        return reps
    
    def get_peak(reps):
        """Get peak multiplicity, return inf if any target uncovered."""
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    def evaluate(A_list):
        """Evaluate a set given as a sorted list. Returns (peak, reps)."""
        A_arr = np.array(A_list, dtype=np.int32)
        reps = compute_reps_fast(A_arr)
        peak = get_peak(reps)
        return peak, reps
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts with local search
    # Key insight: we want a set A such that every t in [128,256] has at least one
    # representation a+b=t with a,b in A, a<=b, and we want to minimize the max count.
    
    # Lower bound: total number of unordered pairs from A is |A|*(|A|+1)/2.
    # These pairs produce sums in [0, 256]. The target interval [128,256] has 129 values.
    # If all pairs fell in [128,256] (they don't), we'd need |A|*(|A|+1)/2 >= 129, so |A| >= 16.
    # But many pairs fall outside, so we need larger A.
    # With |A|=40, total pairs = 820. If roughly half fall in target range, that's ~410
    # representations spread over 129 targets, average ~3.2. So peak of ~5-6 might be achievable.
    
    # Try various sizes
    for size in [35, 40, 45, 50, 55, 60]:
        if time.time() > deadline:
            break
        
        # Multiple restarts for each size
        max_restarts = 10 if size <= 45 else 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: random sample
            A = sorted(random.sample(range(0, n + 1), size))
            peak, reps = evaluate(A)
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)
            
            # Local search: swap operations
            # Also try: add/remove if we allow variable size? No, keep fixed size per restart.
            # Actually, let's allow the size to change during search.
            # For now, fixed size local search.
            
            # Precompute for speed: keep A as a set and sorted list
            A_set = set(A)
            
            for iteration in range(3000):
                if time.time() > deadline:
                    break
                
                # Random swap: pick one element to remove, one to add
                old_val = random.choice(A)
                new_val = random.randint(0, n)
                
                if new_val == old_val or new_val in A_set:
                    continue
                
                # Try the swap
                A_new = A[:]
                A_new[A.index(old_val)] = new_val
                A_new_sorted = sorted(A_new)
                
                new_peak, new_reps = evaluate(A_new_sorted)
                
                if new_peak < peak:
                    peak = new_peak
                    A = A_new_sorted
                    A_set = set(A)
                    if peak < best_peak:
                        best_peak = peak
                        best_A = list(A)
                elif new_peak == peak and random.random() < 0.15:
                    A = A_new_sorted
                    A_set = set(A)
    
    # Additional phase: try to reduce size of best_A found
    if best_A is not None:
        if time.time() < deadline - 10:
            # Try removing elements
            A = list(best_A)
            peak, reps = evaluate(A)
            
            for attempt in range(200):
                if time.time() > deadline:
                    break
                if len(A) <= 20:
                    break
                
                # Try removing a random element
                idx = random.randint(0, len(A) - 1)
                A_trial = A[:idx] + A[idx+1:]
                new_peak, new_reps = evaluate(A_trial)
                
                if new_peak <= peak:
                    A = A_trial
                    peak = new_peak
                    if peak < best_peak:
                        best_peak = peak
                        best_A = list(A)
    
    if best_A is None:
        # Fallback: a guaranteed valid set
        return list(range(64, 129))
    
    return best_A
