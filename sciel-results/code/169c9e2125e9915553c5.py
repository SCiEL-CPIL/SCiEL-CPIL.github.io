"""Additive bases with few representations – optimized search with fast evaluation."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_len = max_sum - min_sum + 1  # 129 values
    
    # Precompute for fast evaluation
    # v is a boolean array of length 129 (indices 0..128)
    # We need to compute unordered pair sum multiplicities for sums in [128, 256]
    
    def compute_counts_from_v(v):
        """
        v: boolean array of length 129
        Returns: array of length 129 with counts for sums 128..256
        Uses convolution for speed.
        """
        vf = v.astype(np.float64)
        # Full convolution gives ordered pair counts for sums 0..256
        c = np.convolve(vf, vf, mode='full')
        # c[s] = sum_{a} v[a]*v[s-a] for s in 0..256
        # For unordered pairs: count[s] = (c[s] + diag[s]) / 2
        # where diag[s] = v[s//2] if s is even and s//2 in [0,128], else 0
        # We only need s in [128, 256]
        # diag for s in [128, 256]: s even means s//2 in [64, 128]
        # v has indices 0..128, so s//2 in [64,128] is valid
        target_s = np.arange(min_sum, max_sum + 1)
        c_target = c[min_sum:max_sum + 1]
        # Compute diagonal contribution
        diag = np.zeros(target_len)
        even_mask = (target_s % 2 == 0)
        for i, s in enumerate(target_s):
            if even_mask[i]:
                a = s // 2
                if 0 <= a <= 128:
                    diag[i] = v[a]
        counts = (c_target + diag) / 2.0
        return counts
    
    def objective_from_v(v):
        """
        Returns (peak, uncovered_count) for the target interval.
        """
        counts = compute_counts_from_v(v)
        peak = int(np.max(counts))
        uncovered = int(np.sum(counts < 1))
        return peak, uncovered
    
    def is_valid_v(v):
        counts = compute_counts_from_v(v)
        return bool(np.all(counts >= 1))
    
    def combined_objective(v):
        """Peak + heavy penalty for uncovered."""
        peak, uncovered = objective_from_v(v)
        return peak + 1000 * uncovered
    
    # Greedy repair: add elements to ensure coverage
    def repair_v(v):
        """Add elements to v to ensure full coverage. Returns modified v."""
        v = v.copy()
        for _ in range(200):
            counts = compute_counts_from_v(v)
            uncovered = np.where(counts < 1)[0]
            if len(uncovered) == 0:
                break
            # Find best element to add
            best_elem = -1
            best_gain = -1
            for x in range(129):
                if v[x]:
                    continue
                new_v = v.copy()
                new_v[x] = True
                new_counts = compute_counts_from_v(new_v)
                new_uncovered = int(np.sum(new_counts < 1))
                gain = len(uncovered) - new_uncovered
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            if best_elem == -1:
                break
            v[best_elem] = True
        return v
    
    # Local search with fast evaluation
    def local_search(v, max_iters=2000, time_limit=10.0):
        """
        Local search on binary vector v.
        Tries swaps, adds, removes.
        Returns best v found.
        """
        t0 = time.time()
        current_v = v.copy()
        current_peak, current_uncov = objective_from_v(current_v)
        best_v = current_v.copy()
        best_peak = current_peak
        best_uncov = current_uncov
        
        if current_uncov == 0:
            best_obj = current_peak
        else:
            best_obj = current_peak + 1000 * current_uncov
        
        iters = 0
        while iters < max_iters:
            if time.time() > t0 + time_limit:
                break
            iters += 1
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Swap: remove one, add one
                in_elems = np.where(current_v)[0]
                out_elems = np.where(~current_v)[0]
                if len(in_elems) == 0 or len(out_elems) == 0:
                    continue
                rem = random.choice(in_elems)
                add = random.choice(out_elems)
                new_v = current_v.copy()
                new_v[rem] = False
                new_v[add] = True
            elif move_type < 0.7:
                # Add
                out_elems = np.where(~current_v)[0]
                if len(out_elems) == 0:
                    continue
                add = random.choice(out_elems)
                new_v = current_v.copy()
                new_v[add] = True
            else:
                # Remove
                in_elems = np.where(current_v)[0]
                if len(in_elems) <= 16:
                    continue
                rem = random.choice(in_elems)
                new_v = current_v.copy()
                new_v[rem] = False
            
            new_peak, new_uncov = objective_from_v(new_v)
            new_obj = new_peak + 1000 * new_uncov
            
            # Accept if better, or with some probability if slightly worse
            delta = new_obj - best_obj
            if new_obj < best_obj:
                best_v = new_v.copy()
                best_peak = new_peak
                best_uncov = new_uncov
                best_obj = new_obj
                current_v = new_v
                current_peak = new_peak
                current_uncov = new_uncov
            elif delta <= 0:
                current_v = new_v
                current_peak = new_peak
                current_uncov = new_uncov
            elif random.random() < 0.05:
                # Accept worse with small probability for exploration
                current_v = new_v
                current_peak = new_peak
                current_uncov = new_uncov
        
        return best_v, best_peak, best_uncov
    
    # Simulated annealing
    def simulated_annealing(v, time_limit=15.0):
        """SA with fast evaluation."""
        t0 = time.time()
        current_v = v.copy()
        current_peak, current_uncov = objective_from_v(current_v)
        best_v = current_v.copy()
        best_peak = current_peak
        best_uncov = current_uncov
        best_obj = current_peak + 1000 * current_uncov
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < t0 + time_limit and T > min_T:
            move_type = random.random()
            
            if move_type < 0.4:
                in_elems = np.where(current_v)[0]
                out_elems = np.where(~current_v)[0]
                if len(in_elems) == 0 or len(out_elems) == 0:
                    T *= cooling
                    continue
                rem = random.choice(in_elems)
                add = random.choice(out_elems)
                new_v = current_v.copy()
                new_v[rem] = False
                new_v[add] = True
            elif move_type < 0.7:
                out_elems = np.where(~current_v)[0]
                if len(out_elems) == 0:
                    T *= cooling
                    continue
                add = random.choice(out_elems)
                new_v = current_v.copy()
                new_v[add] = True
            else:
                in_elems = np.where(current_v)[0]
                if len(in_elems) <= 16:
                    T *= cooling
                    continue
                rem = random.choice(in_elems)
                new_v = current_v.copy()
                new_v[rem] = False
            
            new_peak, new_uncov = objective_from_v(new_v)
            new_obj = new_peak + 1000 * new_uncov
            delta = new_obj - (current_peak + 1000 * current_uncov)
            
            if delta <= 0 or random.random() < np.exp(-max(0, delta) / T):
                current_v = new_v
                current_peak = new_peak
                current_uncov = new_uncov
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_v = new_v.copy()
                    best_peak = new_peak
                    best_uncov = new_uncov
            
            T *= cooling
        
        return best_v, best_peak, best_uncov
    
    # Run multi-start search
    best_v = None
    best_peak = float('inf')
    best_uncov = float('inf')
    
    # Try different sizes and restarts
    for k in range(16, 28):
        if time.time() > deadline - 10:
            break
        
        for restart in range(8):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            v = np.zeros(129, dtype=bool)
            chosen = random.sample(range(129), k)
            v[chosen] = True
            
            # Repair to ensure coverage
            v = repair_v(v)
            
            if not is_valid_v(v):
                continue
            
            # Local search
            ls_time = min(5, deadline - time.time() - 10)
            if ls_time > 1:
                v, peak, uncov = local_search(v, max_iters=1000, time_limit=ls_time)
            
            # Simulated annealing
            sa_time = min(8, deadline - time.time() - 5)
            if sa_time > 2:
                v, peak, uncov = simulated_annealing(v, time_limit=sa_time)
            
            if uncov == 0 and peak < best_peak:
                best_peak = peak
                best_uncov = uncov
                best_v = v.copy()
    
    # Final polish on best solution
    if best_v is not None and time.time() < deadline - 3:
        polish_time = min(5, deadline - time.time() - 2)
        if polish_time > 1:
            best_v, best_peak, best_uncov = local_search(best_v, max_iters=2000, time_limit=polish_time)
    
    # If we don't have a valid solution, use a known valid one
    if best_v is None or best_uncov > 0:
        # Use A = {64, 65, ..., 128} as fallback
        v = np.zeros(129, dtype=bool)
        v[64:129] = True
        best_v = v
        best_peak, best_uncov = objective_from_v(best_v)
    
    # Convert to list
    result = [i for i in range(129) if best_v[i]]
    return sorted(result)
