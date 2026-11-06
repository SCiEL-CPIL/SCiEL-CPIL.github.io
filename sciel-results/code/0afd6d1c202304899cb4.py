"""Optimized additive base construction using simulated annealing with fast evaluation."""
import numpy as np
import time
import random

def construct():
    n = 128
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    def fast_eval(a_bool):
        """a_bool is a numpy bool array of length n+1. Returns (peak, covered)."""
        # Use FFT-based convolution for speed
        a_float = a_bool.astype(np.float64)
        # Zero-pad to avoid circular convolution
        size = 2 * (n + 1)
        conv = np.fft.irfft(np.fft.rfft(a_float, size) ** 2)[:2 * (n + 1)]
        conv = np.round(conv).astype(np.int32)
        
        # For each target sum s in [2n, 2n], compute unordered pair count
        # conv[s] = number of ordered pairs (a,b) with a+b=s where a,b in A
        # Unordered: if s is even and s//2 in A, add 1 to make it odd, then divide by 2
        target_sums = np.arange(2 * n, 2 * n + 1)
        counts = conv[target_sums].copy()
        
        # For even sums, check if half is in A
        for i, s in enumerate(target_sums):
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= n and a_bool[half]:
                    counts[i] += 1
        counts = counts // 2
        
        if np.any(counts < 1):
            return None, False
        return int(np.max(counts)), True
    
    # Start from seed solution
    a_bool = np.zeros(n + 1, dtype=bool)
    for x in range(64, 129):
        a_bool[x] = True
    
    best_set = set(range(64, 129))
    best_peak, _ = fast_eval(a_bool)
    
    current_a = a_bool.copy()
    current_peak = best_peak
    
    rng = random.Random(12345)
    
    T_start = 1.5
    T_end = 0.005
    alpha = 0.997
    T = T_start
    
    no_improve = 0
    max_no_improve = 800
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.30:
            # Add element
            candidates = np.where(~current_a)[0]
            if len(candidates) == 0:
                continue
            elem = rng.choice(candidates.tolist())
            new_a = current_a.copy()
            new_a[elem] = True
        elif move_type < 0.65:
            # Remove element
            candidates = np.where(current_a)[0]
            if len(candidates) <= 8:
                continue
            elem = rng.choice(candidates.tolist())
            new_a = current_a.copy()
            new_a[elem] = False
        else:
            # Swap
            candidates = np.where(current_a)[0]
            if len(candidates) <= 8:
                continue
            remove_elem = rng.choice(candidates.tolist())
            add_candidates = np.where(~current_a)[0]
            if len(add_candidates) == 0:
                continue
            add_elem = rng.choice(add_candidates.tolist())
            new_a = current_a.copy()
            new_a[remove_elem] = False
            new_a[add_elem] = True
        
        new_peak, covered = fast_eval(new_a)
        
        if not covered:
            delta = 50
            accept = False
        else:
            delta = new_peak - current_peak
            if delta <= 0:
                accept = True
            else:
                if T > 0.01:
                    prob = np.exp(-delta / T)
                    accept = rng.random() < prob
                else:
                    accept = False
        
        if accept:
            current_a = new_a
            current_peak = new_peak if covered else current_peak
            if covered and new_peak < best_peak:
                best_peak = new_peak
                best_set = set(np.where(current_a)[0].tolist())
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= alpha
        if T < T_end:
            T = T_end
        
        if no_improve > max_no_improve and time.time() < deadline - 10.0:
            # Restart from best with perturbation
            current_a = np.zeros(n + 1, dtype=bool)
            for x in best_set:
                current_a[x] = True
            for _ in range(rng.randint(3, 10)):
                if rng.random() < 0.5:
                    cands = np.where(~current_a)[0]
                    if len(cands) > 0:
                        current_a[rng.choice(cands.tolist())] = True
                else:
                    cands = np.where(current_a)[0]
                    if len(cands) > 8:
                        current_a[rng.choice(cands.tolist())] = False
            current_peak, cov = fast_eval(current_a)
            if not cov:
                current_a = np.zeros(n + 1, dtype=bool)
                for x in best_set:
                    current_a[x] = True
                current_peak = best_peak
            no_improve = 0
            T = max(T, 0.3)
    
    if best_peak is not None:
        return list(best_set)
    return list(range(64, 129))
