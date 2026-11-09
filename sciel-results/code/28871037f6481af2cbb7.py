"""Suite task: distinct subset sums, n=12, minimize the largest element."""
# EVOLVE-BLOCK-START
import time
import random

def _check_valid_fast(arr):
    """Fast check: all 4096 subset sums distinct for n=12."""
    n = 12
    sums = [0] * (1 << n)
    for i in range(n):
        a = arr[i]
        for mask in range(1 << i):
            sums[(1 << i) | mask] = sums[mask] + a
    seen = set()
    for s in sums:
        if s in seen:
            return False
        seen.add(s)
    return True

def construct():
    """Return exactly 12 distinct positive integers <= 1048576 with all 4096
    subset sums distinct. Minimizes the largest integer.
    
    Uses simulated annealing to explore non-superincreasing sequences
    that may have a smaller maximum element than powers of 2.
    """
    _t0 = time.monotonic()
    _DEADLINE = 85.0
    n = 12
    random.seed(777)
    
    best = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    best_max = 2048
    
    def simulated_annealing(arr, deadline):
        arr = list(arr)
        current_max = max(arr)
        best_arr = list(arr)
        best_max_local = current_max
        T = 100.0
        cooling = 0.999
        
        while (time.monotonic() - _t0) < deadline and T > 0.1:
            i = random.randint(0, n - 1)
            old_val = arr[i]
            used = set(arr)
            used.discard(arr[i])
            
            if random.random() < 0.7:
                new_val = random.randint(1, max(1, old_val - 1))
            else:
                new_val = random.randint(1, old_val + 50)
            
            if new_val in used:
                T *= cooling
                continue
            
            arr[i] = new_val
            if not _check_valid_fast(arr):
                arr[i] = old_val
                T *= cooling
                continue
            
            new_max = max(arr)
            delta = new_max - current_max
            
            if delta < 0 or random.random() < (2.718281828 ** (-delta / T)):
                current_max = new_max
                if new_max < best_max_local:
                    best_arr = list(arr)
                    best_max_local = new_max
            
            T *= cooling
        
        return best_arr
    
    best = simulated_annealing(best, _DEADLINE)
    
    if not _check_valid_fast(best):
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best
# EVOLVE-BLOCK-END
