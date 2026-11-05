import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 129
    target_sums = np.arange(128, 257)
    mid_indices = target_sums // 2
    is_even = (target_sums % 2 == 0)
    
    def evaluate(v):
        c = np.convolve(v, v)
        v_mid = np.where(is_even, v[mid_indices], 0)
        counts = (c[target_sums] + v_mid) / 2.0
        peak = np.max(counts)
        min_count = np.min(counts)
        uncovered = np.sum(counts < 1)
        return peak, uncovered, counts

    def objective(peak, uncovered):
        if uncovered > 0:
            return 1000 + uncovered * 10 + peak
        return peak

    best_v = None
    best_obj = float('inf')
    best_peak = float('inf')
    best_uncovered = float('inf')
    
    # Restart loop
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Initialize
        # Try a mix of random and structured
        if restart == 0:
            v = np.zeros(n, dtype=np.int8)
            v[64:129] = 1
        else:
            # Random subset of size ~60
            size = np.random.randint(50, 70)
            indices = np.random.choice(n, size, replace=False)
            v = np.zeros(n, dtype=np.int8)
            v[indices] = 1
            # Ensure min and max are covered
            if np.min(indices) > 64:
                v[64] = 1
            if np.max(indices) < 128:
                v[128] = 1
            
        current_peak, current_uncovered, _ = evaluate(v)
        current_obj = objective(current_peak, current_uncovered)
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_v = v.copy()
            best_peak = current_peak
            best_uncovered = current_uncovered
            
        # Simulated Annealing
        temp = 5.0
        cooling = 0.999
        stuck_count = 0
        
        while time.time() < deadline - 5.0: # Leave 5s for final check
            idx = np.random.randint(0, n)
            v[idx] = 1 - v[idx]
            
            new_peak, new_uncovered, _ = evaluate(v)
            new_obj = objective(new_peak, new_uncovered)
            
            delta = new_obj - current_obj
            
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_obj = new_obj
                current_peak = new_peak
                current_uncovered = new_uncovered
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_v = v.copy()
                    best_peak = current_peak
                    best_uncovered = current_uncovered
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                v[idx] = 1 - v[idx]
                stuck_count += 1
                
            temp *= cooling
            
            if stuck_count > 1000:
                break

    if best_v is None or best_uncovered > 0:
        return list(range(64, 129))
        
    return list(np.where(best_v == 1)[0])
