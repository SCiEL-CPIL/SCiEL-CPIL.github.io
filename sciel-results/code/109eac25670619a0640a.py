import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    target_len = target_max - target_min + 1
    
    best_peak = float('inf')
    best_A = None
    
    # Try different set sizes
    for k in range(18, 41):
        if time.time() > deadline - 10.0:
            break
            
        # We can limit restarts based on remaining time
        # Estimate time per restart
        # Let's just run until time runs out or we find a perfect solution (peak 1 or 2)
        
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize A
            # Random start
            A = set(random.sample(range(n + 1), k))
            A_arr = np.array(list(A), dtype=int)
            
            def get_mults(arr):
                # arr is 1D numpy array
                sums = arr[:, None] + arr[None, :]
                # We only care about upper triangle including diagonal
                i, j = np.triu_indices(len(arr), k=0)
                valid_sums = sums[i, j]
                mask = (valid_sums >= target_min) & (valid_sums <= target_max)
                valid_sums = valid_sums[mask]
                # bincount requires non-negative ints
                # valid_sums are in [128, 256]
                mults = np.bincount(valid_sums, minlength=target_max + 1)
                return mults[target_min:target_max+1]
            
            mults = get_mults(A_arr)
            uncovered = np.sum(mults == 0)
            
            def get_cost(mults):
                unc = np.sum(mults == 0)
                if unc > 0:
                    return 10000 + unc * 100 + np.max(mults)
                return np.max(mults)
            
            current_cost = get_cost(mults)
            
            T = 10.0
            T_min = 0.1
            cooling_rate = 0.9995
            
            while time.time() < deadline - 1.0:
                # Pick random swap
                x_idx = random.randint(0, len(A_arr) - 1)
                x = A_arr[x_idx]
                y = random.randint(0, n)
                if x == y:
                    continue
                    
                # Create new array
                new_A_arr = A_arr.copy()
                new_A_arr[x_idx] = y
                # Ensure uniqueness? 
                # random.sample gave unique. Swap might create duplicate if y is already in A.
                # If y is in A, then we have a duplicate. 
                # The problem says "subset", so elements must be unique.
                # So if y is in A (and y != x), we can't just swap.
                # We should pick y from [0, 128] \ A.
                
                # Efficient way: pick y from available
                # Or just check if y in A.
                # Since k is small, checking `y in A` (set) is O(1).
                if y in A:
                    continue
                
                A.remove(x)
                A.add(y)
                A_arr = np.array(list(A), dtype=int)
                
                new_mults = get_mults(A_arr)
                new_cost = get_cost(new_mults)
                
                delta = new_cost - current_cost
                if delta < 0:
                    current_cost = new_cost
                    mults = new_mults
                else:
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        mults = new_mults
                    else:
                        # Revert
                        A.remove(y)
                        A.add(x)
                        A_arr = np.array(list(A), dtype=int)
                        mults = get_mults(A_arr) # Recalc to be safe? Or keep old?
                        # Keeping old mults is fine if we revert state correctly.
                        # But A_arr was modified. We reverted A and A_arr.
                        # mults corresponds to the reverted state?
                        # No, mults was from previous accepted state.
                        # If we reject, we stay at previous state.
                        # So we don't need to recalc.
                        pass
                        
                T *= cooling_rate
                if T < T_min:
                    T = T_min
                    
            if current_cost < 10000:
                peak = np.max(mults)
                if peak < best_peak:
                    best_peak = peak
                    best_A = A.copy()
                    
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)
