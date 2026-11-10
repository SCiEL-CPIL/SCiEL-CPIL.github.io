import numpy as np
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    target_min = 128
    target_max = 256
    num_targets = target_max - target_min + 1
    
    # Helper to compute unordered counts for a mask
    def get_counts(mask):
        # mask is a 1D array of 0s and 1s of length n+1
        # Convolution gives ordered counts
        # conv[k] = sum_{i+j=k} mask[i]mask[j]
        conv = np.convolve(mask, mask, mode='full')
        # We need counts for sums 128 to 256
        # conv index k corresponds to sum k
        rel_conv = conv[target_min:target_max+1]
        
        # Diagonal correction for unordered pairs
        # U[k] = (ordered[k] + diag[k]) / 2
        # diag[k] is 1 if k is even and mask[k/2] is 1
        k_vals = np.arange(target_min, target_max + 1)
        is_even = (k_vals % 2 == 0)
        half_k = k_vals // 2
        # half_k is in [64, 128], which is within mask bounds [0, 128]
        diag = np.zeros(num_targets, dtype=np.int8)
        diag[is_even] = mask[half_k[is_even]]
        
        unordered = (rel_conv + diag) // 2
        return unordered

    def evaluate(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return None, float('inf')
        peak = np.max(counts)
        # Tie-breaker: minimize sum of squares to flatten distribution
        # This helps escape plateaus where peak is same but distribution is worse
        sum_sq = np.sum(counts**2)
        return peak, sum_sq

    def repair(mask):
        """
        Ensure all targets are covered.
        Adds elements to mask until valid.
        Returns True if repaired within budget, False otherwise.
        """
        A_set = set(np.where(mask)[0])
        for _ in range(200):
            if time.time() > deadline:
                return False
            
            counts = get_counts(mask)
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                return True
            
            # Pick a random uncovered sum
            idx = random.choice(uncovered)
            S = target_min + idx
            
            # Find candidates to add
            # We want to add 'a' such that S-a is in A (or a==S-a)
            # To minimize increase in peak, we might prefer 'a' that is already "central"
            # But for repair, any valid 'a' works.
            candidates = []
            for a in range(n + 1):
                if mask[a] == 1:
                    continue
                b = S - a
                if 0 <= b <= n:
                    if mask[b] == 1:
                        candidates.append(a)
                    elif a == b:
                        candidates.append(a)
            
            if candidates:
                # Choose candidate that minimizes resulting peak?
                # Too expensive. Just pick random.
                new_a = random.choice(candidates)
                mask[new_a] = 1
                A_set.add(new_a)
            else:
                # No single addition can cover S with existing elements.
                # This happens if S is very small or large relative to A.
                # Force add S//2 if possible, or random.
                if S % 2 == 0:
                    mid = S // 2
                    if 0 <= mid <= n and mask[mid] == 0:
                        mask[mid] = 1
                        A_set.add(mid)
                    else:
                        # Add random
                        rand_a = random.randint(0, n)
                        mask[rand_a] = 1
                        A_set.add(rand_a)
                else:
                    rand_a = random.randint(0, n)
                    mask[rand_a] = 1
                    A_set.add(rand_a)
                    
        return False

    best_peak = float('inf')
    best_sum_sq = float('inf')
    best_mask = None
    
    # Run multiple trials with Simulated Annealing
    num_trials = 10
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        # Start with a random set of size 16-18
        size = random.randint(16, 18)
        A_init = set(random.sample(range(n + 1), size))
        mask = np.zeros(n + 1, dtype=np.int8)
        for a in A_init:
            mask[a] = 1
        
        # 2. Repair
        if not repair(mask):
            continue
        
        # 3. Simulated Annealing
        current_peak, current_sq = evaluate(mask)
        if current_peak is None:
            continue
            
        if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sum_sq):
            best_peak = current_peak
            best_sum_sq = current_sq
            best_mask = mask.copy()
            
        T = 10.0
        cooling = 0.995
        no_improve_steps = 0
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick an element to remove
            A_list = np.where(mask)[0]
            if len(A_list) == 0:
                break
            a_remove = A_list[random.randint(0, len(A_list) - 1)]
            
            # Pick an element to add
            # Bias towards elements that are "close" to existing ones or random
            # Random is fine.
            a_add = random.randint(0, n)
            if a_add == a_remove:
                continue
            if mask[a_add] == 1:
                continue
            
            # Perform move
            mask[a_remove] = 0
            mask[a_add] = 1
            
            new_peak, new_sq = evaluate(mask)
            
            if new_peak is None:
                # Invalid, revert
                mask[a_add] = 0
                mask[a_remove] = 1
                no_improve_steps += 1
                continue
            
            # Calculate delta
            # We want to minimize peak, then sum_sq
            # Let's use a combined score: peak * 1000 + sum_sq
            old_score = current_peak * 1000 + current_sq
            new_score = new_peak * 1000 + new_sq
            delta = new_score - old_score
            
            if delta <= 0:
                # Improvement
                current_peak = new_peak
                current_sq = new_sq
                no_improve_steps = 0
                
                if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sum_sq):
                    best_peak = current_peak
                    best_sum_sq = current_sq
                    best_mask = mask.copy()
            else:
                # Worse, accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current_peak = new_peak
                    current_sq = new_sq
                    no_improve_steps = 0
                else:
                    # Revert
                    mask[a_add] = 0
                    mask[a_remove] = 1
                    no_improve_steps += 1
            
            T *= cooling
            if T < 0.01:
                break
                
            if no_improve_steps > 500:
                break

    if best_mask is None:
        # Fallback: return a known valid set (though peak might be high)
        # e.g., 0..128
        return list(range(n + 1))
        
    return list(np.where(best_mask)[0])

