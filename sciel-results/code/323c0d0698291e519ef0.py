import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    target_sums = np.arange(128, 257)
    
    best_A = None
    best_peak = float('inf')
    
    def get_mask(A_set):
        mask = np.zeros(n+1, dtype=np.int8)
        for a in A_set:
            mask[a] = 1
        return mask

    def evaluate_from_mask(mask):
        # Convolution
        conv = np.convolve(mask, mask[::-1], mode='full')
        # conv[k] corresponds to sum k?
        # mask indices 0..n.
        # conv index i corresponds to sum i.
        # Wait, standard convolution: (f*g)[k] = sum f[i]g[k-i].
        # If g is reversed, g[j] = f[n-j]?
        # Let's check.
        # mask = [1, 0, 1] (indices 0, 1, 2). n=2.
        # mask[::-1] = [1, 0, 1].
        # conv = [1, 2, 1, 0, 1].
        # Sum 0: 0+0=0. mask[0]*mask[0]=1. conv[0]=1.
        # Sum 1: 0+1=1. mask[0]*mask[1]=0. mask[1]*mask[0]=0. conv[1]=0?
        # My manual conv above was wrong.
        
        # Let's use np.convolve(mask, mask, mode='full')
        # This gives ordered pairs.
        # conv[k] = sum_{i+j=k} mask[i]mask[j].
        
        ordered_counts = np.convolve(mask, mask, mode='full')
        
        # We need unordered counts for sums in [128, 256].
        # Unordered count U[k] = (ordered_counts[k] + diag[k]) / 2
        # where diag[k] is 1 if k is even and mask[k/2]==1, else 0.
        
        # We only care about k in 128..256.
        # ordered_counts has length 2*n + 1 = 257.
        # Indices 0..256.
        
        # Extract relevant slice
        rel_ordered = ordered_counts[128:257]
        
        # Calculate diagonal contribution
        # For each k in 128..256:
        # if k % 2 == 0:
        #    a = k // 2
        #    if 0 <= a <= n and mask[a] == 1: diag = 1
        #    else: diag = 0
        # else: diag = 0
        
        # Vectorized diag calculation
        k_vals = np.arange(128, 257)
        is_even = (k_vals % 2 == 0)
        half_k = k_vals // 2
        # half_k is in [64, 128]. All valid indices for mask (0..128).
        diag_mask = np.zeros(len(k_vals), dtype=np.int8)
        diag_mask[is_even] = mask[half_k[is_even]]
        
        unordered_counts = (rel_ordered + diag_mask) // 2
        
        # Check coverage
        if np.any(unordered_counts == 0):
            return None, 0
            
        peak = np.max(unordered_counts)
        return unordered_counts, peak

    def repair(A_set):
        A_set = set(A_set)
        mask = get_mask(A_set)
        
        for _ in range(100):
            _, peak = evaluate_from_mask(mask)
            # Check coverage
            # We need to know if any is 0.
            # evaluate_from_mask returns None if uncovered.
            # But we need to find WHICH sum is uncovered to repair.
            
            # Let's make a version that returns the counts or None
            ordered_counts = np.convolve(mask, mask, mode='full')
            rel_ordered = ordered_counts[128:257]
            k_vals = np.arange(128, 257)
            is_even = (k_vals % 2 == 0)
            half_k = k_vals // 2
            diag_mask = np.zeros(len(k_vals), dtype=np.int8)
            diag_mask[is_even] = mask[half_k[is_even]]
            unordered_counts = (rel_ordered + diag_mask) // 2
            
            if np.all(unordered_counts > 0):
                return A_set
                
            # Find first uncovered sum
            uncovered_idx = np.where(unordered_counts == 0)[0]
            if len(uncovered_idx) == 0:
                return A_set
            S = 128 + uncovered_idx[0]
            
            # Find candidates
            candidates = []
            for a in range(n+1):
                if a in A_set:
                    continue
                b = S - a
                if 0 <= b <= n:
                    if b in A_set:
                        candidates.append(a)
                    elif a == b:
                        candidates.append(a)
                        
            if candidates:
                new_a = random.choice(candidates)
                A_set.add(new_a)
                mask[new_a] = 1
            else:
                if S % 2 == 0:
                    A_set.add(S//2)
                    mask[S//2] = 1
                else:
                    A_set.add(random.randint(0, n))
                    mask[random.randint(0, n)] = 1 # BUG: need to use same random
                    # Fix:
                    rand_a = random.randint(0, n)
                    A_set.add(rand_a)
                    mask[rand_a] = 1
                    
        return A_set

    for trial in range(50):
        if time.time() > deadline:
            break
            
        size = random.randint(50, 65)
        A_set = set(random.sample(range(n+1), size))
        A_set = repair(A_set)
        
        mask = get_mask(A_set)
        counts, peak = evaluate_from_mask(mask)
        
        if counts is None:
            continue
            
        if peak < best_peak:
            best_peak = peak
            best_A = list(A_set)
            
        # Local Search
        current_peak = peak
        current_A = set(A_set)
        current_mask = mask.copy()
        
        no_improve = 0
        for i in range(500):
            if time.time() > deadline:
                break
                
            x = random.choice(list(current_A))
            y = random.randint(0, n)
            if y in current_A:
                continue
                
            # Update mask
            current_mask[x] = 0
            current_mask[y] = 1
            
            new_counts, new_peak = evaluate_from_mask(current_mask)
            
            if new_counts is None:
                # Revert
                current_mask[x] = 1
                current_mask[y] = 0
                continue
                
            if new_peak < current_peak:
                current_peak = new_peak
                current_A.remove(x)
                current_A.add(y)
                no_improve = 0
            else:
                # Revert
                current_mask[x] = 1
                current_mask[y] = 0
                no_improve += 1
                
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(current_A)
                
            if no_improve > 100:
                break
                
    if best_A is None:
        return list(range(64, 129))
        
    return best_A
