import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)
    sum_to_idx = {s: i for i, s in enumerate(target_sums)}
    
    def compute_counts(A_set):
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        m = len(A_list)
        A_arr = np.array(A_list)
        for i in range(m):
            ai = A_arr[i]
            for j in range(i, m):
                s = ai + A_arr[j]
                if min_sum <= s <= max_sum:
                    counts[sum_to_idx[s]] += 1
        return counts

    def get_peak(counts):
        return int(np.max(counts)) if len(counts) > 0 else 0

    def is_valid(counts):
        return np.all(counts >= 1)

    def update_counts_remove(A_set, counts, x):
        new_counts = counts.copy()
        for y in A_set:
            s = x + y
            if min_sum <= s <= max_sum:
                new_counts[sum_to_idx[s]] -= 1
        return new_counts

    def update_counts_add(A_set, counts, x):
        new_counts = counts.copy()
        for y in A_set:
            s = x + y
            if min_sum <= s <= max_sum:
                new_counts[sum_to_idx[s]] += 1
        s = 2 * x
        if min_sum <= s <= max_sum:
            new_counts[sum_to_idx[s]] += 1
        return new_counts

    best_A = None
    best_peak = float('inf')
    
    # Run multiple restarts to escape local minima
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with full set or slightly perturbed
        A = set(range(0, 129))
        counts = compute_counts(A)
        current_peak = get_peak(counts)
        
        # Greedy Removal
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            candidates = list(A)
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if x not in A:
                    continue
                new_counts = update_counts_remove(A, counts, x)
                if not is_valid(new_counts):
                    continue
                new_peak = get_peak(new_counts)
                if new_peak < current_peak:
                    A.remove(x)
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                elif new_peak == current_peak and random.random() < 0.05:
                    A.remove(x)
                    counts = new_counts
                    improved = True
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = A.copy()
            
        # Local Search (Swaps)
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        no_improve_count = 0
        
        while time.time() < deadline - 2 and no_improve_count < 2000:
            if not A:
                break
            x = random.choice(list(A))
            # Efficiently pick y not in A
            # Since n is small, list comprehension is ok
            not_in_A = [i for i in range(129) if i not in A]
            if not not_in_A:
                break
            y = random.choice(not_in_A)
            
            # Swap x and y
            counts_no_x = update_counts_remove(A, counts, x)
            A_temp = A.copy()
            A_temp.remove(x)
            counts_swap = update_counts_add(A_temp, counts_no_x, y)
            
            if not is_valid(counts_swap):
                no_improve_count += 1
                continue
                
            new_peak = get_peak(counts_swap)
            delta = new_peak - current_peak
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < np.exp(-delta / T)
                
            if accept:
                A.remove(x)
                A.add(y)
                counts = counts_swap
                current_peak = new_peak
                no_improve_count = 0
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = A.copy()
            else:
                no_improve_count += 1
                
            T = max(T_min, T * cooling)

    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))
    
    return sorted(best_A)
