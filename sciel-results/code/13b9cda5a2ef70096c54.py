import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256

    def fast_objective(x_bin):
        """
        Vectorized objective using convolution.
        x_bin is a binary array of length 129.
        Returns (peak, uncovered_count).
        """
        if x_bin.sum() == 0:
            return float('inf'), 129
        
        # Convolve x with itself to get sum counts
        # convolve[x] = sum over i of x[i]*x[x-i]
        # This counts ordered pairs; unordered = (ordered + diagonal) / 2
        # But we can work with ordered and adjust
        conv = np.convolve(x_bin, x_bin, mode='full')
        # conv[s] = number of ordered pairs (a,b) in A with a+b=s
        # For unordered pairs allowing equal: count = (conv[s] + x_bin[s//2] if s even) / 2
        # Actually simpler: just use conv and divide appropriately
        # For our purposes, we just need relative ordering, so use conv directly
        # but we need to handle the diagonal correction for exact counts
        
        # Get counts for target range
        counts = conv[min_sum:max_sum+1].astype(np.float64)
        
        # Correct for unordered: for sum s, unordered count = (conv[s] + (1 if s even and s//2 in A else 0)) / 2
        # conv[s] counts (a,b) and (b,a) separately, plus (a,a) once
        # Unordered with replacement: for s, count = number of pairs {a,b} with a<=b, a+b=s
        # = (conv[s] + diag[s]) / 2 where diag[s]=1 if s even and s/2 in A, else 0
        
        diag = np.zeros(max_sum - min_sum + 1)
        for idx, s in enumerate(range(min_sum, max_sum + 1)):
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= 128:
                    diag[idx] = x_bin[mid]
        
        unordered_counts = (counts + diag) / 2.0
        unordered_counts = np.round(unordered_counts).astype(np.int32)
        
        uncovered = np.sum(unordered_counts == 0)
        if uncovered > 0:
            return float('inf'), int(uncovered)
        
        peak = int(np.max(unordered_counts))
        return peak, 0

    # Initialize with structured approach: try various two-interval sets
    best_peak = float('inf')
    best_A = None

    for a in range(0, 70):
        for b in range(60, 129):
            if a >= b:
                continue
            A_set = set(range(0, a + 1)) | set(range(b, 129))
            x_bin = np.zeros(129, dtype=np.int8)
            for x in A_set:
                x_bin[x] = 1
            peak, unc = fast_objective(x_bin)
            if peak < best_peak:
                best_peak = peak
                best_A = A_set.copy()

    if best_A is None:
        best_A = set(range(0, 70))
        best_peak = 100

    # Simulated annealing on binary vector
    random.seed(42)
    np.random.seed(42)
    
    x_current = np.zeros(129, dtype=np.int8)
    for x in best_A:
        x_current[x] = 1
    
    x_best = x_current.copy()
    best_peak_sa = best_peak
    
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.9995
    T = T_start
    
    iter_count = 0
    while time.time() < deadline and T > T_end:
        # Check time frequently
        if time.time() > deadline - 2:
            break
        
        # Propose a move
        move_type = random.random()
        x_proposed = x_current.copy()
        
        if move_type < 0.35:
            # Add an element
            candidates = np.where(x_current == 0)[0]
            if len(candidates) > 0:
                idx = random.choice(candidates)
                x_proposed[idx] = 1
        elif move_type < 0.65:
            # Remove an element
            if x_current.sum() > 10:
                candidates = np.where(x_current == 1)[0]
                if len(candidates) > 0:
                    idx = random.choice(candidates)
                    x_proposed[idx] = 0
        else:
            # Swap
            if x_current.sum() > 10:
                ones = np.where(x_current == 1)[0]
                zeros = np.where(x_current == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    idx_remove = random.choice(ones)
                    idx_add = random.choice(zeros)
                    x_proposed[idx_remove] = 0
                    x_proposed[idx_add] = 1
        
        peak_prop, unc_prop = fast_objective(x_proposed)
        peak_cur, unc_cur = fast_objective(x_current)
        
        # Objective: minimize peak, but coverage is critical
        obj_cur = peak_cur + 1000 * unc_cur
        obj_prop = peak_prop + 1000 * unc_prop
        
        delta_obj = obj_prop - obj_cur
        
        if delta_obj < 0 or (T > 0 and random.random() < np.exp(-delta_obj / T)):
            x_current = x_proposed
            if peak_prop < best_peak_sa and unc_prop == 0:
                best_peak_sa = peak_prop
                x_best = x_proposed.copy()
        
        T *= cooling_rate
        iter_count += 1
    
    # Extract best solution
    best_A_final = set(np.where(x_best > 0)[0].tolist())
    if len(best_A_final) == 0:
        best_A_final = best_A
    
    # Final local search refinement with fast objective
    current_A = best_A_final.copy()
    x_cur = np.zeros(129, dtype=np.int8)
    for x in current_A:
        x_cur[x] = 1
    
    cur_peak, _ = fast_objective(x_cur)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        move_type = random.random()
        x_new = x_cur.copy()
        
        if move_type < 0.4:
            candidates = np.where(x_cur == 0)[0]
            if len(candidates) == 0:
                break
            idx = random.choice(candidates)
            x_new[idx] = 1
        elif move_type < 0.7:
            if x_cur.sum() <= 10:
                no_improve += 1
                continue
            candidates = np.where(x_cur == 1)[0]
            if len(candidates) == 0:
                break
            idx = random.choice(candidates)
            x_new[idx] = 0
        else:
            if x_cur.sum() <= 10:
                no_improve += 1
                continue
            ones = np.where(x_cur == 1)[0]
            zeros = np.where(x_cur == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            idx_rem = random.choice(ones)
            idx_add = random.choice(zeros)
            x_new[idx_rem] = 0
            x_new[idx_add] = 1
        
        new_peak, new_unc = fast_objective(x_new)
        if new_unc == 0 and new_peak < cur_peak:
            x_cur = x_new
            cur_peak = new_peak
            no_improve = 0
        else:
            no_improve += 1
    
    final_A = set(np.where(x_cur > 0)[0].tolist())
    if len(final_A) == 0:
        final_A = best_A_final
    
    return sorted(final_A)
