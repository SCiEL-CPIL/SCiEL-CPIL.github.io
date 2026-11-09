import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    
    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH using unordered pairs.
        Uses convolution for speed."""
        # Indicator vector for A over [0, 128]
        ind = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A:
            ind[a] = 1
        # Convolution: ordered pairs count for each sum
        conv = np.convolve(ind, ind, mode='full')  # length 2*MAX_VAL+1 = 257
        # For sum s, ordered count = conv[s]
        # Unordered count = (ordered + diag) // 2, where diag = 1 if s even and s//2 in A
        # We only need sums in [N_LOW, N_HIGH]
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            ordered = conv[s]
            diag = 1 if (s % 2 == 0 and ind[s // 2] == 1) else 0
            reps[idx] = (ordered + diag) // 2
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # --- Phase 1: Construct a structured initial candidate ---
    best_A = None
    best_peak = float('inf')
    
    # Greedy construction with bitset-based coverage tracking
    A_current = {0, 128}
    reps = calc_reps(A_current)
    
    # Track coverage with a bitset for speed
    covered_mask = np.zeros(RANGE_SIZE, dtype=np.int32)
    for i in range(RANGE_SIZE):
        if reps[i] > 0:
            covered_mask[i] = 1
    
    # Greedily add elements
    while not np.all(covered_mask) and len(A_current) < 80:
        best_x = None
        best_new_covered = 0
        best_reps = None
        best_peak_val = float('inf')
        
        candidates = list(range(0, MAX_VAL + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in A_current:
                continue
            new_A = A_current | {x}
            new_reps = calc_reps(new_A)
            
            new_covered_count = 0
            for i in range(RANGE_SIZE):
                if covered_mask[i] == 0 and new_reps[i] > 0:
                    new_covered_count += 1
            
            if new_covered_count > best_new_covered:
                best_new_covered = new_covered_count
                best_x = x
                best_reps = new_reps
                best_peak_val = get_peak(new_reps)
            elif new_covered_count == best_new_covered:
                peak = get_peak(new_reps)
                if peak < best_peak_val:
                    best_peak_val = peak
                    best_x = x
                    best_reps = new_reps
        
        if best_x is None:
            break
        
        A_current.add(best_x)
        reps = best_reps if best_reps is not None else calc_reps(A_current)
        
        for i in range(RANGE_SIZE):
            if reps[i] > 0:
                covered_mask[i] = 1
        
        if np.all(covered_mask):
            break
    
    if is_covered(reps):
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A_current)
    else:
        # Fallback
        A_current = set(range(64, 129))
        reps = calc_reps(A_current)
        if is_covered(reps):
            best_peak = get_peak(reps)
            best_A = list(A_current)
    
    # --- Phase 2: Try to improve by removing elements ---
    if best_A is not None:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in list(current_A):
                test_A = current_A - {x}
                if len(test_A) == 0:
                    continue
                test_reps = calc_reps(test_A)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak < current_peak:
                        current_A = test_A
                        current_reps = test_reps
                        current_peak = test_peak
                        improved = True
                        break
                    elif test_peak == current_peak:
                        current_A = test_A
                        current_reps = test_reps
                        improved = True
                        break
        
        if is_covered(current_reps):
            peak = get_peak(current_reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(current_A)
    
    # --- Phase 3: Local search to balance representations ---
    if best_A is not None and time.time() < deadline - 10:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        while time.time() < deadline - 5 and T > min_T:
            if len(current_A) == 0:
                break
            
            item_to_remove = random.choice(list(current_A))
            new_item = random.randint(0, 128)
            
            if new_item == item_to_remove:
                T *= cooling_rate
                continue
            
            new_A = set(current_A)
            new_A.remove(item_to_remove)
            new_A.add(new_item)
            
            if len(new_A) == 0:
                T *= cooling_rate
                continue
            
            new_reps = calc_reps(new_A)
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                delta = new_peak - current_peak
                
                if delta < 0:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_reps = new_reps
                        current_peak = new_peak
            else:
                pass
            
            T *= cooling_rate
        
        # Try removal after local search
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in list(current_A):
                    test_A = current_A - {x}
                    if len(test_A) == 0:
                        continue
                    test_reps = calc_reps(test_A)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak < current_peak:
                            current_A = test_A
                            current_reps = test_reps
                            current_peak = test_peak
                            improved = True
                            break
                        elif test_peak == current_peak:
                            current_A = test_A
                            current_reps = test_reps
                            improved = True
                            break
            
            if is_covered(current_reps):
                peak = get_peak(current_reps)
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(current_A)
    
    # --- Phase 4: Final peak-breaking polish ---
    if best_A is not None and time.time() < deadline - 3:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        polish_iter = 0
        while time.time() < deadline - 3 and polish_iter < 5000:
            polish_iter += 1
            if current_peak <= 1:
                break
            
            peak_indices = [i for i in range(RANGE_SIZE) if current_reps[i] == current_peak]
            if not peak_indices:
                break
            target_s = N_LOW + random.choice(peak_indices)
            
            valid_removals = []
            for a in current_A:
                b = target_s - a
                if b in current_A and a <= b:
                    valid_removals.append(a)
            
            if not valid_removals:
                continue
            
            a_to_remove = random.choice(valid_removals)
            best_new_A = None
            best_new_peak = current_peak
            best_new_reps = None
            
            for _ in range(5):
                new_item = random.randint(0, 128)
                if new_item == a_to_remove:
                    continue
                new_A = set(current_A)
                new_A.remove(a_to_remove)
                new_A.add(new_item)
                if len(new_A) == 0:
                    continue
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_new_peak:
                        best_new_peak = new_peak
                        best_new_A = new_A
                        best_new_reps = new_reps
            
            if best_new_A is not None and best_new_peak < current_peak:
                current_A = best_new_A
                current_reps = best_new_reps
                current_peak = best_new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(current_A)
    
    if best_A is None:
        return list(range(64, 129))
    
    return best_A
