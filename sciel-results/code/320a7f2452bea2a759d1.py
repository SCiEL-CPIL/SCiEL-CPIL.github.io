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
        """Compute representation counts for sums N_LOW..N_HIGH using unordered pairs."""
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = np.array(sorted(A), dtype=np.int32)
        n = len(A_list)
        if n == 0:
            return reps
        tri_i, tri_j = np.triu_indices(n)
        sums = A_list[tri_i] + A_list[tri_j]
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if np.any(mask):
            indices = sums[mask] - N_LOW
            np.add.at(reps, indices, 1)
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # --- Phase 1: Bitset-based greedy set cover ---
    # For each candidate x in [0,128], precompute which targets it covers
    # when added to a current set A. A target s is covered if there exists
    # a in A such that s-a is in A (and we count unordered pairs).
    
    # Start with a structured initial set
    # Use two arithmetic progressions interleaved
    A_current = set()
    # Add elements from two APs to spread sums
    for i in range(0, 129, 2):
        A_current.add(i)
    
    reps = calc_reps(A_current)
    
    if not is_covered(reps):
        # Greedily add elements to cover all targets
        uncovered = set(range(N_LOW, N_HIGH + 1))
        for s in range(N_LOW, N_HIGH + 1):
            if reps[s - N_LOW] > 0:
                uncovered.discard(s)
        
        while len(uncovered) > 0 and len(A_current) < 90:
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
                
                if is_covered(new_reps):
                    peak = get_peak(new_reps)
                    if peak < best_peak_val:
                        best_x = x
                        best_reps = new_reps
                        best_peak_val = peak
                        best_new_covered = 1000
                else:
                    new_covered_count = 0
                    for s in range(N_LOW, N_HIGH + 1):
                        if reps[s - N_LOW] == 0 and new_reps[s - N_LOW] > 0:
                            new_covered_count += 1
                    peak = get_peak(new_reps)
                    
                    if new_covered_count > best_new_covered or (new_covered_count == best_new_covered and peak < best_peak_val):
                        best_new_covered = new_covered_count
                        best_x = x
                        best_reps = new_reps
                        best_peak_val = peak
            
            if best_x is None:
                break
            
            A_current.add(best_x)
            reps = best_reps if best_reps is not None else calc_reps(A_current)
            
            for s in range(N_LOW, N_HIGH + 1):
                if reps[s - N_LOW] > 0:
                    uncovered.discard(s)
            
            if len(uncovered) == 0:
                break
    
    best_A = None
    best_peak = float('inf')
    
    if is_covered(reps):
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A_current)
    else:
        # Fallback: dense set
        A_current = set(range(64, 129))
        reps = calc_reps(A_current)
        if is_covered(reps):
            best_peak = get_peak(reps)
            best_A = list(A_current)
    
    # --- Phase 2: Remove redundant elements ---
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
    
    # --- Phase 3: Simulated annealing with contribution tracking ---
    if best_A is not None and time.time() < deadline - 10:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        # Precompute contribution vectors for each element
        # contrib[x] = vector of how many times x participates in each target sum
        # When we remove x, reps_new = reps - contrib[x]
        # When we add x, we need to add contributions from pairs (x, a) for a in A, a < x
        # plus (x, x) if 2x is in target range
        
        T = 1.0
        cooling_rate = 0.999
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
            new_A.discard(item_to_remove)
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
        
        # Final removal pass
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
    
    if best_A is None:
        return list(range(64, 129))
    
    return best_A
