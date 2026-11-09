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
        ind = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A:
            ind[a] = 1
        conv = np.convolve(ind, ind, mode='full')
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
    
    # Greedy construction with incremental updates is complex to set up initially.
    # Let's stick to the fast greedy but optimize the candidate evaluation.
    # Actually, let's just use the existing greedy but make it faster by precomputing 
    # or using a simpler heuristic if possible.
    # The existing greedy is O(|A| * |Candidates| * Convolution).
    # |A| ~ 50, |Candidates| ~ 129. 50 * 129 * Convolution.
    # Convolution is fast. Let's assume it's okay for Phase 1.
    
    A_current = {0, 128}
    reps = calc_reps(A_current)
    
    uncovered = set(range(N_LOW, N_HIGH + 1))
    for s in range(N_LOW, N_HIGH + 1):
        if reps[s - N_LOW] > 0:
            uncovered.discard(s)
    
    # Greedily add elements
    while len(uncovered) > 0 and len(A_current) < 80:
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
            
            if not is_covered(new_reps):
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
            else:
                peak = get_peak(new_reps)
                if peak < best_peak_val or best_reps is None:
                    best_x = x
                    best_reps = new_reps
                    best_peak_val = peak
                    best_new_covered = 1000
        
        if best_x is None:
            break
        
        A_current.add(best_x)
        reps = best_reps if best_reps is not None else calc_reps(A_current)
        
        for s in range(N_LOW, N_HIGH + 1):
            if reps[s - N_LOW] > 0:
                uncovered.discard(s)
        
        if len(uncovered) == 0:
            break
    
    if is_covered(reps):
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A_current)
    else:
        A_current = set(range(64, 129))
        reps = calc_reps(A_current)
        if is_covered(reps):
            best_peak = get_peak(reps)
            best_A = list(A_current)
    
    # --- Phase 2: Removal ---
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
    
    # --- Phase 3: Local Search with Incremental Updates ---
    if best_A is not None and time.time() < deadline - 10:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        # Helper to update reps incrementally
        def update_reps_remove(reps, A, x):
            # reps is a numpy array for N_LOW..N_HIGH
            # A is the set BEFORE removal
            # We need to subtract contributions of x
            # Pairs {x, x} -> 2x
            # Pairs {a, x} -> a+x for a in A\{x}
            
            # Make a copy to avoid modifying in place if we reject
            new_reps = reps.copy()
            
            # Remove {x, x}
            s = 2*x
            if N_LOW <= s <= N_HIGH:
                new_reps[s - N_LOW] -= 1
            
            # Remove {a, x}
            for a in A:
                if a == x:
                    continue
                s = a + x
                if N_LOW <= s <= N_HIGH:
                    new_reps[s - N_LOW] -= 1
            return new_reps

        def update_reps_add(reps, A, y):
            # A is the set BEFORE addition
            new_reps = reps.copy()
            
            # Add {y, y}
            s = 2*y
            if N_LOW <= s <= N_HIGH:
                new_reps[s - N_LOW] += 1
            
            # Add {a, y}
            for a in A:
                if a == y:
                    continue
                s = a + y
                if N_LOW <= s <= N_HIGH:
                    new_reps[s - N_LOW] += 1
            return new_reps

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
            
            # Calculate new reps incrementally
            # 1. Remove item_to_remove
            reps_after_remove = update_reps_remove(current_reps, current_A, item_to_remove)
            A_after_remove = current_A - {item_to_remove}
            
            # 2. Add new_item
            new_reps = update_reps_add(reps_after_remove, A_after_remove, new_item)
            new_A = A_after_remove | {new_item}
            
            if len(new_A) == 0:
                T *= cooling_rate
                continue
            
            # Check coverage and peak
            if np.any(new_reps < 0): # Sanity check, shouldn't happen
                T *= cooling_rate
                continue
                
            if np.all(new_reps > 0):
                new_peak = int(np.max(new_reps))
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
                # Reject if not covered
                pass
            
            T *= cooling_rate
        
        # Polish
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
