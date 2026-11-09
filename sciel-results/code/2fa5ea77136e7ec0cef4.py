import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH."""
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

    def objective(reps):
        """Lower is better. Penalize uncovered sums heavily."""
        if not is_covered(reps):
            uncovered = np.sum(reps == 0)
            return 1000 + uncovered * 10 + get_peak(reps)
        return get_peak(reps)

    best_peak = float('inf')
    best_A = None
    best_reps = None

    # Structural change: Use greedy set-cover initialization + targeted local search
    # Key insight: First greedily pick elements that cover the most uncovered sums,
    # then use local search that specifically targets the highest-multiplicity sums.
    
    def greedy_initialization():
        """Greedily build A by picking elements that cover the most uncovered sums."""
        A = set()
        # Must include 128 to cover 256
        A.add(128)
        
        reps = calc_reps(A)
        
        # Greedily add elements that maximize coverage of uncovered sums
        while not is_covered(reps) and time.time() < deadline - 20:
            best_elem = None
            best_gain = -1
            best_new_reps = None
            
            for candidate in range(MAX_VAL + 1):
                if candidate in A:
                    continue
                test_A = A | {candidate}
                test_reps = calc_reps(test_A)
                
                # Gain: number of newly covered sums
                newly_covered = np.sum((test_reps > 0) & (reps == 0))
                # Also consider reducing peak among already-covered sums
                covered_mask = reps > 0
                if np.any(covered_mask):
                    peak_before = np.max(reps[covered_mask])
                    peak_after = np.max(test_reps[covered_mask])
                    peak_reduction = peak_before - peak_after
                else:
                    peak_reduction = 0
                
                # Score: prioritize covering new sums, then reducing peak
                score = newly_covered * 10 + peak_reduction
                
                if score > best_gain:
                    best_gain = score
                    best_elem = candidate
                    best_new_reps = test_reps
            
            if best_elem is None or best_gain <= 0:
                break
            
            A.add(best_elem)
            reps = best_new_reps
        
        return A, reps

    # Phase 1: Greedy initialization
    A_greedy, reps_greedy = greedy_initialization()
    
    if is_covered(reps_greedy):
        peak = get_peak(reps_greedy)
        best_peak = peak
        best_A = list(A_greedy)
        best_reps = reps_greedy
    else:
        # Fallback: try random initialization
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            size = random.randint(25, 45)
            A_rand = set(random.sample(range(MAX_VAL + 1), size))
            if 128 not in A_rand:
                A_rand.remove(random.choice(list(A_rand)))
                A_rand.add(128)
            
            reps_rand = calc_reps(A_rand)
            if is_covered(reps_rand):
                peak = get_peak(reps_rand)
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A_rand)
                    best_reps = reps_rand

    # Phase 2: Targeted local search that attacks highest-multiplicity sums
    if best_A is not None:
        curr_A = set(best_A)
        curr_reps = calc_reps(curr_A)
        curr_peak = get_peak(curr_reps)
        
        # Targeted local search: specifically try to reduce the peak
        # by removing elements that contribute to the highest-multiplicity sums
        # or swapping them with elements that don't.
        
        max_iterations = 500
        for iter_num in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if curr_peak <= 2:
                break
            
            # Find the sums with the highest multiplicity
            peak_sums = np.where(curr_reps == curr_peak)[0] + N_LOW
            
            # Try to reduce peak by modifying A
            improved_this_iter = False
            
            # Strategy 1: Try removing an element that contributes to peak sums
            for x in list(curr_A):
                if time.time() > deadline - 5:
                    break
                
                test_A = curr_A - {x}
                if len(test_A) == 0:
                    continue
                test_reps = calc_reps(test_A)
                
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak < curr_peak:
                        curr_A = test_A
                        curr_reps = test_reps
                        curr_peak = test_peak
                        improved_this_iter = True
                        break
                    elif test_peak == curr_peak and len(test_A) < len(curr_A):
                        curr_A = test_A
                        curr_reps = test_reps
                        improved_this_iter = True
                        break
            
            if improved_this_iter:
                if curr_peak < best_peak:
                    best_peak = curr_peak
                    best_A = list(curr_A)
                    best_reps = curr_reps
                continue
            
            # Strategy 2: Try swapping an element with another to reduce peak
            # Focus on elements that appear in many peak-sum representations
            for x in list(curr_A):
                if time.time() > deadline - 5:
                    break
                
                # Try swapping x with some other value
                for new_x in random.sample(range(MAX_VAL + 1), min(20, MAX_VAL + 1 - len(curr_A))):
                    if new_x == x or new_x in curr_A:
                        continue
                    
                    test_A = curr_A - {x}
                    test_A.add(new_x)
                    test_reps = calc_reps(test_A)
                    
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak < curr_peak:
                            curr_A = test_A
                            curr_reps = test_reps
                            curr_peak = test_peak
                            improved_this_iter = True
                            break
                        elif test_peak == curr_peak:
                            # Accept if it doesn't increase peak
                            curr_A = test_A
                            curr_reps = test_reps
                            improved_this_iter = True
                            break
                
                if improved_this_iter:
                    if curr_peak < best_peak:
                        best_peak = curr_peak
                        best_A = list(curr_A)
                        best_reps = curr_reps
                    break
            
            if not improved_this_iter:
                # Try adding an element to reduce peak (counterintuitive but possible)
                for new_x in random.sample(range(MAX_VAL + 1), min(10, MAX_VAL + 1 - len(curr_A))):
                    if new_x in curr_A:
                        continue
                    
                    test_A = curr_A | {new_x}
                    test_reps = calc_reps(test_A)
                    
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak < curr_peak:
                            curr_A = test_A
                            curr_reps = test_reps
                            curr_peak = test_peak
                            improved_this_iter = True
                            break
                
                if improved_this_iter:
                    if curr_peak < best_peak:
                        best_peak = curr_peak
                        best_A = list(curr_A)
                        best_reps = curr_reps

    # Fallback if no valid set found
    if best_A is None:
        best_A = list(range(64, 129))

    return best_A
