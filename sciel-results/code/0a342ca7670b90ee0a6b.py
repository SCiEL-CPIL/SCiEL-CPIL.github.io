import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    def evaluate(A):
        """
        Evaluates the set A using vectorized operations for speed.
        Returns (peak, is_covered).
        peak: max multiplicity in [128, 256]
        is_covered: True if all sums in [128, 256] have count > 0
        """
        if not A:
            return float('inf'), False
            
        A_arr = np.array(list(A), dtype=np.int32)
        n = len(A_arr)
        
        # We need to count pairs (i, j) with i <= j such that A[i] + A[j] is in [N_LOW, N_HIGH]
        # Vectorized approach:
        # For each i, consider sums A[i] + A[i:].
        # This is O(n^2) in Python if done naively, but we can use broadcasting or loops with numpy.
        # The inspiration used a loop over i and np.add.at. This is O(n^2) but with low constant factor.
        
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Optimization: If n is small, this is fast. If n is large, it might be slow.
        # n is expected to be 30-60. 60^2 = 3600 ops. Very fast.
        
        for i in range(n):
            ai = A_arr[i]
            # Sums with current element and all subsequent elements (including itself)
            # A_arr[i:] contains A[i], A[i+1], ...
            sums = ai + A_arr[i:]
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            valid_sums = sums[mask]
            if len(valid_sums) > 0:
                indices = valid_sums - N_LOW
                np.add.at(counts, indices, 1)
                
        if not np.all(counts > 0):
            return float('inf'), False
            
        return int(np.max(counts)), True

    def greedy_repair(A):
        """
        Attempts to make A valid by adding elements if coverage is missing.
        Returns (repaired_A, is_valid)
        """
        A_set = set(A)
        current_peak, is_covered = evaluate(A_set)
        
        if is_covered:
            return A_set, True
            
        # Find missing sums
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_arr = np.array(list(A_set), dtype=np.int32)
        n = len(A_arr)
        for i in range(n):
            ai = A_arr[i]
            sums = ai + A_arr[i:]
            mask = (sums >= N_LOW) & (sums <= N_HIGH)
            valid_sums = sums[mask]
            if len(valid_sums) > 0:
                indices = valid_sums - N_LOW
                np.add.at(counts, indices, 1)
                
        missing_sums = [s + N_LOW for s in range(RANGE_SIZE) if counts[s] == 0]
        
        # Try to add elements to cover missing sums
        # Heuristic: For a missing sum S, we need a+b=S.
        # If we add x, we cover x+y for all y in A.
        # We can try adding random elements and check if coverage improves.
        
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if not missing_sums:
                break
            
            # Pick a random missing sum
            target_sum = random.choice(missing_sums)
            
            # Try to find an element x such that adding x helps cover target_sum
            # Ideally, we want x such that target_sum - x is in A.
            # So x = target_sum - y for some y in A.
            # We can pick a random y in A, compute x = target_sum - y, and add x if valid.
            
            if not A_set:
                A_set.add(0)
                continue
                
            y = random.choice(list(A_set))
            x = target_sum - y
            
            if 0 <= x <= MAX_VAL and x not in A_set:
                A_set.add(x)
                # Re-evaluate to see if we covered more
                new_peak, new_covered = evaluate(A_set)
                if new_covered:
                    return A_set, True
                # Update missing sums list
                counts = np.zeros(RANGE_SIZE, dtype=np.int32)
                A_arr = np.array(list(A_set), dtype=np.int32)
                n = len(A_arr)
                for i in range(n):
                    ai = A_arr[i]
                    sums = ai + A_arr[i:]
                    mask = (sums >= N_LOW) & (sums <= N_HIGH)
                    valid_sums = sums[mask]
                    if len(valid_sums) > 0:
                        indices = valid_sums - N_LOW
                        np.add.at(counts, indices, 1)
                missing_sums = [s + N_LOW for s in range(RANGE_SIZE) if counts[s] == 0]
            else:
                # If x is out of bounds or already in set, try another y
                continue

        # Final check
        _, is_covered = evaluate(A_set)
        return A_set, is_covered

    def simulated_annealing(initial_A):
        """
        Runs Simulated Annealing starting from initial_A.
        Returns (best_A, best_peak)
        """
        current_A = set(initial_A)
        current_peak, is_covered = evaluate(current_A)
        
        if not is_covered:
            # Try to repair first
            current_A, is_covered = greedy_repair(current_A)
            if not is_covered:
                return None, float('inf')
            current_peak, _ = evaluate(current_A)
            
        best_A = set(current_A)
        best_peak = current_peak
        
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.1
        
        # Local best tracking
        local_best_peak = current_peak
        local_best_A = set(current_A)
        
        while time.time() < deadline - 5 and T > min_T:
            # Choose move type
            move_type = random.random()
            new_A = set(current_A)
            
            if move_type < 0.7:
                # Swap
                if not new_A:
                    continue
                to_remove = random.choice(list(new_A))
                to_add = random.randint(0, MAX_VAL)
                if to_remove == to_add:
                    continue
                new_A.remove(to_remove)
                new_A.add(to_add)
            elif move_type < 0.85:
                # Add
                if len(new_A) > MAX_VAL:
                    continue
                to_add = random.randint(0, MAX_VAL)
                if to_add in new_A:
                    continue
                new_A.add(to_add)
            else:
                # Remove
                if len(new_A) <= 2:
                    continue
                to_remove = random.choice(list(new_A))
                new_A.remove(to_remove)
                
            new_peak, is_covered_new = evaluate(new_A)
            
            if is_covered_new:
                if new_peak < local_best_peak:
                    local_best_peak = new_peak
                    local_best_A = set(new_A)
                    
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = set(new_A)
                
                # Acceptance criterion
                if current_peak == float('inf'):
                    # Was not covered, now is
                    current_A = new_A
                    current_peak = new_peak
                else:
                    delta = new_peak - current_peak
                    if delta < 0:
                        current_A = new_A
                        current_peak = new_peak
                    else:
                        # Prob of accepting worse
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_peak = new_peak
            else:
                # Not covered
                if is_covered:
                    # Was covered, now not: Reject
                    pass
                else:
                    # Was not covered, still not covered:
                    # We might accept if we are "closer" to covered?
                    # Simple heuristic: Reject.
                    pass
                    
            T *= cooling_rate
            
        # Update global best if local best is better
        if local_best_peak < best_peak:
            best_peak = local_best_peak
            best_A = local_best_A
            
        return best_A, best_peak

    best_peak_global = float('inf')
    best_A_global = None

    # Try several different initial sizes
    # Smaller sets have lower average multiplicity, but harder to cover.
    # Optimal k is likely around 30-50.
    sizes_to_try = [30, 35, 40, 45, 50, 55, 60]
    random.shuffle(sizes_to_try)

    for target_size in sizes_to_try:
        if time.time() > deadline - 10:
            break
            
        # Perform a few restarts for each size
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            A_init = set()
            # Always include 0 and 128 to help with boundaries
            A_init.add(0)
            A_init.add(MAX_VAL)
            
            # Add random elements
            candidates = list(range(1, MAX_VAL))
            random.shuffle(candidates)
            
            for c in candidates:
                if len(A_init) >= target_size:
                    break
                A_init.add(c)
            
            # Run SA
            A_res, peak_res = simulated_annealing(A_init)
            
            if A_res is not None and peak_res < best_peak_global:
                best_peak_global = peak_res
                best_A_global = A_res

    # If no valid A was found, fallback
    if best_A_global is None:
        # Fallback: A set that definitely covers
        # Use a dense set
        best_A_global = set(range(0, MAX_VAL + 1))
        # Try to optimize this quickly?
        # No, just return it. It's valid but high peak.
        # Better fallback: Greedy repair on a random set
        A_fallback = set([0, MAX_VAL])
        for i in range(1, MAX_VAL):
            A_fallback.add(i)
            if time.time() > deadline - 2:
                break
            p, c = evaluate(A_fallback)
            if c:
                break
        best_A_global = A_fallback
        p, c = evaluate(best_A_global)
        if not c:
            best_A_global = set(range(0, MAX_VAL + 1))

    return list(best_A_global)
