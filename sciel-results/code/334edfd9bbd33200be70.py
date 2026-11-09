import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    DOMAIN = np.arange(MAX_VAL + 1, dtype=np.int32)

    def calc_reps(A):
        """
        Returns array of counts for sums N_LOW..N_HIGH using unordered pairs (i<=j).
        Uses vectorized outer sum and bincount for speed.
        """
        A_arr = np.array(sorted(A), dtype=np.int32)
        n = len(A_arr)
        if n == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Compute all pairwise sums (ordered)
        # sums[i,j] = A[i] + A[j]
        # To get unordered (i<=j), we can take the upper triangle of the sum matrix.
        # However, creating the full n x n matrix is O(n^2) memory and time.
        # For n~50, 50x50=2500 ints, which is fine.
        
        # Alternative: loop over i, add A[i:] to counts.
        # This is O(n^2) ops but avoids large intermediate arrays if n is large.
        # Given n < 129, n^2 is small.
        
        # Let's use the loop method but vectorized inside the loop for each i.
        # This is what the current program does, but let's see if we can optimize.
        # The current program:
        # for i in range(n):
        #   sums = ai + A_arr[i:]
        #   ... bincount ...
        
        # Actually, the "inspiration" suggests using outer sum + triu.
        # Let's try that as it might be faster due to less Python overhead in the loop.
        
        sums = A_arr[:, None] + A_arr[None, :]
        # Extract upper triangle (i <= j)
        # np.triu_indices(n) returns indices for upper triangle
        # This creates arrays of size n*(n+1)/2
        # For n=128, size ~8256. Very fast.
        
        # Get the values
        # Instead of extracting values and then bincounting, can we do it faster?
        # np.bincount on the flattened upper triangle values masked to range.
        
        idx = np.triu_indices(n)
        upper_sums = sums[idx]
        
        mask = (upper_sums >= N_LOW) & (upper_sums <= N_HIGH)
        if not np.any(mask):
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        valid_sums = upper_sums[mask]
        indices = (valid_sums - N_LOW).astype(np.int32)
        counts = np.bincount(indices, minlength=RANGE_SIZE)
        return counts[:RANGE_SIZE].astype(np.int32)

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def element_peak_contribution(x, A, reps):
        """
        Returns the number of sums in [N_LOW, N_HIGH] that are at the peak level
        and involve the element x.
        Note: If x is removed, the count for those sums drops by 1 (if x+x was counted, it drops by 1? No, x+x is one pair. x+y is one pair. So removing x removes all pairs involving x).
        Wait, if x is in A, pairs involving x are (x, x) and (x, y) for y != x.
        In the unordered representation, (x, y) is one pair.
        So removing x decreases the count for sum x+x by 1, and for sum x+y by 1.
        
        We want to identify elements that contribute to the peak.
        """
        A_arr = np.array(sorted(A), dtype=np.int32)
        peak = get_peak(reps)
        
        # Sums involving x
        sums = x + A_arr
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if not np.any(mask):
            return 0
        
        valid_sums = sums[mask]
        indices = valid_sums - N_LOW
        rep_vals = reps[indices]
        
        # Count how many of these sums are at the peak
        return int(np.sum(rep_vals == peak))

    def find_worst_elements(A, reps, k=5):
        """
        Find the k elements with the highest peak contribution.
        """
        peak = get_peak(reps)
        if peak == 0:
            return []
        
        A_list = list(A)
        scores = []
        for x in A_list:
            pc = element_peak_contribution(x, A, reps)
            scores.append((pc, x))
        
        # Sort by peak contribution descending
        scores.sort(reverse=True)
        return [x for _, x in scores[:k]]

    def repair_coverage(A):
        """
        Add elements to A until coverage is achieved.
        Greedy: at each step, pick the element that covers the most missing sums.
        """
        A = set(A)
        reps = calc_reps(A)
        
        max_iter = 100
        for _ in range(max_iter):
            if is_covered(reps):
                break
            
            missing = np.where(reps == 0)[0] + N_LOW
            if len(missing) == 0:
                break
                
            best_gain = -1
            best_elem = None
            
            # Try a subset of candidates to speed up
            # Candidates are 0..128 not in A
            candidates = [x for x in range(129) if x not in A]
            random.shuffle(candidates)
            
            # Evaluate first 20 candidates
            for c in candidates[:20]:
                # Fast check: which missing sums can c cover?
                # c + y = s => y = s - c
                # y must be in A.
                # We can just compute new reps, but that's expensive.
                # Let's approximate gain by checking if c helps with missing sums.
                
                # For each missing sum s, is there y in A such that s - c == y?
                # i.e., is (s - c) in A?
                
                # Vectorized check:
                # missing_sums - c
                # check if in A_set
                # This is O(len(missing))
                
                potential_y = missing - c
                valid_y = (potential_y >= 0) & (potential_y <= 128)
                potential_y = potential_y[valid_y]
                
                # Check how many of these are in A
                # Using set lookup is fast
                count_covered = sum(1 for y in potential_y if y in A)
                
                # Note: this counts how many missing sums are covered by adding c.
                # It doesn't account for new peaks, but for repair, coverage is priority.
                
                if count_covered > best_gain:
                    best_gain = count_covered
                    best_elem = c
                    
            if best_elem is not None and best_gain > 0:
                A.add(best_elem)
                reps = calc_reps(A)
            else:
                # Fallback: add random
                if candidates:
                    A.add(random.choice(candidates))
                else:
                    break
                reps = calc_reps(A)
                
        return A, reps

    best_A = None
    best_peak = float('inf')
    
    def update_best(A, reps):
        nonlocal best_A, best_peak
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)

    def random_init():
        """
        Generate a random initial set.
        """
        A = set()
        # Ensure boundaries
        A.add(0)
        A.add(128)
        
        # Add some structured elements to help coverage
        # Low end: 1, 2, 3...
        # High end: 127, 126, 125...
        # Mid: 64
        
        for x in [1, 2, 3, 125, 126, 127, 64]:
            A.add(x)
            
        # Fill with random
        target_size = random.randint(40, 60)
        while len(A) < target_size:
            A.add(random.randint(0, 128))
            
        return A

    # Main Search Loop
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # 1. Initialize
        current_A = random_init()
        
        # 2. Repair Coverage
        current_A, current_reps = repair_coverage(current_A)
        
        if not is_covered(current_reps):
            continue
            
        update_best(current_A, current_reps)
        
        # 3. Local Search: Peak-aware Greedy Removal + Swap
        # Strategy: 
        # - Try to remove elements that contribute to the peak.
        # - If removal breaks coverage, try swapping.
        # - If swap improves peak or maintains coverage with lower peak, accept.
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        stuck_counter = 0
        max_stuck = 100
        
        while time.time() < deadline - 5 and T > min_T:
            if stuck_counter > max_stuck:
                break
                
            current_peak = get_peak(current_reps)
            
            # Identify elements to try removing
            # Prioritize those with high peak contribution
            worst_elems = find_worst_elements(current_A, current_reps, k=10)
            
            if not worst_elems:
                worst_elems = list(current_A)
                
            # Pick a random element from the worst list (or all elements if list is small)
            # To ensure exploration, sometimes pick a random element from the whole set
            if random.random() < 0.8:
                to_remove = random.choice(worst_elems)
            else:
                to_remove = random.choice(list(current_A))
                
            # Try removal
            new_A = set(current_A)
            new_A.remove(to_remove)
            new_reps = calc_reps(new_A)
            
            accepted = False
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                
                if new_peak < current_peak:
                    # Improvement!
                    current_A = new_A
                    current_reps = new_reps
                    update_best(current_A, current_reps)
                    stuck_counter = 0
                    accepted = True
                elif new_peak == current_peak:
                    # Neutral: accept with some probability
                    if random.random() < 0.5:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                        accepted = True
                else:
                    # Worse: accept with probability exp(-delta/T)
                    delta = new_peak - current_peak
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_A = new_A
                        current_reps = new_reps
                        stuck_counter = 0
                        accepted = True
            else:
                # Removal broke coverage. Try swap.
                # Find an element to add that repairs coverage.
                # We can reuse the repair logic but only for one step.
                
                # Find missing sums
                missing = np.where(new_reps == 0)[0] + N_LOW
                
                best_swap_gain = -1
                best_add = None
                
                candidates = [x for x in range(129) if x not in new_A]
                random.shuffle(candidates)
                
                # Check a few candidates
                for c in candidates[:10]:
                    # Check if c covers any missing sums
                    potential_y = missing - c
                    valid_y = (potential_y >= 0) & (potential_y <= 128)
                    potential_y = potential_y[valid_y]
                    count_covered = sum(1 for y in potential_y if y in new_A)
                    
                    # Also consider peak impact? For now, just coverage repair.
                    if count_covered > best_swap_gain:
                        best_swap_gain = count_covered
                        best_add = c
                        
                if best_add is not None and best_swap_gain > 0:
                    # Perform swap
                    swap_A = set(new_A)
                    swap_A.add(best_add)
                    swap_reps = calc_reps(swap_A)
                    
                    if is_covered(swap_reps):
                        swap_peak = get_peak(swap_reps)
                        
                        if swap_peak < current_peak:
                            current_A = swap_A
                            current_reps = swap_reps
                            update_best(current_A, current_reps)
                            stuck_counter = 0
                            accepted = True
                        elif swap_peak == current_peak:
                            if random.random() < 0.3:
                                current_A = swap_A
                                current_reps = swap_reps
                                update_best(current_A, current_reps)
                                stuck_counter = 0
                                accepted = True
                        else:
                            delta = swap_peak - current_peak
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                current_A = swap_A
                                current_reps = swap_reps
                                stuck_counter = 0
                                accepted = True
                                                    
            if not accepted:
                stuck_counter += 1
                
            T *= cooling

    # Fallback
    if best_A is None:
        # Return a valid set, even if peak is high
        A = set(range(129))
        best_A = list(A)

    return best_A
