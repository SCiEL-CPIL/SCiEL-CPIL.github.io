import time
import random
import numpy as np

def _compute_peak_and_cov(counts):
    """
    counts: np.array of length 129 (for sums 128..256)
    Returns: peak (int), coverage_ok (bool)
    """
    peak = int(counts.max())
    coverage_ok = bool((counts > 0).all())
    return peak, coverage_ok

def _init_valid_set():
    """
    Greedily construct a set A that covers [128, 256].
    """
    A = set()
    # Start with some structure
    # Try adding elements that cover the most uncovered sums
    counts = np.zeros(129, dtype=np.int32)
    
    # We want to cover [128, 256].
    # Uncovered sums are those with count 0.
    
    # Heuristic: Add elements symmetrically or in blocks.
    # Let's try a random restart of greedy addition until covered.
    for _ in range(20): # 20 random initializations
        A_curr = set()
        # Add some random start points
        for _ in range(5):
            A_curr.add(random.randint(0, 128))
        
        # Greedy addition
        for _ in range(100): # Max 100 additions
            # Check if covered
            # Efficient check: just compute counts from scratch for this small set?
            # Or maintain incremental? For init, scratch is fine if set is small.
            A_list = sorted(A_curr)
            counts = np.zeros(129, dtype=np.int32)
            for i in range(len(A_list)):
                a = A_list[i]
                for j in range(i, len(A_list)):
                    s = a + A_list[j]
                    if 128 <= s <= 256:
                        counts[s - 128] += 1
            
            if (counts > 0).all():
                return A_curr
            
            # Find best candidate to add
            # We want to maximize coverage of currently 0-count sums.
            best_cand = -1
            best_gain = 0
            # Sample candidates to speed up
            candidates = random.sample(range(129), min(50, 129 - len(A_curr)))
            for cand in candidates:
                if cand in A_curr:
                    continue
                gain = 0
                # Check which uncovered sums this cand would cover
                # Sum s is covered if s-cand in A_curr.
                for idx in range(129):
                    if counts[idx] == 0:
                        target_sum = idx + 128
                        partner = target_sum - cand
                        if 0 <= partner <= 128 and partner in A_curr:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand != -1:
                A_curr.add(best_cand)
            else:
                # Stuck, add random
                A_curr.add(random.randint(0, 128))
                
    # Fallback: Return a dense set
    return set(range(0, 129, 2)) # Even numbers, size 65

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 55s limit to be safe
    
    # 1. Get an initial valid set
    A_init = _init_valid_set()
    
    # 2. Simulated Annealing
    # State: in_A (bool array), counts (int array)
    in_A = np.zeros(129, dtype=bool)
    for x in A_init:
        in_A[x] = True
    
    # Compute initial counts
    counts = np.zeros(129, dtype=np.int32)
    A_list = [i for i in range(129) if in_A[i]]
    for i in range(len(A_list)):
        a = A_list[i]
        for j in range(i, len(A_list)):
            s = a + A_list[j]
            if 128 <= s <= 256:
                counts[s - 128] += 1
                
    peak, cov_ok = _compute_peak_and_cov(counts)
    best_peak = peak
    best_in_A = in_A.copy()
    
    # Track peak frequency for O(1) peak update?
    # Simpler: just recompute peak if we suspect it changed, or use a histogram.
    # Let's use a histogram of counts.
    # hist[c] = number of sums with count c.
    # Max count is bounded by |A|.
    max_possible_count = 129
    hist = np.zeros(max_possible_count + 1, dtype=np.int32)
    for c in counts:
        hist[c] += 1
    
    current_peak = peak
    
    def get_current_peak():
        # Find max c such that hist[c] > 0
        # Since counts are small, we can scan from top
        # Or just keep current_peak and update
        return current_peak

    def update_count(s_idx, delta):
        """
        Update counts[s_idx] by delta.
        s_idx is 0..128 corresponding to sum 128..256.
        """
        nonlocal current_peak
        old_c = counts[s_idx]
        new_c = old_c + delta
        
        # Update hist
        hist[old_c] -= 1
        hist[new_c] += 1
        
        counts[s_idx] = new_c
        
        # Update current_peak
        if new_c > current_peak:
            current_peak = new_c
        elif old_c == current_peak and hist[old_c] == 0:
            # The peak decreased, find new max
            # Scan down from current_peak - 1
            # Optimization: only scan if necessary
            while current_peak > 0 and hist[current_peak] == 0:
                current_peak -= 1

    def apply_swap(x, y):
        """
        Remove x, Add y.
        x must be in A, y must not be in A.
        Returns delta_peak (new_peak - old_peak) and delta_cov (new_uncovered - old_uncovered)
        Actually, we just update state and return the new cost.
        """
        nonlocal current_peak
        
        # 1. Remove x
        in_A[x] = False
        # Update counts for sums involving x
        # Pairs (x, z) where z in A (z != x)
        # Since we are removing x, we subtract 1 from count(x+z)
        # Note: (x, x) is also removed if x was in A.
        # Our counts include (a, b) with a <= b.
        # When we remove x:
        # - (x, x) sum 2x
        # - (x, z) for z in A, z > x
        # - (z, x) for z in A, z < x -> this is (z, x) which is counted as (min(z,x), max(z,x))
        
        # Let's iterate all z in A (excluding x)
        # And handle (x, x) separately?
        # Easier: Iterate all z in A (current state before removal, so includes x).
        # For each z:
        #   if z == x: sum 2x
        #   else: sum x+z
        #   decrement count(sum)
        
        # To be safe and simple, iterate all z in A_list (dynamic list?)
        # We don't have a dynamic list easily.
        # We can iterate all 0..128 and check in_A[z].
        
        for z in range(129):
            if in_A[z]:
                s = x + z
                if 128 <= s <= 256:
                    update_count(s - 128, -1)
        
        # 2. Add y
        in_A[y] = True
        # Update counts for sums involving y
        # Iterate all z in A (now includes y)
        for z in range(129):
            if in_A[z]:
                s = y + z
                if 128 <= s <= 256:
                    update_count(s - 128, 1)

    # Main SA Loop
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # We need to track coverage count (number of zeros in counts)
    # Initial coverage count
    uncov_count = int(np.sum(counts == 0))
    
    def get_cost():
        # Penalize uncovered sums heavily
        return current_peak + 100 * uncov_count

    current_cost = get_cost()
    best_cost = current_cost
    
    # Precompute list of in_A elements for faster swap selection?
    # We can just pick random x from 0..128 where in_A[x], and random y where not.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick x to remove
        # Find a random element in A
        # Simple: pick random index, check if in_A. Repeat.
        # Or maintain a list.
        # Given k~50, finding a random element is fast.
        x = random.randint(0, 128)
        while not in_A[x]:
            x = random.randint(0, 128)
            
        # Pick y to add
        y = random.randint(0, 128)
        while in_A[y]:
            y = random.randint(0, 128)
            
        if x == y:
            continue
            
        # Apply swap
        old_peak = current_peak
        old_uncov = uncov_count
        
        # We need to update uncov_count during update_count
        # Let's modify update_count to track uncov_count
        
        # Re-define update_count to handle uncov_count
        # This requires nonlocal uncov_count
        
        # Let's refactor update_count to return delta_uncov?
        # Or just recompute uncov_count after swap?
        # Recomputing is O(129). Swap is O(129) for the loop.
        # So recomputing is fine.
        
        # But we need the cost to decide acceptance.
        # So we must compute new_peak and new_uncov.
        
        # Let's do the swap, compute new metrics, and if rejected, revert.
        
        # Save old state
        old_in_A = in_A.copy()
        old_counts = counts.copy()
        old_hist = hist.copy()
        old_peak = current_peak
        
        # Perform swap
        # Remove x
        in_A[x] = False
        for z in range(129):
            if in_A[z]:
                s = x + z
                if 128 <= s <= 256:
                    # Update count
                    idx = s - 128
                    old_c = counts[idx]
                    new_c = old_c - 1
                    hist[old_c] -= 1
                    hist[new_c] += 1
                    counts[idx] = new_c
                    if new_c > current_peak:
                        current_peak = new_c
                    elif old_c == current_peak and hist[old_c] == 0:
                        while current_peak > 0 and hist[current_peak] == 0:
                            current_peak -= 1

        # Add y
        in_A[y] = True
        for z in range(129):
            if in_A[z]:
                s = y + z
                if 128 <= s <= 256:
                    idx = s - 128
                    old_c = counts[idx]
                    new_c = old_c + 1
                    hist[old_c] -= 1
                    hist[new_c] += 1
                    counts[idx] = new_c
                    if new_c > current_peak:
                        current_peak = new_c
                    elif old_c == current_peak and hist[old_c] == 0:
                        while current_peak > 0 and hist[current_peak] == 0:
                            current_peak -= 1

        new_peak = current_peak
        new_uncov = int(np.sum(counts == 0))
        new_cost = new_peak + 100 * new_uncov
        
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if T > 0:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_cost = new_cost
            uncov_count = new_uncov
            if new_cost < best_cost:
                best_cost = new_cost
                best_peak = new_peak
                best_in_A = in_A.copy()
        else:
            # Revert
            in_A[:] = old_in_A
            counts[:] = old_counts
            hist[:] = old_hist
            current_peak = old_peak
            # uncov_count is derived from counts, so it reverts automatically
            # But we stored uncov_count variable, so revert that too if we used it in cost
            # Actually, we just use new_uncov in cost calc.
            # So we just need to revert in_A, counts, hist, current_peak.
            
        T *= cooling
        
    # Extract best set
    best_set = [i for i in range(129) if best_in_A[i]]
    return sorted(best_set)
