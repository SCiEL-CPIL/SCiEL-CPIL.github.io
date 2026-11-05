import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Target interval
    lo, hi = 128, 256
    targets = list(range(lo, hi + 1))
    n_targets = len(targets)
    
    # Universe for A
    universe = list(range(0, 129))
    
    def get_counts(A):
        """
        Returns a dictionary or array of counts for sums in [lo, hi].
        Also returns a boolean if all targets are covered.
        """
        # Use a list for counts, indexed by sum - lo
        counts = [0] * (hi - lo + 1)
        A_set = set(A)
        # Unordered pairs with replacement
        # We can just iterate over sorted A
        As = sorted(A)
        for i in range(len(As)):
            for j in range(i, len(As)):
                s = As[i] + As[j]
                if lo <= s <= hi:
                    counts[s - lo] += 1
        # Check coverage
        covered = all(c > 0 for c in counts)
        peak = max(counts) if covered else 10**9
        return counts, peak, covered

    def initial_random_set(k):
        return random.sample(universe, k)
    
    def repair_set(A):
        """
        Adds random elements to A until it covers all targets.
        """
        A = list(A)
        A_set = set(A)
        while True:
            counts, peak, covered = get_counts(A)
            if covered:
                return A
            # Find a missing target
            missing_indices = [i for i, c in enumerate(counts) if c == 0]
            if not missing_indices:
                return A
            target = missing_indices[0] + lo
            # Try to add an element that helps cover 'target'
            # Ideally, we want a pair (a, b) where a is in A and b is new, or both new.
            # Simplest: just add random elements until covered.
            # More efficient: Pick a random a in A, need b = target - a.
            # If b is in universe and not in A, add it.
            added = False
            for a in A:
                b = target - a
                if 0 <= b <= 128 and b not in A_set:
                    A.append(b)
                    A_set.add(b)
                    added = True
                    break
            if not added:
                # Pick random b
                b = random.choice(universe)
                if b not in A_set:
                    A.append(b)
                    A_set.add(b)
            # Safety: if stuck, add random
            if len(A) > 128:
                break
        return A

    def local_search(A, time_limit):
        """
        Simulated Annealing / Hill Climbing to minimize peak multiplicity.
        """
        A = list(A)
        A_set = set(A)
        counts, peak, covered = get_counts(A)
        
        if not covered:
            A = repair_set(A)
            counts, peak, covered = get_counts(A)
        
        best_peak = peak
        best_A = list(A)
        
        # Temperature for Simulated Annealing
        T = 1.0
        cooling = 0.995
        
        while time.time() < time_limit:
            # Choose a move
            move_type = random.random()
            new_A = list(A)
            new_A_set = set(A)
            
            if move_type < 0.4:
                # Remove an element
                if len(A) > 1:
                    idx = random.randrange(len(A))
                    elem = A[idx]
                    new_A.pop(idx)
                    new_A_set.discard(elem)
            elif move_type < 0.8:
                # Add an element
                elem = random.choice(universe)
                if elem not in new_A_set:
                    new_A.append(elem)
                    new_A_set.add(elem)
            else:
                # Swap an element
                if len(A) > 0:
                    idx = random.randrange(len(A))
                    old_elem = A[idx]
                    new_elem = random.choice(universe)
                    if new_elem not in new_A_set or new_elem == old_elem:
                        new_A[idx] = new_elem
                        new_A_set.discard(old_elem)
                        new_A_set.add(new_elem)
                    else:
                        # Skip swap if new_elem already exists
                        continue

            new_counts, new_peak, new_covered = get_counts(new_A)
            
            # If not covered, penalty is huge
            if not new_covered:
                new_peak = 10**9
            
            delta = new_peak - peak
            
            if delta < 0 or (delta == 0 and len(new_A) < len(A)):
                # Accept improvement
                A = new_A
                A_set = new_A_set
                counts = new_counts
                peak = new_peak
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            else:
                # Accept worse with probability exp(-delta/T)
                # Only if delta is small
                if delta < 10: 
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        A = new_A
                        A_set = new_A_set
                        counts = new_counts
                        peak = new_peak
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_A, best_peak

    best_overall_peak = 10**9
    best_overall_A = list(range(64, 129)) # Fallback to seed
    
    # Try different initial sizes
    for k in [20, 30, 40, 50]:
        if time.time() > deadline:
            break
            
        # Run multiple restarts for each k
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            # Random initial set
            A_init = initial_random_set(k)
            # Repair it
            A_repaired = repair_set(A_init)
            
            # Local search with a sub-deadline
            # Allow ~5-10s per local search run
            ls_deadline = time.time() + 5.0
            if ls_deadline > deadline:
                ls_deadline = deadline - 1.0
            
            A_opt, peak_opt = local_search(A_repaired, ls_deadline)
            
            if peak_opt < best_overall_peak:
                best_overall_peak = peak_opt
                best_overall_A = A_opt
                # If we find peak 1 or 2, we can stop early as it's likely optimal or near-optimal
                if best_overall_peak <= 2:
                    return best_overall_A

    # If we haven't found anything better than seed, return seed
    # But check if best_overall_A is valid
    _, _, is_valid = get_counts(best_overall_A)
    if not is_valid:
        return list(range(64, 129))
        
    return best_overall_A

