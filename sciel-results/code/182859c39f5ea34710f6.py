"""Improved program: iterative reweighting / load-balancing for additive bases."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    target_low = 128
    target_high = 256
    target_len = target_high - target_low + 1  # 129 values
    
    # We need to find A ⊆ [0, 128] such that every s in [128, 256] is a sum of two elements of A
    # and we minimize the maximum multiplicity (number of unordered pairs {a,b} with a+b=s, a<=b, a,b in A)
    
    # Strategy: Start with a reasonable initial set, then iteratively adjust
    # Use the fact that if we take A = [0, 64] ∪ [64, 128] = [0,128], every sum in [0,256] is covered
    # but multiplicities are high. We want a smaller/sparser set.
    
    # Key insight: If A contains both x and 128-x for many x, we get good coverage.
    # A symmetric set around 64 works well.
    
    # Let's try a structured approach:
    # 1. Start with a random feasible set
    # 2. Use local search to minimize peak multiplicity
    
    def compute_multiplicities(A_set):
        """Compute the number of unordered pairs (a,b) with a<=b, a,b in A_set, a+b=s for s in [128,256]."""
        mult = np.zeros(target_len, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if target_low <= s <= target_high:
                    mult[s - target_low] += 1
        return mult
    
    def is_feasible(mult):
        """Check if all target sums are covered."""
        return np.all(mult > 0)
    
    def get_peak(mult):
        """Get the maximum multiplicity."""
        return int(np.max(mult))
    
    # Initial solution: use a symmetric set
    # Let's start with A = {0, 1, ..., 64} ∪ {64, 65, ..., 128} but that's just [0,128]
    # Better: start with a smaller set. 
    # If we take A = [0, 64], then sums range from 0 to 128. Not enough.
    # If we take A = [64, 128], sums range from 128 to 256. Perfect coverage!
    # Let's check multiplicities for A = [64, 128]
    
    A_init = list(range(64, 129))
    mult_init = compute_multiplicities(set(A_init))
    
    # For A = [64, 128], sum s = 128 + k where k in [0, 128]
    # Number of pairs (a, b) with a, b in [64, 128], a <= b, a+b = 128+k
    # Let a = 64 + i, b = 64 + j, i, j in [0, 64], i <= j
    # i + j = k
    # Number of pairs: for k <= 64: floor(k/2) + 1... wait let me think
    # i + j = k, 0 <= i <= j <= 64
    # i ranges from 0 to k//2, so count = k//2 + 1 for k <= 64
    # For k > 64: i ranges from k-64 to 64, but i <= j means i <= k-i, so i <= k/2
    # i ranges from max(0, k-64) to min(64, k//2)
    # For k in [65, 128]: i from k-64 to 64 (since k//2 < 64 when k < 128, and at k=128, i=64)
    # Actually for k in [65, 127]: i from k-64 to 63 (since i <= k//2 and k//2 < 64)
    # Count = 64 - (k-64) = 128 - k
    # For k = 128: i = 64, count = 1
    
    # So the peak is at k=64: count = 64//2 + 1 = 33. That's very high.
    
    # We need a much better set. Let's think about this differently.
    
    # A good approach: use a "difference set" or "Golomb ruler" like structure.
    # Or use random initialization + local search.
    
    # Let's try: A should be such that it's a basis of order 2 for [128, 256].
    # Minimum size: each element in A can contribute to at most 65 sums (with itself and 64 others)
    # Actually each element a in A contributes to sums a+0, a+1, ..., a+128 but only those in [128,256]
    # and only with other elements of A.
    
    # Let's just do random search with local improvement.
    
    def random_init():
        """Generate a random feasible set."""
        # Start with a base that guarantees coverage, then try to remove elements
        # A = [64, 128] always works. Let's start there and try to remove elements.
        A = set(range(64, 129))
        return A
    
    def try_remove(A_set):
        """Try to remove an element from A_set while maintaining feasibility."""
        A_list = sorted(A_set)
        random.shuffle(A_list)
        for a in A_list:
            new_A = A_set - {a}
            if len(new_A) < 2:
                continue
            mult = compute_multiplicities(new_A)
            if is_feasible(mult):
                return new_A, a
        return A_set, None
    
    def try_add(A_set):
        """Try to add an element to A_set."""
        candidates = [x for x in range(n+1) if x not in A_set]
        random.shuffle(candidates)
        for a in candidates:
            new_A = A_set | {a}
            mult = compute_multiplicities(new_A)
            if is_feasible(mult):
                return new_A, a
        return A_set, None
    
    def local_search(A_set, max_iter=5000):
        """Local search: try to minimize peak multiplicity by swapping elements."""
        mult = compute_multiplicities(A_set)
        if not is_feasible(mult):
            return A_set
        current_peak = get_peak(mult)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try removing an element
            A_list = sorted(A_set)
            random.shuffle(A_list)
            for a in A_list:
                new_A = A_set - {a}
                if len(new_A) < 2:
                    continue
                mult_new = compute_multiplicities(new_A)
                if is_feasible(mult_new):
                    peak_new = get_peak(mult_new)
                    if peak_new < current_peak:
                        A_set = new_A
                        mult = mult_new
                        current_peak = peak_new
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            A_list = sorted(A_set)
            random.shuffle(A_list)
            for a_remove in A_list:
                if time.time() > deadline - 5:
                    return A_set
                base_A = A_set - {a_remove}
                # Try adding different elements
                candidates = [x for x in range(n+1) if x not in base_A]
                random.shuffle(candidates)
                for a_add in candidates[:50]:  # Limit to keep it fast
                    new_A = base_A | {a_add}
                    mult_new = compute_multiplicities(new_A)
                    if is_feasible(mult_new):
                        peak_new = get_peak(mult_new)
                        if peak_new < current_peak:
                            A_set = new_A
                            mult = mult_new
                            current_peak = peak_new
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try a random restart within the neighborhood
                # Just do a few random swaps
                for _ in range(10):
                    if time.time() > deadline - 5:
                        return A_set
                    a_remove = random.choice(sorted(A_set))
                    base_A = A_set - {a_remove}
                    candidates = [x for x in range(n+1) if x not in base_A]
                    a_add = random.choice(candidates)
                    new_A = base_A | {a_add}
                    mult_new = compute_multiplicities(new_A)
                    if is_feasible(mult_new):
                        peak_new = get_peak(mult_new)
                        if peak_new <= current_peak:
                            A_set = new_A
                            mult = mult_new
                            current_peak = peak_new
        
        return A_set
    
    # Phase 1: Start with [64, 128] and try to shrink
    A = random_init()
    A = local_search(A, max_iter=2000)
    
    best_peak = get_peak(compute_multiplicities(A))
    best_A = set(A)
    
    # Phase 2: Try multiple random restarts
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initialization: start with a subset of [0, 128] that covers [128, 256]
        # Strategy: pick a random set of size ~80-100 and check if it works, then repair
        A_restart = set()
        
        # Ensure coverage: we need sums to cover [128, 256]
        # A simple approach: include elements that help cover gaps
        # Start with a random set of ~70 elements
        A_restart = set(random.sample(range(n+1), 70))
        
        # Repair: check which sums are missing, add elements to cover them
        for _ in range(100):
            mult = compute_multiplicities(A_restart)
            if is_feasible(mult):
                break
            missing = [s for s in range(target_len) if mult[s] == 0]
            if not missing:
                break
            # For each missing sum, find an element that would help
            # A sum s is missing if no pair (a,b) in A_restart has a+b=s
            # We can add an element x such that for some missing s, s-x is in A_restart
            # Or x + y = s for some y in A_restart
            added = False
            for s in missing[:5]:
                # Find x such that s - x in A_restart and x not in A_restart
                for y in A_restart:
                    x = s - y
                    if 0 <= x <= n and x not in A_restart:
                        A_restart.add(x)
                        added = True
                        break
                if added:
                    break
            if not added:
                # Try adding a random element
                candidates = [x for x in range(n+1) if x not in A_restart]
                if candidates:
                    A_restart.add(random.choice(candidates))
        
        if not is_feasible(compute_multiplicities(A_restart)):
            continue
        
        # Now do local search
        A_restart = local_search(A_restart, max_iter=1000)
        mult = compute_multiplicities(A_restart)
        if is_feasible(mult):
            peak = get_peak(mult)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A_restart)
        
        if best_peak <= 2:
            break
    
    # Final check
    final_mult = compute_multiplicities(best_A)
    if not is_feasible(final_mult):
        # Fallback to the known-good solution
        return list(range(64, 129))
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
