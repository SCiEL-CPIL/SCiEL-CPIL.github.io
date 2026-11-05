import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute: for each possible element value v in [0,128],
    # and for each target sum s in [128,256], which partner p = s - v
    # is valid (0 <= p <= 128, and for unordered pairs with a<=b, we need v <= p)
    # Actually, let's think differently.
    
    # For a set A (as a boolean array of length 129), the multiplicity of sum s is:
    # count of pairs (a,b) with a <= b, a+b=s, a in A, b in A
    # = count of a in A such that a <= s-a, s-a in A, 0<=a<=128, 0<=s-a<=128
    # = count of a in A such that max(0, s-128) <= a <= min(128, s//2) and (s-a) in A
    
    # Let's precompute for each target s, the list of (a, b) pairs where a <= b, a+b=s
    pair_lists = {}
    for s in range(target_lo, target_hi + 1):
        pairs = []
        a_min = max(0, s - n)
        a_max = min(n, s // 2)
        for a in range(a_min, a_max + 1):
            b = s - a
            pairs.append((a, b))
        pair_lists[s] = pairs
    
    def compute_multiplicities(A_set):
        """Compute multiplicity for each target sum given A as a set."""
        mults = np.zeros(num_targets, dtype=np.int32)
        for idx, s in enumerate(range(target_lo, target_hi + 1)):
            count = 0
            for a, b in pair_lists[s]:
                if a in A_set and b in A_set:
                    count += 1
            mults[idx] = count
        return mults
    
    def is_covered(mults):
        return np.all(mults > 0)
    
    def peak(mults):
        return int(mults.max())
    
    # Initial solution: A = [64, 128]
    A = set(range(64, 129))
    mults = compute_multiplicities(A)
    best_peak = peak(mults)
    best_A = set(A)
    
    if not is_covered(mults):
        # Shouldn't happen for [64,128], but just in case
        pass
    
    # Local search
    random.seed(42)
    current_A = set(A)
    current_mults = mults.copy()
    current_peak = best_peak
    
    all_elements = list(range(0, 129))
    
    # Helper to update multiplicities incrementally when adding/removing an element
    def get_affected_sums(v):
        """Get list of (s_idx, a, b, partner) for sums affected by element v."""
        affected = []
        for s in range(target_lo, target_hi + 1):
            s_idx = s - target_lo
            a_min = max(0, s - n)
            a_max = min(n, s // 2)
            # v participates if v <= s-v (i.e., v <= s//2) and s-v in [0,128]
            # or if s-v <= v (i.e., s-v < v, but we only count a<=b so a=min(v,s-v))
            # Actually, v is in pair (a,b) with a<=b if:
            #   v == a and v <= b, or v == b and a <= v
            # Since a <= b and a+b=s:
            #   v is in the pair if: max(0,s-128) <= min(v, s-v) <= min(128, s//2)
            #   and the pair is (min(v,s-v), max(v,s-v))
            if v <= s - v:  # v is the smaller element
                a = v
                b = s - v
                if 0 <= a <= 128 and 0 <= b <= 128 and a <= b:
                    affected.append((s_idx, a, b))
            elif s - v < v:  # v is the larger element, partner is s-v
                a = s - v
                b = v
                if 0 <= a <= 128 and 0 <= b <= 128 and a <= b:
                    affected.append((s_idx, a, b))
        return affected
    
    # Precompute affected sums for each element
    affected_by_element = {}
    for v in range(129):
        affected_by_element[v] = get_affected_sums(v)
    
    def add_element(v, A_set, mults):
        """Add element v to A_set, update mults. Returns new mults."""
        A_set.add(v)
        new_mults = mults.copy()
        for s_idx, a, b in affected_by_element[v]:
            # Check if the OTHER element is in A_set
            other = b if a == v else a
            if other in A_set:
                new_mults[s_idx] += 1
        return new_mults
    
    def remove_element(v, A_set, mults):
        """Remove element v from A_set, update mults. Returns new mults."""
        A_set.remove(v)
        new_mults = mults.copy()
        for s_idx, a, b in affected_by_element[v]:
            other = b if a == v else a
            if other in A_set:
                new_mults[s_idx] -= 1
        return new_mults
    
    # Simulated annealing / local search
    temp = 5.0
    cooling = 0.995
    iterations = 0
    
    while time.time() < deadline and temp > 0.1:
        # Choose a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Add an element
            candidates = [v for v in all_elements if v not in current_A]
            if not candidates:
                temp *= cooling
                continue
            v = random.choice(candidates)
            new_A = set(current_A)
            new_mults = add_element(v, new_A, current_mults)
            new_peak = peak(new_mults)
            covered = is_covered(new_mults)
            if covered:
                delta_peak = new_peak - current_peak
                if delta_peak <= 0 or random.random() < np.exp(-delta_peak / temp):
                    current_A = new_A
                    current_mults = new_mults
                    current_peak = new_peak
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(current_A)
            else:
                # Not covered, penalize heavily
                pass
        elif move_type < 0.6:
            # Remove an element
            candidates = list(current_A)
            if not candidates:
                temp *= cooling
                continue
            v = random.choice(candidates)
            new_A = set(current_A)
            new_mults = remove_element(v, new_A, current_mults)
            new_peak = peak(new_mults)
            covered = is_covered(new_mults)
            if covered:
                delta_peak = new_peak - current_peak
                if delta_peak <= 0 or random.random() < np.exp(-delta_peak / temp):
                    current_A = new_A
                    current_mults = new_mults
                    current_peak = new_peak
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(current_A)
            else:
                pass
        else:
            # Swap: remove one, add another
            if len(current_A) < 2:
                temp *= cooling
                continue
            v_remove = random.choice(list(current_A))
            candidates_add = [v for v in all_elements if v not in current_A]
            if not candidates_add:
                temp *= cooling
                continue
            v_add = random.choice(candidates_add)
            
            new_A = set(current_A)
            new_mults = current_mults.copy()
            # Remove v_remove
            for s_idx, a, b in affected_by_element[v_remove]:
                other = b if a == v_remove else a
                if other in new_A:
                    new_mults[s_idx] -= 1
            new_A.remove(v_remove)
            # Add v_add
            for s_idx, a, b in affected_by_element[v_add]:
                other = b if a == v_add else a
                if other in new_A:
                    new_mults[s_idx] += 1
            new_A.add(v_add)
            
            new_peak = peak(new_mults)
            covered = is_covered(new_mults)
            if covered:
                delta_peak = new_peak - current_peak
                if delta_peak <= 0 or random.random() < np.exp(-delta_peak / temp):
                    current_A = new_A
                    current_mults = new_mults
                    current_peak = new_peak
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(current_A)
            else:
                pass
        
        iterations += 1
        temp *= cooling
    
    # Try a few more targeted improvements
    # Try removing elements that don't contribute to coverage
    for v in list(best_A):
        if time.time() > deadline - 5:
            break
        test_A = set(best_A)
        test_mults = compute_multiplicities(test_A)
        # Try removing v
        test_A.remove(v)
        test_mults2 = compute_multiplicities(test_A)
        if is_covered(test_mults2) and peak(test_mults2) <= best_peak:
            if peak(test_mults2) < best_peak:
                best_peak = peak(test_mults2)
                best_A = test_A
    
    # Final check
    final_mults = compute_multiplicities(best_A)
    if not is_covered(final_mults):
        # Fallback to seed
        return list(range(64, 129))
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
