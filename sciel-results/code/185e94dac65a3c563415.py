import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    min_sum = 128
    max_sum = 256
    num_targets = max_sum - min_sum + 1  # 129 targets

    # For each target sum s in [128, 256], the possible unordered pairs (a, b) with a <= b, a+b=s, a,b in [0,128]
    # Precompute all valid pairs for each target
    pairs_for_target = []
    for s in range(min_sum, max_sum + 1):
        pairs = []
        for a in range(0, 129):
            b = s - a
            if b < a:
                break
            if 0 <= a <= 128 and 0 <= b <= 128:
                pairs.append((a, b))
        pairs_for_target.append(pairs)

    # Build a quick lookup: for each element x in [0,128], which targets can use it (and in what role)
    # element_usage[x] = list of (target_idx, pair_idx_in_target)
    # But that's complex. Instead, let's think differently.

    # Strategy: CSP with backtracking.
    # Variables: target sums s in [128, 256]
    # Each variable assigned a pair (a,b) from A (but A is also being built)
    # Actually, this is a joint problem: we need to find both A and the assignment.
    
    # Alternative approach: 
    # 1. Start with a lower bound on K (max multiplicity)
    # 2. Try to find A and assignment where no element is used in more than K pairs
    # 3. Increment K until found

    # But what's the lower bound? 
    # Total pairs needed: 129 (one per target, minimum)
    # Each element can appear in at most K pairs
    # 129 elements, each in at most K pairs -> max 129*K pair slots
    # But each pair uses 2 elements (or 1 if a=b)
    # Actually, we need at least 129 pairs (one per target), each using 2 element-slots
    # So 2*129 = 258 element-uses needed
    # With 129 elements each used at most K times: 129*K >= 258 -> K >= 2
    # But that's very loose. Let's think more carefully.

    # Actually, the number of elements in A matters too. If A is small, K must be larger.
    # If A is large (close to 129), K can be small.

    # Let me try a different structural approach:
    # Greedy set construction with multiplicity tracking, then local refinement.

    # Start with A = [0, 128] (all elements), which trivially covers everything.
    # Then try to remove elements while maintaining coverage and tracking peak multiplicity.

    # Fast coverage check using numpy convolution
    def compute_counts(A_set):
        """Return array of unordered pair counts for sums in [128, 256]."""
        x_bin = np.zeros(129, dtype=np.int8)
        for x in A_set:
            x_bin[x] = 1
        conv = np.convolve(x_bin, x_bin, mode='full')
        # conv[s] = number of ordered pairs (a,b) in A with a+b=s
        # Unordered count for sum s: 
        # If s is even and s/2 in A: unordered = (conv[s] - 1) / 2 + 1 = (conv[s] + 1) / 2
        # If s is odd: unordered = conv[s] / 2
        # Actually: conv[s] = sum_{a+b=s} x[a]*x[b]
        # For unordered pairs {a,b} with a<=b:
        # count = number of pairs (a,b) with a<=b, a+b=s
        # = (conv[s] + diag[s]) / 2 where diag[s] = 1 if s even and s/2 in A, else 0
        counts = np.zeros(num_targets, dtype=np.float64)
        for idx, s in enumerate(range(min_sum, max_sum + 1)):
            diag = 1.0 if (s % 2 == 0 and s // 2 in A_set) else 0.0
            counts[idx] = (conv[s] + diag) / 2.0
        return np.round(counts).astype(np.int32)

    # Phase 1: Start with full set, greedily remove elements
    A_current = set(range(0, 129))
    counts = compute_counts(A_current)
    peak = int(np.max(counts))
    
    # Track multiplicity of each element: how many pairs (a,b) in the current "covering" use element a
    # This is tricky because we need to track which pairs are actually used for coverage.
    # Let's use a different approach: iterative removal with coverage check.

    # Build a set of elements to consider for removal
    # Elements that appear in fewer pairs are safer to remove
    # Compute element usage: for each x, how many pairs (a,b) with a<=b, a+b in [128,256] include x
    # This is an upper bound on how many targets x could be needed for.

    # Let's just do greedy: try removing each element, check if coverage still holds, if yes remove it.
    # Sort elements by "removability" - elements that are in fewer potential pairs.

    # For each x in [0,128], count how many targets s in [128,256] have a valid pair (a,b) with a+b=s that includes x.
    # If x is removed, targets that ONLY had pairs involving x become uncovered.
    
    # Precompute: for each target s, list of pairs. Then for each x, which targets can still be covered without x.
    
    # This is getting complex. Let me use a simpler heuristic:
    # Try removing elements in order of their "degree" (number of pairs they participate in for target sums).

    def element_degree(x, A_set):
        """Count how many target sums can be formed using x and another element in A_set."""
        count = 0
        for s in range(max(128, x), min(256, x + 128) + 1):
            y = s - x
            if 0 <= y <= 128 and y in A_set and y >= x:
                count += 1
        return count

    # Greedy removal phase
    removal_order = sorted(range(0, 129), key=lambda x: element_degree(x, A_current))
    
    for x in removal_order:
        if time.time() > deadline - 30:
            break
        if x not in A_current:
            continue
        A_test = A_current - {x}
        counts_test = compute_counts(A_test)
        if np.min(counts_test) > 0:
            # Coverage still holds
            peak_test = int(np.max(counts_test))
            if peak_test <= peak:
                A_current = A_test
                counts = counts_test
                peak = peak_test

    # Phase 2: Local search to reduce peak
    # Try swaps: remove one element, add another, check if peak decreases
    x_best = np.zeros(129, dtype=np.int8)
    for x in A_current:
        x_best[x] = 1
    
    best_peak = peak
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        
        # Try all possible single-element swaps
        ones = [x for x in range(129) if x in A_current]
        zeros = [x for x in range(129) if x not in A_current]
        
        if not zeros:
            break
        
        # For each element to remove, find the best element to add
        for rem in ones:
            if time.time() > deadline - 10:
                break
            if len(A_current) <= 10:
                break
            
            A_without = A_current - {rem}
            counts_without = compute_counts(A_without)
            
            # Find which targets are now at minimum coverage (most vulnerable)
            if np.min(counts_without) < 0:
                continue
            
            # Try adding each zero element
            best_add = None
            best_new_peak = best_peak
            
            for add in zeros:
                A_trial = A_without | {add}
                counts_trial = compute_counts(A_trial)
                if np.min(counts_trial) > 0:
                    trial_peak = int(np.max(counts_trial))
                    if trial_peak < best_new_peak:
                        best_new_peak = trial_peak
                        best_add = add
            
            if best_add is not None:
                A_current = A_without | {best_add}
                peak = best_new_peak
                best_peak = peak
                improved = True
                no_improve = 0
                # Restart the loop with new sets
                break
        
        if not improved:
            no_improve += 1
        
        # Update ones and zeros
        ones = [x for x in range(129) if x in A_current]
        zeros = [x for x in range(129) if x not in A_current]
    
    final_A = sorted(A_current)
    return final_A
