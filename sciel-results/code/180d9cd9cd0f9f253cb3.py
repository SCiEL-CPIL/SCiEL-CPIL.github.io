import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a subset A of [0, 128] such that every integer in [128, 256] is a sum
    of two elements of A (unordered pairs, equal summands allowed), minimizing the
    largest representation multiplicity in that target interval.
    
    Strategy: Lagrangian relaxation with sub-gradient updates.
    - Binary variables x_i ∈ {0,1} for each i in [0,128]
    - Objective: minimize peak multiplicity (max over target sums of pair counts)
    - Constraints: coverage (each target sum must have count >= 1)
    - Dualize coverage constraints with multipliers, use sub-gradient updates
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    target_lo, target_hi = 128, 256
    num_targets = target_hi - target_lo + 1  # 129 targets
    
    # Precompute pair contributions for each element
    # For element i, which target sums does it contribute to?
    # If i is in A, then for each j >= i in A, sum i+j contributes
    # We need: for each target sum s, count = sum over pairs (i,j) with i<=j, i+j=s of x_i * x_j
    
    # Build the quadratic form: for each target s, count_s = sum_{i<=j, i+j=s} x_i * x_j
    # This is a sum of products of binary variables
    
    # Precompute for each target s, the list of (i,j) pairs with i<=j, i+j=s
    pair_lists = [None] * num_targets
    for t in range(num_targets):
        s = target_lo + t
        pairs = []
        for i in range(max(0, s - n), min(n, s // 2) + 1):
            j = s - i
            if j <= n and j >= i:
                pairs.append((i, j))
        pair_lists[t] = pairs
    
    def compute_counts(x):
        """Compute representation counts for all target sums given binary vector x."""
        counts = np.zeros(num_targets, dtype=np.int32)
        for t in range(num_targets):
            for (i, j) in pair_lists[t]:
                if x[i] and x[j]:
                    counts[t] += 1
        return counts
    
    def evaluate(x):
        """Evaluate objective: max multiplicity, with penalty for uncovered sums."""
        counts = compute_counts(x)
        if np.any(counts == 0):
            uncovered = np.sum(counts == 0)
            return 10000 + uncovered
        return int(np.max(counts))
    
    def lagrangian_subgradient(x, multipliers, T):
        """
        Sub-gradient update for Lagrangian relaxation.
        x: binary vector (current solution)
        multipliers: dual variables for coverage constraints (one per target sum)
        T: temperature for stochastic acceptance
        
        Returns updated x.
        """
        counts = compute_counts(x)
        current_max = np.max(counts)
        uncovered = counts == 0
        
        # Sub-gradient for each element i:
        # dL/dx_i = sum over targets s where i participates in pairs:
        #   if x_i=1: contribution to count_s, adjust based on multiplier
        #   if x_i=0: potential contribution if added
        
        # For element i, find all targets s where i appears in some pair (i,j) or (j,i)
        # The marginal effect of flipping x_i depends on current state
        
        # Compute gradient for each element
        grad = np.zeros(n + 1, dtype=np.float64)
        
        for i in range(n + 1):
            # Find all pairs involving i
            for t in range(num_targets):
                s = target_lo + t
                # Check if i participates in any pair for target s
                # Pairs are (a,b) with a<=b, a+b=s
                # i participates if:
                #   (i, s-i) with i <= s-i, i.e., i <= s/2, and s-i <= n
                #   or (s-i, i) with s-i <= i, i.e., s-i <= i, and s-i >= 0
                
                j = s - i
                if j < 0 or j > n:
                    continue
                if j < i:
                    continue  # Only consider i <= j to avoid double counting
                
                # Pair (i, j) contributes to target t
                if x[i] == 1 and x[j] == 1:
                    # Currently contributing: removing i would decrease count_t by 1
                    # If count_t is at max, this helps; if count_t is 0 (uncovered), hurts
                    if counts[t] == current_max and current_max > 1:
                        grad[i] -= 1.0  # Removing helps reduce peak
                    elif counts[t] == 1:
                        grad[i] += 10.0  # Removing causes uncovered, bad
                    else:
                        grad[i] -= 0.1  # Slight benefit to removing
                else:
                    # Not currently contributing
                    if x[i] == 1 and x[j] == 0:
                        # i is in A but j is not: adding j would create this pair
                        pass
                    elif x[i] == 0:
                        # i is not in A: adding i might help
                        if counts[t] == 0:
                            grad[i] -= 5.0  # Adding helps cover
                        elif counts[t] == current_max:
                            grad[i] += 1.0  # Adding would increase peak
                        else:
                            grad[i] += 0.1  # Slight penalty for adding
        
        # Stochastic selection based on gradient
        # Elements with negative gradient (beneficial to flip) are more likely to be added/removed
        
        new_x = x.copy()
        for i in range(n + 1):
            if x[i] == 1:
                # Consider removing i
                # Probability based on how beneficial removal is
                if grad[i] < 0:
                    prob = min(0.8, abs(grad[i]) * 0.1)
                else:
                    prob = 0.05
                if random.random() < prob:
                    new_x[i] = 0
            else:
                # Consider adding i
                if grad[i] < 0:
                    prob = min(0.8, abs(grad[i]) * 0.1)
                else:
                    prob = 0.02
                if random.random() < prob:
                    new_x[i] = 1
        
        return new_x
    
    def repair(x):
        """Ensure all target sums are covered. If not, add elements greedily."""
        counts = compute_counts(x)
        while np.any(counts == 0):
            # Find uncovered targets
            uncovered_targets = np.where(counts == 0)[0]
            
            # For each uncovered target, find an element that can cover it
            best_elem = -1
            best_gain = 0
            
            for t in uncovered_targets:
                s = target_lo + t
                # Find pairs for this target
                for (i, j) in pair_lists[t]:
                    # Check if adding i or j would help
                    if x[i] == 0:
                        # Count how many other uncovered targets i would help
                        gain = 0
                        for t2 in uncovered_targets:
                            s2 = target_lo + t2
                            for (a, b) in pair_lists[t2]:
                                if (a == i or b == i) and (x[a] == 1 or x[b] == 1):
                                    gain += 1
                                    break
                        if gain > best_gain:
                            best_gain = gain
                            best_elem = i
                    if x[j] == 0 and j != i:
                        gain = 0
                        for t2 in uncovered_targets:
                            s2 = target_lo + t2
                            for (a, b) in pair_lists[t2]:
                                if (a == j or b == j) and (x[a] == 1 or x[b] == 1):
                                    gain += 1
                                    break
                        if gain > best_gain:
                            best_gain = gain
                            best_elem = j
            
            if best_elem >= 0:
                x[best_elem] = 1
                counts = compute_counts(x)
            else:
                # Can't cover, break
                break
        
        return x
    
    # Initial solution: start with the seed [64, 128]
    x_init = np.zeros(n + 1, dtype=np.int8)
    x_init[64:129] = 1
    x_init = repair(x_init)
    
    best_x = x_init.copy()
    best_score = evaluate(best_x)
    
    # Lagrangian relaxation with sub-gradient updates
    T = 5.0
    T_min = 0.1
    alpha = 0.9995
    max_iter = 50000
    
    current_x = x_init.copy()
    current_score = best_score
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sub-gradient update
        new_x = lagrangian_subgradient(current_x, None, T)
        
        # Ensure coverage
        new_x = repair(new_x)
        
        new_score = evaluate(new_x)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_x = new_x
            current_score = new_score
            if current_score < best_score:
                best_x = current_x.copy()
                best_score = current_score
        
        T *= alpha
        if T < T_min:
            break
    
    # Try a few more initializations with perturbation
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # Perturb the best solution
        x_perturbed = best_x.copy()
        num_flips = random.randint(5, 15)
        for _ in range(num_flips):
            idx = random.randint(0, n)
            x_perturbed[idx] = 1 - x_perturbed[idx]
        
        x_perturbed = repair(x_perturbed)
        
        # Run short Lagrangian search from this start
        T = 3.0
        current_x = x_perturbed.copy()
        current_score = evaluate(current_x)
        
        for it in range(20000):
            if time.time() > deadline:
                break
            
            new_x = lagrangian_subgradient(current_x, None, T)
            new_x = repair(new_x)
            new_score = evaluate(new_x)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_x = new_x
                current_score = new_score
                if current_score < best_score:
                    best_x = current_x.copy()
                    best_score = current_score
            
            T *= 0.9995
            if T < T_min:
                break
    
    # Convert to list
    result = [i for i in range(n + 1) if best_x[i] == 1]
    
    # Ensure non-empty
    if not result:
        result = [64]
    
    return result

# EVOLVE-BLOCK-END
