"""Incentive-compatible finite auction design for one additive bidder, two goods."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of values {0,1,2,3} x {0,1,2,3}
        constraints: constraint specifications
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin
    
    # Values for each good: 0, 1, 2, 3
    vals = [0, 1, 2, 3]
    n_vals = 4
    
    # Generate all 16 value profiles
    value_profiles = list(product(vals, vals))
    # Index mapping: (v1, v2) -> index
    idx_map = {}
    for i, (v1, v2) in enumerate(value_profiles):
        idx_map[(v1, v2)] = i
    
    # Parse distribution
    # distribution_spec could be a dict, list, or other format
    # We need to extract probabilities for each of the 16 profiles
    probs = _parse_distribution(distribution_spec, value_profiles)
    
    # For a single additive bidder, truthful mechanisms require:
    # 1. Allocation x(v1,v2) = (x1(v1,v2), x2(v1,v2)) is monotone in each vi
    # 2. Payment p(v1,v2) = v1*x1 + v2*x2 - U(v1,v2) where U >= 0 (IR)
    # 3. U satisfies envelope conditions
    
    # Strategy: Search over monotone allocation rules
    # For each good, allocation depends on value of that good (and possibly the other)
    # Simplest: x1 depends only on v1, x2 depends only on v2 (separable)
    # More general: x1(v1,v2) monotone in v1, x2(v1,v2) monotone in v2
    
    # Let's start with the most general case but use a structured search
    
    # For separable case: x1 = f1(v1), x2 = f2(v2) where f1, f2 are non-decreasing
    # f1: [0,1,2,3] -> [0,1], f2: [0,1,2,3] -> [0,1]
    
    # The utility for a separable mechanism:
    # U(v1,v2) = sum_{v1'<=v1} x1(v1') * (v1 - v1' + 1... ) -- need to be careful
    
    # Actually, for discrete values, the envelope condition gives:
    # U(v1,v2) >= U(0,v2) + sum_{k=1}^{v1} x1(k,v2) * 1  (if values are integers)
    # Wait, let me reconsider.
    
    # For values in {0,1,2,3}, the envelope condition states:
    # U(v1,v2) - U(v1-1,v2) >= x1(v1,v2) * 1  for v1 >= 1
    # U(v1,v2) - U(v1,v2-1) >= x2(v1,v2) * 1  for v2 >= 1
    
    # So: U(v1,v2) >= U(0,0) + sum_{k=1}^{v1} x1(k,v2) + sum_{k=1}^{v2} x2(v1,k)
    
    # For IR: U(v1,v2) >= 0 for all (v1,v2)
    # To minimize expected utility (maximize revenue), set U as small as possible:
    # U(v1,v2) = max(0, U(0,0) + sum_{k=1}^{v1} x1(k,v2) + sum_{k=1}^{v2} x2(v1,k))
    # But U(0,0) >= 0, so: U(v1,v2) = max(0, sum_{k=1}^{v1} x1(k,v2) + sum_{k=1}^{v2} x2(v1,k))
    # Setting U(0,0) = 0 is optimal.
    
    # So: U(v1,v2) = max(0, sum_{k=1}^{v1} x1(k,v2) + sum_{k=1}^{v2} x2(v1,k))
    
    # For separable: x1(k,v2) = f1(k), x2(v1,k) = f2(k)
    # U(v1,v2) = max(0, sum_{k=1}^{v1} f1(k) + sum_{k=1}^{v2} f2(k))
    
    # Revenue = E[v1*f1(v1) + v2*f2(v2) - U(v1,v2)]
    
    # Let's search over f1, f2 where f1, f2 are non-decreasing maps from {0,1,2,3} to [0,1]
    
    # Grid search: each f can take values in a fine grid
    # But 4 values each, non-decreasing, so we can parameterize by the 4 values
    
    # Let's use a coarse-to-fine approach
    grid_coarse = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid_fine = [i/20.0 for i in range(21)]
    
    best_revenue = -1
    best_f1 = [0, 0, 0, 0]
    best_f2 = [0, 0, 0, 0]
    
    def compute_revenue(f1, f2, probs):
        """Compute expected revenue for separable mechanism."""
        total_rev = 0.0
        total_util = 0.0
        for i, (v1, v2) in enumerate(value_profiles):
            p = probs[i]
            if p < 1e-15:
                continue
            a1 = f1[v1]
            a2 = f2[v2]
            value = v1 * a1 + v2 * a2
            # Utility
            s1 = sum(f1[k] for k in range(1, v1+1)) if v1 > 0 else 0.0
            s2 = sum(f2[k] for k in range(1, v2+1)) if v2 > 0 else 0.0
            util = max(0.0, s1 + s2)
            payment = value - util
            if payment < -1e-10:
                return -1  # Infeasible (shouldn't happen with our construction)
            total_rev += p * payment
            total_util += p * util
        return total_rev
    
    # Coarse search
    for f1_vals in product(grid_coarse, repeat=4):
        if not (f1_vals[0] <= f1_vals[1] <= f1_vals[2] <= f1_vals[3]):
            continue
        if time.time() - start_time > DEADLINE:
            break
        for f2_vals in product(grid_coarse, repeat=4):
            if not (f2_vals[0] <= f2_vals[1] <= f2_vals[2] <= f2_vals[3]):
                continue
            rev = compute_revenue(list(f1_vals), list(f2_vals), probs)
            if rev > best_revenue:
                best_revenue = rev
                best_f1 = list(f1_vals)
                best_f2 = list(f2_vals)
    
    # Fine search around best
    if time.time() - start_time < DEADLINE - 5:
        fine_grid = [max(0.0, min(1.0, best_f1[i] + d)) for d in [-0.1, -0.05, 0, 0.05, 0.1] for i in range(4)]
        # Better: local search
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            for i in range(4):
                for delta in [-0.02, -0.01, 0.01, 0.02]:
                    new_f1 = best_f1[:]
                    new_f1[i] = max(0.0, min(1.0, new_f1[i] + delta))
                    # Check monotonicity
                    mono = all(new_f1[j] <= new_f1[j+1] + 1e-9 for j in range(3))
                    if not mono:
                        continue
                    rev = compute_revenue(new_f1, best_f2, probs)
                    if rev > best_revenue + 1e-12:
                        best_revenue = rev
                        best_f1 = new_f1
                        improved = True
            for i in range(4):
                for delta in [-0.02, -0.01, 0.01, 0.02]:
                    new_f2 = best_f2[:]
                    new_f2[i] = max(0.0, min(1.0, new_f2[i] + delta))
                    mono = all(new_f2[j] <= new_f2[j+1] + 1e-9 for j in range(3))
                    if not mono:
                        continue
                    rev = compute_revenue(best_f1, new_f2, probs)
                    if rev > best_revenue + 1e-12:
                        best_revenue = rev
                        best_f2 = new_f2
                        improved = True
            if not improved:
                break
    
    # Now also try non-separable mechanisms for potentially better results
    # General case: x1(v1,v2) monotone in v1, x2(v1,v2) monotone in v2
    # This is a larger search space. Let's try a structured approach.
    
    # For now, let's also consider the case where allocations can depend on both values
    # but maintain monotonicity constraints
    
    if time.time() - start_time < DEADLINE - 10:
        # Try full allocation matrix search with monotonicity
        # x1 is a 4x4 matrix (v1 x v2), monotone in v1 (rows)
        # x2 is a 4x4 matrix, monotone in v2 (columns)
        
        # Use a simpler parameterization: threshold-based
        # x1(v1,v2) = 1 if v1 >= t1(v2) else 0
        # x2(v1,v2) = 1 if v2 >= t2(v1) else 0
        
        best_threshold_rev = best_revenue
        best_t1 = None  # t1[v2] for v2 in 0,1,2,3
        best_t2 = None  # t2[v1] for v1 in 0,1,2,3
        
        # t1: for each v2, threshold in v1 (0,1,2,3,4) where 4 means never allocate
        # t2: for each v1, threshold in v2 (0,1,2,3,4)
        thresholds = [0, 1, 2, 3, 4]
        
        for t1_vals in product(thresholds, repeat=4):
            if time.time() - start_time > DEADLINE - 10:
                break
            for t2_vals in product(thresholds, repeat=4):
                # Check monotonicity: t1 should be non-increasing in v2 (higher v2 -> lower threshold for good 1? No...)
                # Actually for truthfulness, we need x1 monotone in v1.
                # x1(v1,v2) = 1[v1 >= t1(v2)] is monotone in v1 for any t1(v2).
                # Similarly x2(v1,v2) = 1[v2 >= t2(v1)] is monotone in v2.
                # But we also need the combined mechanism to be truthful.
                # For single bidder, monotonicity in own value is sufficient.
                
                # Compute revenue
                total_rev = 0.0
                for i, (v1, v2) in enumerate(value_profiles):
                    p = probs[i]
                    if p < 1e-15:
                        continue
                    a1 = 1.0 if v1 >= t1_vals[v2] else 0.0
                    a2 = 1.0 if v2 >= t2_vals[v1] else 0.0
                    value = v1 * a1 + v2 * a2
                    # Utility: U(v1,v2) = max(0, sum_{k=1}^{v1} x1(k,v2) + sum_{k=1}^{v2} x2(v1,k))
                    s1 = 0.0
                    for k in range(1, v1+1):
                        s1 += 1.0 if k >= t1_vals[v2] else 0.0
                    s2 = 0.0
                    for k in range(1, v2+1):
                        s2 += 1.0 if k >= t2_vals[v1] else 0.0
                    util = max(0.0, s1 + s2)
                    payment = value - util
                    if payment < -1e-10:
                        break
                    total_rev += p * payment
                else:
                    if total_rev > best_threshold_rev:
                        best_threshold_rev = total_rev
                        best_t1 = t1_vals
                        best_t2 = t2_vals
        
        if best_threshold_rev > best_revenue:
            # Use threshold mechanism
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i, (v1, v2) in enumerate(value_profiles):
                a1 = 1.0 if v1 >= best_t1[v2] else 0.0
                a2 = 1.0 if v2 >= best_t2[v1] else 0.0
                allocations[i] = [a1, a2]
                value = v1 * a1 + v2 * a2
                s1 = 0.0
                for k in range(1, v1+1):
                    s1 += 1.0 if k >= best_t1[v2] else 0.0
                s2 = 0.0
                for k in range(1, v2+1):
                    s2 += 1.0 if k >= best_t2[v1] else 0.0
                util = max(0.0, s1 + s2)
                payments[i] = max(0.0, value - util)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
    
    # Use separable mechanism
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for i, (v1, v2) in enumerate(value_profiles):
        a1 = best_f1[v1]
        a2 = best_f2[v2]
        allocations[i] = [a1, a2]
        value = v1 * a1 + v2 * a2
        s1 = sum(best_f1[k] for k in range(1, v1+1)) if v1 > 0 else 0.0
        s2 = sum(best_f2[k] for k in range(1, v2+1)) if v2 > 0 else 0.0
        util = max(0.0, s1 + s2)
        payments[i] = max(0.0, value - util)
    
    return {
        'allocations': allocations,
        'payments': payments
    }

def _parse_distribution(distribution_spec, value_profiles):
    """Parse the distribution specification into probabilities for each value profile."""
    n = len(value_profiles)
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if arr.shape[0] == n:
            return arr / arr.sum()
        # Maybe it's a 2D array
        arr2d = np.array(distribution_spec, dtype=float)
        if arr2d.shape == (4, 4):
            return arr2d.flatten() / arr2d.sum()
    
    if isinstance(distribution_spec, dict):
        # Could be {value_profile: prob} or other format
        probs = np.zeros(n)
        for key, val in distribution_spec.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                v1, v2 = key
                if v1 in [0,1,2,3] and v2 in [0,1,2,3]:
                    idx = value_profiles.index((v1, v2))
                    probs[idx] = val
            elif isinstance(key, int) and 0 <= key < n:
                probs[key] = val
        total = probs.sum()
        if total > 1e-15:
            return probs / total
    
    # Default: uniform
    return np.ones(n) / n

# EVOLVE-BLOCK-END
