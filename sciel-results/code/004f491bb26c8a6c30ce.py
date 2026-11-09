import time
from fractions import Fraction
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # Denominators are 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of all denominators for exact integer scaling
    # LCM of 2..48
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        # Early check if L is getting too large
        if L > 10**18:
            break
    
    # Use a smaller set if LCM is too large; fall back to meet-in-the-middle
    # For now, try MILP with the full LCM if feasible
    # LCM(2..48) is huge (~2.3e18), so we need to be careful
    
    # Strategy: Use MILP with integer weights = LCM // denominator
    # Maximize sum of weights subject to sum < L (i.e., sum <= L - 1)
    # And at least one item selected
    
    # Check if LCM is manageable
    L_full = 1
    for d in denominators:
        L_full = lcm(L_full, d)
    
    # If LCM is too large for MILP, use meet-in-the-middle fallback
    if L_full > 10**15:
        return _meet_in_the_middle(denominators, start_time, deadline)
    
    # MILP formulation:
    # Variables: x_i in {0, 1} for each denominator
    # Maximize: sum(weight_i * x_i) where weight_i = L // d_i
    # Subject to: sum(weight_i * x_i) <= L - 1  (strictly less than L)
    #             sum(x_i) >= 1  (non-empty)
    
    weights = np.array([L_full // d for d in denominators], dtype=np.int64)
    
    # scipy.optimize.milp
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    # Objective: minimize -sum(weights * x) to maximize sum
    c = -weights.astype(np.float64)
    
    # Constraints:
    # 1) sum(weights * x) <= L_full - 1
    # 2) sum(x) >= 1
    
    # Linear constraints: A @ x <= b
    # Constraint 1: weights @ x <= L_full - 1
    # Constraint 2: -sum(x) <= -1  (i.e., sum(x) >= 1)
    
    A = np.array([weights.astype(np.float64), -np.ones(n)], dtype=np.float64)
    b = np.array([L_full - 1, -1], dtype=np.float64)
    
    constraints = LinearConstraint(A, -np.inf, b)
    
    # Variable bounds: 0 <= x_i <= 1
    integrality = np.ones(n)  # All variables are integer (binary)
    bounds = Bounds(0, 1)
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0:  # Optimal
            x = result.x
            selected = [i for i in range(n) if x[i] > 0.5]
            
            if not selected:
                selected = [0]  # Fallback: pick index 0 (denominator 2)
            
            # Verify with exact fractions
            total = Fraction(0)
            for idx in selected:
                total += Fraction(1, denominators[idx])
            
            if total < 1:
                return selected
            else:
                # Should not happen, but fallback
                return [0]
        else:
            # MILP failed, use fallback
            return _meet_in_the_middle(denominators, start_time, deadline)
            
    except Exception as e:
        # Fallback to meet-in-the-middle
        return _meet_in_the_middle(denominators, start_time, deadline)


def _meet_in_the_middle(denominators, start_time, deadline):
    """Fallback: meet-in-the-middle approach"""
    import bisect
    
    n = len(denominators)
    mid = n // 2
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    left_size = len(left_denoms)
    right_size = len(right_denoms)
    
    left_recs = [1.0 / d for d in left_denoms]
    right_recs = [1.0 / d for d in right_denoms]
    
    # Generate left sums
    left_sums = []
    for mask in range(1 << left_size):
        s = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += left_recs[i]
            m >>= 1
            i += 1
        if s < 1.0:
            left_sums.append((s, mask))
    
    left_sums.sort(key=lambda x: x[0])
    left_sum_values = [x[0] for x in left_sums]
    
    best_deficit = 1.0
    best_left_mask = 0
    best_right_mask = 0
    
    for mask in range(1 << right_size):
        if time.time() > deadline:
            break
            
        s_right = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s_right += right_recs[i]
            m >>= 1
            i += 1
            
        if s_right >= 1.0:
            continue
            
        target = 1.0 - s_right
        pos = bisect.bisect_left(left_sum_values, target)
        if pos > 0:
            idx = pos - 1
            s_left = left_sum_values[idx]
            l_mask = left_sums[idx][1]
            
            deficit = 1.0 - (s_left + s_right)
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = l_mask
                best_right_mask = mask
    
    # Verify with fractions
    left_frac_recs = [Fraction(1, d) for d in left_denoms]
    right_frac_recs = [Fraction(1, d) for d in right_denoms]
    
    candidates = [(best_left_mask, best_right_mask), (0, 0)]
    
    best_exact_deficit = None
    best_exact_l_mask = 0
    best_exact_r_mask = 0
    
    for lm, rm in candidates:
        s_left = Fraction(0)
        m = lm
        i = 0
        while m:
            if m & 1:
                s_left += left_frac_recs[i]
            m >>= 1
            i += 1
            
        s_right = Fraction(0)
        m = rm
        i = 0
        while m:
            if m & 1:
                s_right += right_frac_recs[i]
            m >>= 1
            i += 1
            
        total = s_left + s_right
        if total < 1:
            deficit = 1 - total
            if best_exact_deficit is None or deficit < best_exact_deficit:
                best_exact_deficit = deficit
                best_exact_l_mask = lm
                best_exact_r_mask = rm
    
    indices = []
    m = best_exact_l_mask
    i = 0
    while m:
        if m & 1:
            indices.append(i)
        m >>= 1
        i += 1
        
    m = best_exact_r_mask
    i = 0
    while m:
        if m & 1:
            indices.append(mid + i)
        m >>= 1
        i += 1
        
    if not indices:
        indices = [0]
        
    return indices
