"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 128
    target_low = 128
    target_high = 256
    
    # Precompute pairs for each sum s in [128, 256]
    # pairs[s] is a list of tuples (a, b) with a <= b, a+b=s, a,b in [0, 128]
    pairs_for_sum = {}
    for s in range(target_low, target_high + 1):
        pairs = []
        a_min = max(0, s - 128)
        a_max = min(128, s // 2)
        for a in range(a_min, a_max + 1):
            b = s - a
            pairs.append((a, b))
        pairs_for_sum[s] = pairs

    def objective(x):
        # x is a numpy array of length 129
        # Threshold to binary
        x_bin = (x > 0.5).astype(int)
        
        max_count = 0
        uncovered = 0
        
        for s in range(target_low, target_high + 1):
            count = 0
            for a, b in pairs_for_sum[s]:
                if x_bin[a] == 1 and x_bin[b] == 1:
                    count += 1
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        
        # Strong penalty for uncovered sums to ensure feasibility
        penalty = 1000 * uncovered
        return max_count + penalty

    bounds = [(0, 1)] * 129
    
    # Differential Evolution settings
    # We want to minimize the max count.
    # Using 'latinhypercube' for initialization helps cover the space better.
    # popsize=10-20 is good for 129 dims to keep evals manageable.
    # maxiter=100-200.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=150, 
            popsize=15, 
            tol=0.01,
            init='latinhypercube',
            seed=42,
            polish=False # Polish might be slow or unnecessary if we threshold
        )
    except Exception as e:
        # Fallback if DE fails or times out (though DE has no strict time limit, it has maxiter)
        # We'll just use a simple heuristic fallback
        return list(range(64, 129))

    best_x = result.x
    best_A_set = set(i for i in range(129) if best_x[i] > 0.5)
    
    # Verify and potentially repair if needed (though penalty should handle it)
    # Check if covered
    def is_covered(A_set):
        for s in range(target_low, target_high + 1):
            found = False
            for a, b in pairs_for_sum[s]:
                if a in A_set and b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True
        
    if not is_covered(best_A_set):
        # Fallback to a known good solution if DE failed to find a covered one
        # Simple heuristic: A = [64..128] is always covered
        return list(range(64, 129))
        
    return list(best_A_set)
# EVOLVE-BLOCK-END
