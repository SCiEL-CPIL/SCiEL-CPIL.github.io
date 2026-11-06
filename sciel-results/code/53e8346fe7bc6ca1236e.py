"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from math import gcd
    from functools import reduce

    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s limit
    
    # Denominators are 2, 3, ..., 48
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Calculate LCM of all denominators
    # LCM(a, b) = abs(a*b) // gcd(a, b)
    lcm_val = reduce(lambda a, b: a * b // gcd(a, b), denoms)
    
    # We want to select a subset S such that sum(1/d_i for i in S) < 1
    # This is equivalent to sum(lcm_val/d_i for i in S) < lcm_val
    # Let d_i_scaled = lcm_val // denoms[i]
    # We want to minimize deficit = 1 - sum(1/d_i) = (lcm_val - sum(d_i_scaled)) / lcm_val
    # Minimizing the deficit is equivalent to minimizing (lcm_val - sum(d_i_scaled))
    # Which is equivalent to maximizing sum(d_i_scaled)
    # Since the objective in milp is minimization, we minimize -sum(d_i_scaled)
    # Or simply minimize (lcm_val - sum(d_i_scaled)) directly if we can express it.
    # Actually, it's easier to maximize the sum. Let's minimize -sum.
    
    scaled_denoms = np.array([lcm_val // d for d in denoms], dtype=np.int64)
    
    # Objective: minimize -sum(scaled_denoms * x)
    # This is equivalent to maximizing sum(scaled_denoms * x)
    c = -scaled_denoms.astype(np.float64)
    
    # Constraints:
    # 1. sum(scaled_denoms * x) < lcm_val
    # Since x are integers, sum(scaled_denoms * x) is an integer.
    # sum < lcm_val  <=> sum <= lcm_val - 1
    # So: sum(scaled_denoms * x) <= lcm_val - 1
    # This is a linear constraint: A @ x <= b
    # A is a row vector of scaled_denoms, b is lcm_val - 1
    
    A = scaled_denoms.astype(np.float64)
    b_max = lcm_val - 1
    
    # LinearConstraint for: A @ x <= b_max
    # milp expects constraints in the form lb <= A @ x <= ub
    # So we set lb = -inf, ub = b_max
    constraint = LinearConstraint(A, lb=-np.inf, ub=b_max)
    
    # Bounds: x_i in {0, 1}
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success and res.x is not None:
            # res.x contains the optimal binary solution
            x_opt = res.x
            # Convert to indices
            indices = [i for i in range(n) if x_opt[i] > 0.5]
            
            # Validate
            if len(indices) > 0:
                # Check sum strictly below 1
                # Using exact rational check to be safe, though ILP should guarantee it
                # But floating point in ILP might be slightly off? 
                # The constraint was integer based (lcm_val - 1), so it should be strict.
                # Let's verify with Fraction just in case.
                from fractions import Fraction
                total = sum(Fraction(1, denoms[i]) for i in indices)
                if total < 1:
                    return indices
                else:
                    # Fallback if validation fails (shouldn't happen)
                    # Remove the smallest contribution to make it strictly below?
                    # Or just return empty? No, must be non-empty.
                    # If total >= 1, we can remove the index with the smallest 1/d (largest d)
                    # to reduce the sum.
                    # Sort indices by denominator descending (smallest fraction first)
                    indices_sorted = sorted(indices, key=lambda i: denoms[i], reverse=True)
                    if indices_sorted:
                        # Remove the one that reduces sum the least? No, we need to drop below 1.
                        # Dropping any one will reduce the sum.
                        # To maximize the remaining sum (minimize deficit), we should drop the smallest fraction.
                        # The smallest fraction corresponds to the largest denominator.
                        drop_idx = indices_sorted[0]
                        new_indices = [i for i in indices if i != drop_idx]
                        if new_indices:
                             total_check = sum(Fraction(1, denoms[i]) for i in new_indices)
                             if total_check < 1:
                                 return new_indices
                        return [indices[0]] # Last resort single item
                    else:
                        return [0]
            else:
                return [0]
        else:
            # Fallback to greedy if ILP fails
            from fractions import Fraction
            selected, total = [], Fraction()
            for i, d in enumerate(denoms):
                if total + Fraction(1, d) < 1:
                    selected.append(i)
                    total += Fraction(1, d)
            if selected:
                return selected
            return [0]
            
    except Exception as e:
        # Fallback to greedy
        from fractions import Fraction
        selected, total = [], Fraction()
        for i, d in enumerate(denoms):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        if selected:
            return selected
        return [0]
# EVOLVE-BLOCK-END
