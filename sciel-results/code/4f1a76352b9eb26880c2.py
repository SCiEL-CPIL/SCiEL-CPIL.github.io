def construct():
    from fractions import Fraction
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    denoms = list(range(2, 49))
    n = len(denoms)
    vals = [Fraction(1, d) for d in denoms]
    one = Fraction(1, 1)
    
    # Weights for ILP: 1/d_i
    # Using float for ILP
    weights_float = np.array([1.0 / d for d in denoms])
    
    # Objective: Maximize sum(weights_float * x)
    # milp minimizes c^T x, so c = -weights_float
    c = -weights_float
    
    # Constraint: sum(weights_float * x) <= 1.0
    # Note: Strictly below 1 is handled by post-processing exact check.
    # If we use 1.0, solver might return sum=1.0.
    # We can try 1.0 - 1e-15 to be safe?
    # But 1e-15 is much larger than 1/L (~1e-19).
    # So it might exclude valid solutions close to 1.
    # Better to use 1.0 and filter.
    
    A = weights_float.reshape(1, -1)
    lb = 0.0
    ub = 1.0
    constraint = LinearConstraint(A, lb, ub)
    
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    best_sum = Fraction(0)
    best_set = set()
    
    def evaluate(s):
        tot = Fraction(0)
        for i in s:
            tot += vals[i]
        return tot
    
    def try_update(s):
        nonlocal best_sum, best_set
        if not s:
            return
        tot = evaluate(s)
        if tot < one and tot > best_sum:
            best_sum = tot
            best_set = set(s)
            
    if res.status == 0:
        x = res.x
        # x is float, but should be close to 0 or 1
        s0 = set()
        for i in range(n):
            if x[i] > 0.5:
                s0.add(i)
        
        # Check exact sum
        cur_sum = evaluate(s0)
        
        # If sum >= 1, we need to adjust
        if cur_sum >= one:
            # Try removing elements one by one to get below 1
            # Remove the one with smallest value? No, we want to keep sum high.
            # If sum > 1, we must remove something.
            # To maximize remaining sum, remove the smallest 1/d_i?
            # Actually, if sum > 1, any removal makes it smaller.
            # We want the largest sum < 1.
            # So we should remove the element that causes the smallest drop?
            # No, we want to drop just enough to get below 1.
            # If sum = 1 + delta, we need to drop at least delta.
            # Removing an element with value v drops sum by v.
            # We want to minimize the drop, so remove the smallest v?
            # But we must ensure sum < 1.
            # If sum - v_min < 1, remove v_min.
            # If sum - v_min >= 1, remove v_min + v_second_min, etc.
            
            # Greedy reduction
            s_adj = set(s0)
            cur = cur_sum
            while cur >= one and s_adj:
                # Find element with smallest value to remove
                min_i = min(s_adj, key=lambda i: vals[i])
                s_adj.remove(min_i)
                cur -= vals[min_i]
            
            if s_adj:
                try_update(s_adj)
            else:
                # If empty, try single elements
                for i in range(n):
                    try_update({i})
        else:
            try_update(s0)
            
    # Local Search from best_set
    if best_set:
        s = set(best_set)
        cur = best_sum
        improved = True
        while improved:
            improved = False
            # Try add
            for i in range(n):
                if i not in s:
                    if cur + vals[i] < one:
                        ns = s | {i}
                        ns_sum = cur + vals[i]
                        if ns_sum > cur:
                            s = ns
                            cur = ns_sum
                            improved = True
            # Try swap
            for i in list(s):
                for j in range(n):
                    if j not in s:
                        ns_sum = cur - vals[i] + vals[j]
                        if ns_sum < one and ns_sum > cur:
                            s = (s - {i}) | {j}
                            cur = ns_sum
                            improved = True
        try_update(s)
        
    # If still empty, fallback to random
    if not best_set:
        for i in range(n):
            try_update({i})
            
    return list(best_set)
