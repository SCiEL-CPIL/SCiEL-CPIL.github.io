import time
from fractions import Fraction
import math
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47 elements, indices 0 to 46

    # Compute LCM of all denominators for exact integer arithmetic
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Precompute scaled values: v[i] = L // denominators[i]
    # Then sum of reciprocals = sum(v[i]) / L
    # We want sum(v[i]) < L (strictly)
    # Minimize L - sum(v[i])
    
    v = [L // d for d in denominators]
    
    best_deficit = L  # max possible deficit
    best_combo = None
    
    # Search over subset sizes 1 through 7
    # C(47,1)=47, C(47,2)=1081, C(47,3)=16215, C(47,4)=178365, 
    # C(47,5)=1533939, C(47,6)=10737573, C(47,7)=62121645
    for size in range(1, 8):
        if time.time() > DEADLINE:
            break
        
        for combo in combinations(range(n), size):
            s = 0
            for idx in combo:
                s += v[idx]
            
            if s < L:  # strictly less than 1
                deficit = L - s
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_combo = combo
            
            # Early termination: if all remaining elements would make s >= L
            # (not straightforward, skip for now)
        
        if time.time() > DEADLINE:
            break
    
    if best_combo is None:
        # Fallback: just pick 1/2
        best_combo = (0,)
        best_deficit = L - v[0]
    
    # Convert to list of indices
    selected = list(best_combo)
    
    # Verify with Fraction
    total_frac = Fraction()
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
        
    if total_frac >= 1:
        # Should not happen, but fallback
        selected = [0]
    
    return selected

def greedy_fallback(denominators):
    # Simple greedy: add largest available term if sum < 1
    selected = []
    total = Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
