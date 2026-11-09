import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute suffix sums as Fractions and floats
    suffix_sums = [Fraction(0, 1)] * (n + 1)
    suffix_sums_float = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        val = Fraction(1, denominators[i])
        suffix_sums[i] = suffix_sums[i + 1] + val
        suffix_sums_float[i] = suffix_sums_float[i + 1] + float(val)
        
    # DP: {sum_fraction: (sum_float, indices_list)}
    states = {Fraction(0, 1): (0.0, [])}
    
    best_sum = Fraction(0, 1)
    best_indices = []
    best_sum_float = 0.0
    
    for i, d in enumerate(denominators):
        val = Fraction(1, d)
        val_float = float(val)
        next_suffix = suffix_sums[i + 1]
        next_suffix_float = suffix_sums_float[i + 1]
        
        new_states = {}
        
        for s, (s_float, idxs) in states.items():
            
            # Option 1: Skip d
            if s_float + next_suffix_float > best_sum_float:
                if s not in new_states:
                    new_states[s] = (s_float, idxs)
            
            # Option 2: Pick d
            ns = s + val
            ns_float = s_float + val_float
            
            if ns < 1:  # Strictly below 1
                if ns > best_sum:
                    best_sum = ns
                    best_indices = idxs + [i]
                    best_sum_float = ns_float
                
                if ns_float + next_suffix_float > best_sum_float:
                    if ns not in new_states:
                        new_states[ns] = (ns_float, idxs + [i])
        
        states = new_states
        
        if not states:
            break
            
        if time.time() > deadline:
            break
            
    return best_indices

