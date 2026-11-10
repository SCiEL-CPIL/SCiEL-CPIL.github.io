import time
from math import gcd

# Precompute factors
def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

_PRIME_FACTORS = {}
for _i in range(1, 512):
    _PRIME_FACTORS[_i] = _factorize(_i)

# Weights
_WEIGHTS = {a: 1.0 / (512 - a) for a in range(1, 512)}

# Sort indices by weight descending
_ORDERED_INDICES = sorted(range(1, 512), key=lambda x: _WEIGHTS[x], reverse=True)

# Precompute suffix sums of weights for upper bound
# bound[i] = sum of weights of all nodes from _ORDERED_INDICES[i] to end
_SUFFIX_SUMS = [0] * 512
for i in range(511, -1, -1):
    _SUFFIX_SUMS[i] = _SUFFIX_SUMS[i+1] + _WEIGHTS[_ORDERED_INDICES[i]]

def _is_coprime_to_all(a, chosen_factors):
    fa = _PRIME_FACTORS[a]
    if not fa:
        return True
    for f in chosen_factors:
        if fa & f:
            return False
    return True

def _branch_and_bound(deadline):
    best_score = 0
    best_set = []
    
    # Stack for DFS: (index, chosen_list, chosen_factors_set, current_score)
    # We use a stack to avoid recursion limit issues, though recursion depth is 511.
    # But B&B might be deep.
    
    stack = [(0, [], set(), 0.0)]
    
    while stack:
        idx, chosen, chosen_factors, score = stack.pop()
        
        if time.time() > deadline:
            break
            
        # Update best
        if score > best_score:
            best_score = score
            best_set = list(chosen)
            
        if idx >= 511:
            continue
            
        # Upper bound: current score + sum of all remaining weights
        # This is a loose bound, but effective due to decay.
        # We can tighten it: only sum weights of nodes compatible with chosen_factors.
        # But that's expensive. Let's start with loose bound.
        upper_bound = score + _SUFFIX_SUMS[idx]
        
        if upper_bound <= best_score:
            continue
            
        a = _ORDERED_INDICES[idx]
        
        # Option 1: Pick a
        if _is_coprime_to_all(a, chosen_factors):
            new_factors = set(chosen_factors)
            new_factors.add(_PRIME_FACTORS[a])
            stack.append((idx + 1, chosen + [a], new_factors, score + _WEIGHTS[a]))
            
        # Option 2: Skip a
        stack.append((idx + 1, chosen, chosen_factors, score))
        
    return best_set

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Run B&B
    result = _branch_and_bound(deadline)
    
    return result
