import time
from fractions import Fraction

def _can_form_target(target, values):
    """Check if target can be formed as sum of a non-empty subset of values."""
    if not values:
        return False
    values = sorted(values, reverse=True)
    total = sum(values)
    if total < target:
        return False
    n = len(values)
    # Precompute suffix sums for pruning
    suffix = [Fraction(0, 1)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]
    
    def dfs(idx, current):
        if current == target:
            return True
        if idx == n:
            return False
        if current > target:
            return False
        # Prune
        if current + suffix[idx] < target:
            return False
        # Include
        if dfs(idx + 1, current + values[idx]):
            return True
        # Exclude
        if dfs(idx + 1, current):
            return True
        return False
    
    return dfs(0, Fraction(0, 1))

def _is_valid_set(S):
    S = set(S)
    for x in S:
        others = [y for y in S if y != x]
        target = Fraction(1, x)
        values = [Fraction(1, y) for y in others]
        if _can_form_target(target, values):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 80
    
    best = set(range(13, 25))
    
    for c in range(12, 0, -1):
        if time.time() > deadline:
            break
        trial = best | {c}
        if _is_valid_set(trial):
            best = trial
    
    return sorted(best)
