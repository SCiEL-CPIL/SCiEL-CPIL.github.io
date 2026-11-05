import time
from fractions import Fraction

def _can_form_target(target, values, memo=None):
    """Check if target can be formed as sum of a non-empty subset of values."""
    if not values:
        return False
    if memo is None:
        memo = {}
    
    values = tuple(sorted(values, reverse=True))
    key = (target, values)
    if key in memo:
        return memo[key]
    
    n = len(values)
    total = sum(values)
    if total < target:
        memo[key] = False
        return False
    if total == target:
        memo[key] = True
        return True
    
    # Prune: if current sum + remaining < target, impossible
    # Try including the largest element
    if _can_form_target(target - values[0], values[1:], memo):
        memo[key] = True
        return True
    # Try excluding
    if _can_form_target(target, values[1:], memo):
        memo[key] = True
        return True
    
    memo[key] = False
    return False

def _is_valid_set(S, memo=None):
    S = set(S)
    if memo is None:
        memo = {}
    for x in S:
        others = [y for y in S if y != x]
        target = Fraction(1, x)
        values = [Fraction(1, y) for y in others]
        if _can_form_target(target, values, memo):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85
    
    best = set(range(13, 25))
    best_size = len(best)
    memo = {}
    
    # Strategy 1: Add elements from 12 down to 1
    for c in range(12, 0, -1):
        if time.time() > deadline:
            break
        if c in best:
            continue
        trial = best | {c}
        if _is_valid_set(trial, memo):
            best = trial
            best_size = len(best)
    
    # Strategy 2: Try to add remaining elements in order of increasing value
    remaining = sorted([c for c in range(1, 25) if c not in best])
    for c in remaining:
        if time.time() > deadline:
            break
        trial = best | {c}
        if _is_valid_set(trial, memo):
            best = trial
            best_size = len(best)
    
    # Strategy 3: Local search - try to swap elements
    for _ in range(50):  # Limit iterations
        if time.time() > deadline:
            break
        improved = False
        current_elements = sorted(best)
        for x in current_elements:
            if time.time() > deadline:
                break
            for y in range(1, 25):
                if y in best or y == x:
                    continue
                trial = (best - {x}) | {y}
                if _is_valid_set(trial, memo):
                    # Only accept if it allows adding more later
                    # For now, accept any valid swap
                    best = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return sorted(best)
