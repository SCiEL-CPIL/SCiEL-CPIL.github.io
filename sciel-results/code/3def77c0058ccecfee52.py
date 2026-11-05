# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256

def _precompute_conflicts():
    """Precompute conflict pairs. conflict[x] is a bitmask of elements that conflict with x."""
    conflicts = [0] * (N + 1)
    for x in range(1, N + 1):
        # For each x, find all a < x such that c = 2*x - a > x and c <= N
        # This means a, x, c form a 3-AP
        # a = 2*x - c, so for each c > x with c <= N and 2*x - c >= 1 and 2*x - c < x
        # i.e., c > x and 2*x - c >= 1 => c <= 2*x - 1
        # Also need 2*x - c < x => c > x (already have)
        for c in range(x + 1, min(2 * x, N + 1)):
            a = 2 * x - c
            if 1 <= a < x:
                # a, x, c is a 3-AP: x conflicts with a and c
                conflicts[x] |= (1 << a) | (1 << c)
                conflicts[a] |= (1 << x) | (1 << c)
                conflicts[c] |= (1 << x) | (1 << a)
    return conflicts

# Precompute once
_CONFLICTS = _precompute_conflicts()

def _can_add_mask(mask, x):
    """Check if x can be added given current selection bitmask."""
    return (_CONFLICTS[x] & mask) == 0

def _recip_sum_from_mask(mask):
    """Compute exact reciprocal sum from bitmask."""
    total = Fraction(0, 1)
    for x in range(1, N + 1):
        if mask & (1 << x):
            total += Fraction(1, x)
    return total

def _mask_to_set(mask):
    """Convert bitmask to set."""
    return {x for x in range(1, N + 1) if mask & (1 << x)}

def _greedy_solution():
    """Greedy: add elements in order of reciprocal value (smallest first)."""
    mask = 0
    for x in range(1, N + 1):
        if _can_add_mask(mask, x):
            mask |= (1 << x)
    return mask

def _local_search(initial_mask, time_limit):
    """Local search with bitmask-based fast validity checks."""
    deadline = time.time() + time_limit
    mask = initial_mask
    best_score = _recip_sum_from_mask(mask)
    best_mask = mask
    
    # Precompute all elements sorted by reciprocal value (descending) for priority
    all_elements = list(range(1, N + 1))
    
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Try adding any element not in mask, prioritize smaller x
        for x in all_elements:
            if not (mask & (1 << x)) and _can_add_mask(mask, x):
                new_mask = mask | (1 << x)
                # Quick score estimate: add 1/x
                new_score = best_score + Fraction(1, x)
                if new_score > best_score:
                    mask = new_mask
                    best_score = new_score
                    best_mask = mask
                    improved = True
        
        # Phase 2: Try removing one element and re-optimizing
        for x in all_elements:
            if mask & (1 << x):
                trial_mask = mask & ~(1 << x)
                # Greedily add back all possible elements
                for y in all_elements:
                    if not (trial_mask & (1 << y)) and _can_add_mask(trial_mask, y):
                        trial_mask |= (1 << y)
                new_score = _recip_sum_from_mask(trial_mask)
                if new_score > best_score:
                    mask = trial_mask
                    best_score = new_score
                    best_mask = mask
                    improved = True
        
        # Phase 3: Try swaps - remove one, add another
        for x in all_elements:
            if not (mask & (1 << x)):
                continue
            trial_mask = mask & ~(1 << x)
            for y in all_elements:
                if y == x or (mask & (1 << y)):
                    continue
                if _can_add_mask(trial_mask, y):
                    swap_mask = trial_mask | (1 << y)
                    new_score = best_score - Fraction(1, x) + Fraction(1, y)
                    if new_score > best_score:
                        mask = swap_mask
                        best_score = new_score
                        best_mask = mask
                        improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    
    return best_mask

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    # Get greedy solution
    mask = _greedy_solution()
    
    # Local search with time budget
    mask = _local_search(mask, time_limit)
    
    # Final check
    if mask == 0:
        return [1]
    
    return sorted(_mask_to_set(mask))
# EVOLVE-BLOCK-END
