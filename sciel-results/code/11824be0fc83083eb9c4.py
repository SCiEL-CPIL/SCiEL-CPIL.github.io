"""Combined program: fast integer-weighted subset-sum collision detection with greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_dissociated_int(weights):
    """
    Check if all subset sums of integer weights are distinct.
    weights: list of positive integers (scaled reciprocals).
    Returns True if all 2^n subset sums are unique.
    """
    n = len(weights)
    if n == 0:
        return True
    # Generate all subset sums incrementally, checking for collisions
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _is_dissociated(elements):
    """
    Check if all reciprocal subset sums of elements are distinct.
    Uses integer arithmetic via LCM scaling.
    """
    if not elements:
        return True
    n = len(elements)
    if n > 20:
        # Too large for exact integer check within time budget; use float heuristic
        return _is_dissociated_float(elements)
    try:
        l = reduce(_lcm, elements)
    except Exception:
        return False
    weights = [l // e for e in elements]
    return _is_dissociated_int(weights)

def _is_dissociated_float(elements):
    """
    Floating-point check for dissociation. Used as a heuristic for larger sets.
    Returns True if no duplicate subset sums detected in float arithmetic.
    """
    if not elements:
        return True
    n = len(elements)
    if n > 25:
        # Too large even for float; use a quick necessary-condition check
        # If sum of reciprocals > some threshold, likely collisions, but not definitive
        # Just return True to let the caller decide (risky but fast)
        return True
    
    reciprocals = [1.0 / e for e in elements]
    sums = [0.0]
    for r in reciprocals:
        new_sums = [s + r for s in sums]
        sums.extend(new_sums)
    
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums) - 1):
        # Use a very tight epsilon; distinct unit fraction sums from [1..22]
        # differ by at least ~1/LCM(1..22) ~ 4e-10, well above float eps
        if abs(sorted_sums[i] - sorted_sums[i+1]) < 1e-12:
            return False
    return True

def _check_validity(candidate):
    """Verify candidate is a valid nonempty subset of [1,22] with distinct reciprocal subset sums."""
    if not candidate:
        return False
    if len(candidate) != len(set(candidate)):
        return False
    if any(x < 1 or x > 22 for x in candidate):
        return False
    # Use exact integer check for verification
    return _is_dissociated(sorted(candidate))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    universe = list(range(1, 23))
    
    # --- Phase 1: Greedy construction with multiple orders ---
    # Try different addition orders to escape local minima
    orders = [
        list(range(1, 23)),          # increasing
        list(range(22, 0, -1)),      # decreasing
    ]
    
    # Add random orders
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 15:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            new_set = current_set + [x]
            # Use fast check: for small sets use exact, for larger use float
            if len(new_set) <= 18:
                valid = _is_dissociated(new_set)
            else:
                valid = _is_dissociated_float(new_set)
            if valid:
                current_set = new_set
                used.add(x)
        
        if len(current_set) > best_size:
            # Verify with exact check before accepting
            if _check_validity(current_set):
                best_set = current_set
                best_size = len(current_set)
    
    # --- Phase 2: Local search / hill climbing ---
    # Try to improve by swapping elements or adding/removing
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current = list(best_set)
        
        # Try adding each unused element
        for x in universe:
            if x in current:
                continue
            new_set = current + [x]
            if len(new_set) <= 20:
                valid = _is_dissociated(new_set)
            else:
                valid = _is_dissociated_float(new_set)
            if valid:
                if _check_validity(new_set):
                    current = new_set
                    improved = True
                    break
        
        # Try swapping: remove one, add another
        if not improved:
            for i, a in enumerate(list(current)):
                if time.time() > deadline - 5:
                    break
                for x in universe:
                    if x in current:
                        continue
                    new_set = [y for y in current if y != a] + [x]
                    if len(new_set) <= 20:
                        valid = _is_dissociated(new_set)
                    else:
                        valid = _is_dissociated_float(new_set)
                    if valid:
                        if _check_validity(new_set):
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            if len(current) > best_size:
                best_set = current
                best_size = len(current)
    
    # --- Phase 3: Random restarts with perturbation ---
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        
        # Start from a random subset of the best set, or a random seed
        if random.random() < 0.5:
            # Perturb best set: remove 1-2 elements, try to regrow
            current = list(best_set)
            if len(current) > 5:
                remove_count = random.randint(1, min(2, len(current) - 5))
                for _ in range(remove_count):
                    idx = random.randint(0, len(current) - 1)
                    current.pop(idx)
            else:
                current = []
        else:
            current = []
        
        # Greedy grow
        order = list(range(1, 23))
        random.shuffle(order)
        used = set(current)
        for x in order:
            if x in used:
                continue
            new_set = current + [x]
            if len(new_set) <= 18:
                valid = _is_dissociated(new_set)
            else:
                valid = _is_dissociated_float(new_set)
            if valid:
                current = new_set
                used.add(x)
        
        if len(current) > best_size:
            if _check_validity(current):
                best_set = current
                best_size = len(current)
    
    # Final verification
    if not _check_validity(best_set):
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
