"""Direct construction for 'Large sets with distinct subset products'."""

import time

DEADLINE = 82.0
T_START = time.time()

def _is_valid(numbers):
    """Check if all subset products are distinct via direct enumeration."""
    n = len(numbers)
    if n > 22:
        return None  # unknown
    # Use iterative product building for speed
    prods = [1]
    for num in numbers:
        new_prods = [p * num for p in prods]
        prods.extend(new_prods)
    # Check all distinct
    if len(prods) != len(set(prods)):
        return False
    return True

def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    all_candidates = list(range(2, 41))
    
    # Start with all 12 primes - known valid
    best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Greedy add: try to add each candidate if valid
    for _ in range(10):  # Multiple passes
        if time_left() < 10.0:
            break
        added = False
        for cand in all_candidates:
            if cand in best_set:
                continue
            new_set = best_set + [cand]
            if _is_valid(new_set):
                best_set = new_set
                added = True
        if not added:
            break
    
    # Local search: try swaps to enable further additions
    if time_left() > 20.0:
        for _ in range(5):
            if time_left() < 15.0:
                break
            improved = False
            for i in range(len(best_set)):
                if time_left() < 10.0:
                    break
                for cand in all_candidates:
                    if cand in best_set:
                        continue
                    new_set = best_set[:i] + best_set[i+1:] + [cand]
                    if not _is_valid(new_set):
                        continue
                    # Try to add more to swapped set
                    for cand2 in all_candidates:
                        if cand2 in new_set:
                            continue
                        new_set2 = new_set + [cand2]
                        if _is_valid(new_set2):
                            best_set = new_set2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Final verification
    valid = _is_valid(best_set)
    if not valid:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    res = [int(x) for x in best_set]
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in res)
    assert len(res) == len(set(res))
    
    return res
