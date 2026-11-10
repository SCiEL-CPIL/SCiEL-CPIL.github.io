"""Improved program: greedy by reciprocal weight + local search with time guard."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _has_3ap_with_candidate(selected_set, x):
    """Check if adding x to selected_set creates a 3-term AP."""
    # x = a + 2d (x is largest): need a, a+d, x with a+d in set
    # x = a + d (x is middle): need a, x, x+d with a in set and x+d in set (but x+d may not exist)
    # x = a + 2d (x is smallest): need x, x+d, x+2d with x+d and x+2d in set
    # Case 1: x is largest term: x = a + 2d => a = x - 2d, middle = x - d
    #   Need a >= 1, a in selected, middle in selected
    for d in range(1, (x + 1) // 2):
        a = x - 2 * d
        mid = x - d
        if a < 1:
            break
        if a in selected_set and mid in selected_set:
            return True
    # Case 2: x is middle term: x = a + d => a = x - d, largest = x + d
    #   Need a >= 1, a in selected, largest <= 256, largest in selected
    for d in range(1, 257 - x):
        a = x - d
        largest = x + d
        if a < 1:
            break
        if a in selected_set and largest in selected_set:
            return True
    # Case 3: x is smallest term: x = a, middle = x + d, largest = x + 2d
    #   Need middle <= 256, middle in selected, largest <= 256, largest in selected
    for d in range(1, (257 - x) // 2):
        mid = x + d
        largest = x + 2 * d
        if largest > 256:
            break
        if mid in selected_set and largest in selected_set:
            return True
    return False

def _reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _is_valid_set(selected):
    """Verify no 3-APs in the set."""
    s = set(selected)
    for a in s:
        for b in s:
            if a >= b:
                continue
            c = 2 * b - a
            if c in s and c != a and c != b:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 90s kill
    
    # Greedy: try numbers in order of largest reciprocal (smallest first)
    # But also consider that smaller numbers have higher reciprocal weight
    candidates = list(range(1, 257))
    # Sort by reciprocal weight descending (i.e., by value ascending)
    candidates.sort(key=lambda x: -1.0/x)
    
    selected = []
    selected_set = set()
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if not _has_3ap_with_candidate(selected_set, x):
            selected.append(x)
            selected_set.add(x)
    
    if not selected:
        return [1]
    
    # Local search: try to improve by removing one and adding another
    current_score = _reciprocal_sum(selected)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try removing each element and adding the best available
        for i, x in enumerate(list(selected)):
            if time.time() > deadline - 2:
                break
            # Remove x
            selected.remove(x)
            selected_set.discard(x)
            
            # Try adding each candidate not in set
            best_add = None
            best_score = _reciprocal_sum(selected)
            for y in range(1, 257):
                if y in selected_set:
                    continue
                if _has_3ap_with_candidate(selected_set, y):
                    continue
                new_score = best_score + Fraction(1, y)
                if new_score > best_score:
                    best_score = new_score
                    best_add = y
            
            if best_add is not None:
                selected.append(best_add)
                selected_set.add(best_add)
                improved = True
                if _reciprocal_sum(selected) > current_score:
                    current_score = _reciprocal_sum(selected)
            else:
                # Put x back
                selected.append(x)
                selected_set.add(x)
    
    # Final validity check
    if not _is_valid_set(selected):
        # Fallback: simple greedy
        selected = []
        selected_set = set()
        for x in range(1, 257):
            if not _has_3ap_with_candidate(selected_set, x):
                selected.append(x)
                selected_set.add(x)
    
    return sorted(selected)
# EVOLVE-BLOCK-END
