"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _is_3ap_free(selected):
    """Check if selected set has no 3-term AP."""
    s = set(selected)
    for a in s:
        for b in s:
            if a < b:
                c = 2*b - a
                if c in s and c != a and c != b:
                    return False
    return True

def _has_3ap_with(selected, x):
    """Check if adding x to selected creates a 3-term AP."""
    s = set(selected)
    # x could be the middle term: a, x, c where c = 2x - a
    for a in s:
        c = 2*x - a
        if c in s and c != x:
            return True
    # x could be the last term: a, b, x where b = (a+x)/2
    for a in s:
        b = (a + x) / 2
        if b == int(b) and int(b) in s and int(b) != x:
            return True
    # x could be the first term: x, b, c where b = x + d, c = x + 2d
    for b in s:
        if b > x:
            d = b - x
            c = b + d
            if c in s and c != x:
                return True
    return False

def _reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_descending():
    """Greedy: process from 1 to 256, add if no 3-AP created."""
    selected = []
    for x in range(1, 257):
        if not _has_3ap_with(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, time_budget=45):
    """Try to improve by removing one element and adding another."""
    selected_set = set(selected)
    current_sum = _reciprocal_sum(selected)
    start_time = time.time()
    
    # Precompute which elements can be added
    candidates = list(range(1, 257))
    
    while time.time() - start_time < time_budget:
        improved = False
        # Try removing each element and adding a better one
        for i, rem in enumerate(selected):
            if time.time() - start_time >= time_budget:
                break
            remaining = [x for j, x in enumerate(selected) if j != i]
            rem_set = set(remaining)
            
            # Find best candidate to add
            best_add = None
            best_gain = Fraction(0, 1)
            
            for x in candidates:
                if x in rem_set:
                    continue
                # Check if adding x to remaining creates 3-AP
                if _has_3ap_with(remaining, x):
                    continue
                gain = Fraction(1, x) - Fraction(1, rem)
                if gain > best_gain:
                    best_gain = gain
                    best_add = x
            
            if best_add is not None:
                new_selected = remaining + [best_add]
                new_sum = current_sum + best_gain
                if new_sum > current_sum:
                    selected = new_selected
                    selected_set = set(selected)
                    current_sum = new_sum
                    improved = True
                    break
        
        if not improved:
            break
    
    return selected, current_sum

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy initialization
    selected = _greedy_descending()
    current_sum = _reciprocal_sum(selected)
    
    # Phase 2: Local search
    if time.time() - start_time < 50:
        selected, current_sum = _local_search(selected, time_budget=45)
    
    # Safety: ensure we return within deadline
    if time.time() - start_time > 55:
        return selected
    
    return selected
# EVOLVE-BLOCK-END
