"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from fractions import Fraction

def _has_3ap_with(selected_set, x):
    """Check if adding x to selected_set creates a 3-term AP."""
    for a in selected_set:
        # Check if a, x, b form AP where b = 2x - a
        b = 2 * x - a
        if b != a and b != x and b in selected_set:
            return True
        # Check if x, a, b form AP where b = 2a - x
        b = 2 * a - x
        if b != a and b != x and b in selected_set:
            return True
        # Check if a, b, x form AP where b = (a + x) / 2
        if (a + x) % 2 == 0:
            b = (a + x) // 2
            if b != a and b != x and b in selected_set:
                return True
    return False

def _reciprocal_sum(selected):
    """Compute sum of reciprocals as float."""
    return sum(1.0 / x for x in selected)

def _exact_reciprocal_sum(selected):
    """Compute exact sum of reciprocals as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_by_reciprocal():
    """Greedy: always add the number with largest reciprocal (smallest value) that fits."""
    selected = set()
    # Priority queue: (-1/x, x) so smallest x (largest 1/x) comes first
    # We process in order of decreasing reciprocal, i.e., increasing x
    # But we want to TRY each number in order of largest reciprocal first
    # Since 1/x is decreasing in x, we process x from 1 to 256
    # But greedy by reciprocal means: at each step, pick the unselected x with max 1/x
    # that can be added. Since 1/x is monotone decreasing in x, this is just
    # trying x=1,2,3,... and adding if possible.
    
    # Actually, let's do a proper greedy: maintain a max-heap of candidate reciprocals
    # But since 1/x is determined by x, and we want max 1/x = min x,
    # the greedy is: iterate x from 1 to 256, add if no 3-AP.
    
    # Let's try a different greedy: process in order of 1/x descending
    # which is x ascending. So same as before. But let's also try other orders.
    
    selected = set()
    for x in range(1, 257):
        if not _has_3ap_with(selected, x):
            selected.add(x)
    return selected

def _local_search(initial_set, deadline):
    """Local search: try swaps to improve reciprocal sum."""
    selected = set(initial_set)
    best_sum = _reciprocal_sum(selected)
    best_set = set(selected)
    
    # Try adding any missing element that fits
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 257):
            if x not in selected:
                if not _has_3ap_with(selected, x):
                    selected.add(x)
                    changed = True
                    new_sum = _reciprocal_sum(selected)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = set(selected)
    
    # Try swaps: remove y, add x where 1/x > 1/y (x < y)
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if x not in selected:
            for y in sorted(selected):
                if time.time() > deadline:
                    break
                if y <= x:
                    continue
                # Try removing y and adding x
                new_set = selected - {y}
                if not _has_3ap_with(new_set, x):
                    new_set.add(x)
                    new_sum = _reciprocal_sum(new_set)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = set(new_set)
                        selected = set(new_set)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Start with greedy ascending
    initial = _greedy_by_reciprocal()
    
    # Apply local search
    result = _local_search(initial, deadline)
    
    return list(result)
# EVOLVE-BLOCK-END
