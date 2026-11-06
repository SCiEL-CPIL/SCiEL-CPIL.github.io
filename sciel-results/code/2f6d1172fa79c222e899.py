"""Improved program: maximize reciprocal weight of 3-AP-free subset of [1,256]."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256
DEADLINE_FRACTION = 0.85  # safety margin: stop at 85% of 60s = 51s

def _is_3ap_free(selected):
    """Check if a set of ints has no 3-term arithmetic progression."""
    s = set(selected)
    for a in s:
        for b in s:
            if a < b:
                c = 2 * b - a
                if c != b and c in s:
                    return False
    return True

def _reciprocal_sum(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_descending():
    """Greedy: add numbers in descending order of 1/x (i.e., ascending x), 
    but actually we want to maximize sum of 1/x, so prefer small numbers.
    Greedy by smallest x first (highest reciprocal)."""
    selected = []
    for x in range(1, N + 1):
        # Check if adding x creates a 3-AP
        ok = True
        for a in selected:
            for b in selected:
                if a < b:
                    c = 2 * b - a
                    if c == x:
                        ok = False
                        break
                if a > b:
                    c = 2 * a - b
                    if c == x:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            # Also check: does x as the middle term create AP?
            # 2*x - a in selected for some a, or 2*x - b... 
            # Actually the check above covers: for any a,b in selected, 
            # if 2b-a == x (x is the third term), or we need to check
            # if x is first or second term.
            # Let me redo this more carefully.
            ok2 = True
            for a in selected:
                # x, a, c: 2a - x = c, need c in selected
                c = 2 * a - x
                if c != x and c in set(selected):
                    ok2 = False
                    break
                # a, x, c: 2x - a = c, need c in selected
                c = 2 * x - a
                if c != x and c in selected:
                    ok2 = False
                    break
            if not ok2:
                ok = False
        if ok:
            selected.append(x)
    return selected

def _greedy_by_weight():
    """Greedy: try to add each number, prioritized by 1/x (smallest x first)."""
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates any 3-AP with existing elements
        conflict = False
        for a in selected:
            # Case 1: x is the largest: a, b, x where b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in selected and b != a and b != x:
                    conflict = True
                    break
            # Case 2: x is the middle: a, x, c where c = 2x - a
            c = 2 * x - a
            if c != x and c in selected:
                conflict = True
                break
            # Case 3: x is the smallest: x, b, a where b = (x+a)/2
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if b in selected and b != x and b != a:
                    conflict = True
                    break
        if not conflict:
            selected.add(x)
    return selected

def _local_search(initial, time_limit):
    """Local search: try to improve by adding/removing/swapping elements."""
    selected = set(initial)
    current_sum = _reciprocal_sum(list(selected))
    best_sum = current_sum
    best_set = set(selected)
    
    deadline = time.time() + time_limit
    
    # Precompute: for each pair in selected, what third element would form AP
    def _conflicts_with(x, s):
        """Return True if x conflicts with any element in set s to form a 3-AP."""
        for a in s:
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in s and b != a and b != x:
                    return True
            c = 2 * x - a
            if c != x and c in s:
                return True
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if b in s and b != x and b != a:
                    return True
        return False
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements not in selected
        for x in range(1, N + 1):
            if x in selected:
                continue
            if not _conflicts_with(x, selected):
                new_set = selected | {x}
                new_sum = current_sum + Fraction(1, x)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_set = new_set
                    selected = new_set
                    current_sum = new_sum
                    improved = True
        
        # Try removing and re-adding to get better combinations
        # For each element in selected, try removing it and see if we can add something better
        for x in list(selected):
            if time.time() >= deadline:
                break
            trial = selected - {x}
            # Try adding all non-selected elements to trial
            best_add = None
            best_add_sum = current_sum - Fraction(1, x)
            for y in range(1, N + 1):
                if y in trial:
                    continue
                if not _conflicts_with(y, trial):
                    add_sum = best_add_sum + Fraction(1, y)
                    if add_sum > best_add_sum:
                        best_add_sum = add_sum
                        best_add = y
            if best_add is not None and best_add_sum > current_sum:
                selected = trial | {best_add}
                current_sum = best_add_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = set(selected)
                improved = True
    
    return list(best_set), best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 51.0  # 85% of 60s
    
    # Start with greedy
    selected = _greedy_by_weight()
    current_sum = _reciprocal_sum(selected)
    best_sum = current_sum
    best_set = set(selected)
    
    # Local search
    time_left = deadline - time.time()
    if time_left > 1.0:
        selected, current_sum = _local_search(selected, time_left - 1.0)
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = set(selected)
    
    # Try a few different greedy strategies if time permits
    time_left = deadline - time.time()
    if time_left > 2.0:
        # Greedy descending (largest x first)
        sel2 = []
        for x in range(N, 0, -1):
            ok = True
            for a in sel2:
                if (a + x) % 2 == 0:
                    b = (a + x) // 2
                    if b in sel2 and b != a and b != x:
                        ok = False
                        break
                c = 2 * x - a
                if c != x and c in sel2:
                    ok = False
                    break
                if (x + a) % 2 == 0:
                    b = (x + a) // 2
                    if b in sel2 and b != x and b != a:
                        ok = False
                        break
            if ok:
                sel2.append(x)
        sum2 = _reciprocal_sum(sel2)
        if sum2 > best_sum:
            best_sum = sum2
            best_set = set(sel2)
    
    # Verify final result
    if not best_set:
        best_set = {1}
    
    result = list(best_set)
    # Final validation
    s = set(result)
    for a in s:
        for b in s:
            if a < b:
                c = 2 * b - a
                if c != b and c in s:
                    # Invalid - fall back to greedy
                    result = list(_greedy_by_weight())
                    break
        else:
            continue
        break
    
    return result
# EVOLVE-BLOCK-END
