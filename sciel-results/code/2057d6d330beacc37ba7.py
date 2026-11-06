"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the no-divide condition."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating the condition."""
    candidate = selected + [x]
    n = len(candidate)
    if n < 4:
        return True
    
    # Check new element x: does x divide sum of any 3 others?
    others_without_x = selected
    for t in combinations(others_without_x, 3):
        if sum(t) % x == 0:
            return False
    
    # Check existing elements: does any existing element a divide sum of 3 others 
    # where one of the others is x?
    # For each a in selected, we need to check if there exist 2 other elements b,c in selected 
    # (b != a, c != a, b != c) such that (a + b + c) % x == 0 is NOT the condition...
    # Wait, the condition is: no selected integer may divide the sum of any 3 DISTINCT OTHER selected integers.
    # So for each existing a in selected, we check if a divides (x + b + c) for any distinct b,c in selected, b!=a, c!=a, b!=c.
    
    for a in selected:
        # We need to find if there exist b, c in selected such that b != a, c != a, b != c, and (x + b + c) % a == 0
        # To optimize, we can iterate over pairs (b, c) in selected where b != a and c != a and b != c
        # But that's O(n^2) for each a, so O(n^3) total.
        # Alternatively, for each a, we can precompute sums of pairs?
        # Let's just do the direct check for now, since n is small (max 48).
        for b in selected:
            if b == a:
                continue
            for c in selected:
                if c == a or c == b:
                    continue
                if (x + b + c) % a == 0:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    # Try multiple greedy orders and local search improvements
    best_selected = []
    best_size = 0
    
    # First, run the original greedy from high to low
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    if len(selected) > best_size:
        best_selected = selected[:]
        best_size = len(selected)
    
    # Try greedy from low to high
    selected = []
    for x in range(1, 193):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    if len(selected) > best_size:
        best_selected = selected[:]
        best_size = len(selected)
    
    # Try to improve best_selected via local search: try swapping out one element for another
    if best_size > 0:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = best_selected[:]
            current_set = set(current)
            
            # Try removing each element and adding a better one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                reduced = current[:i] + current[i+1:]
                
                # Try adding each number not in current
                for x in range(192, 0, -1):
                    if x in current_set:
                        continue
                    if _can_add(reduced, x):
                        new_set = reduced + [x]
                        if len(new_set) > len(best_selected):
                            best_selected = new_set
                            best_size = len(best_selected)
                            improved = True
                            break
                        # Even if same size, we might want to swap for a different configuration
                        # But for now, let's just look for size increases
                if improved:
                    break
            # Also try removing two and adding two? Too expensive. Stick with single swaps.
    
    return best_selected
# EVOLVE-BLOCK-END
