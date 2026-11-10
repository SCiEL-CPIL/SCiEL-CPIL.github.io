"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # 80s safety margin before 90s kill
    n = 192
    k = 3
    max_size = 48

    # Precompute for speed: for each candidate x and each pair (a,b) of selected,
    # we need to check if x divides a+b+c for any c in selected (c != a,b,x)
    # Actually the constraint is: for each selected element a, and each triple (x,y,z) of
    # DISTINCT OTHER selected elements, a must not divide x+y+z.
    
    # We'll use a backtracking search with good heuristics.
    # Key insight: larger numbers are "safer" because they're less likely to divide small sums.
    # Also, numbers that are prime or have few divisors are safer.
    
    # Strategy:
    # 1. Start with a greedy construction from large to small.
    # 2. Try to improve via local search (swap, add, remove).
    
    def is_valid_addition(candidate, selected):
        """Check if adding candidate to selected maintains the constraint."""
        # For the new candidate: it must not divide the sum of any 3 distinct OTHER selected elements
        if len(selected) >= 3:
            for combo in combinations(selected, 3):
                if sum(combo) % candidate == 0:
                    return False
        # For each existing element a in selected: a must not divide (candidate + b + c)
        # for any pair (b,c) from selected \ {a}
        for a in selected:
            others = [x for x in selected if x != a]
            # We need to check if a divides candidate + b + c for any distinct b,c in others
            # This is O(len(others)^2) per a
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if (candidate + others[i] + others[j]) % a == 0:
                        return False
        return True
    
    def is_set_valid(selected):
        """Verify the entire set is valid."""
        n_sel = len(selected)
        for idx, a in enumerate(selected):
            others = [selected[i] for i in range(n_sel) if i != idx]
            if len(others) < 3:
                continue
            for combo in combinations(others, 3):
                if sum(combo) % a == 0:
                    return False
        return True
    
    # Greedy construction: try numbers from large to small
    selected = []
    for x in range(n, 0, -1):
        if is_valid_addition(x, selected):
            selected.append(x)
        if time.time() > deadline - 10:
            break
    
    best_size = len(selected)
    best_set = list(selected)
    
    # Local search: try to add more elements or swap to increase size
    # Try adding elements that were skipped
    remaining = [x for x in range(1, n+1) if x not in selected]
    
    # Try to add one element at a time
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding a single element
        for x in remaining:
            if x in selected:
                continue
            if is_valid_addition(x, selected):
                selected.append(x)
                remaining.remove(x)
                improved = True
                best_size = len(selected)
                best_set = list(selected)
                break
        if not improved:
            # Try swapping: remove one element, add another
            for i, a in enumerate(selected):
                if time.time() > deadline - 5:
                    break
                new_selected = selected[:i] + selected[i+1:]
                for x in remaining:
                    if is_valid_addition(x, new_selected):
                        selected = new_selected + [x]
                        remaining.remove(x)
                        selected.append(a)
                        remaining.append(a)
                        improved = True
                        best_size = len(selected)
                        best_set = list(selected)
                        break
                if improved:
                    break
    
    # Verify final result
    if not is_set_valid(best_set):
        # Fallback to a simpler valid set
        best_set = []
        for x in range(n, 0, -1):
            if is_valid_addition(x, best_set):
                best_set.append(x)
    
    return best_set
# EVOLVE-BLOCK-END
