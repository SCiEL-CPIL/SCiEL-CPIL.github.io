"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraint."""
    # For each a in selected + {x}, check if any triple of other selected members sums to a multiple of a
    new_set = selected + [x]
    n = len(new_set)
    for i in range(n):
        a = new_set[i]
        others = [new_set[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy(order):
    """Greedy selection following the given order."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Try to improve by removing one element and adding another."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            # Try removing selected[i] and adding a larger number not in set
            without_i = [selected[j] for j in range(len(selected)) if j != i]
            # Try adding numbers from high to low that aren't already in the set
            current_set = set(selected)
            for x in range(192, 0, -1):
                if x in current_set:
                    continue
                candidate = without_i + [x]
                if _is_valid(candidate):
                    if len(candidate) > len(selected):
                        selected = candidate
                        improved = True
                        break
                elif len(candidate) == len(selected):
                    # Same size but different composition - skip for now
                    pass
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Multiple orderings to try
    orderings = [
        list(range(192, 0, -1)),          # descending
        list(range(1, 193)),              # ascending
        list(range(96, 0, -1)) + list(range(192, 96, -1)),  # mid-high then high
        [x for x in range(192, 0, -1) if x > 96] + [x for x in range(96, 0, -1) if x > 48],  # top half
    ]
    
    best = []
    for order in orderings:
        if time.time() >= deadline:
            break
        result = _greedy(order)
        if len(result) > len(best):
            best = result
    
    # Local improvement on the best result
    if best and time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if best and not _is_valid(best):
        # Fallback to a simple valid set
        best = _greedy(list(range(192, 0, -1)))
    
    return best
# EVOLVE-BLOCK-END
