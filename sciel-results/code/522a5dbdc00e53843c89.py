"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add(selected, x):
    """Check if x can be added to selected without violating the condition."""
    n = len(selected)
    
    # Check 1: Does x divide sum of any 3 distinct in selected?
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    
    # Check 2: Does any a in selected divide sum of 3 distinct in selected U {x} excluding a?
    # Only need to check triples including x: {x, b, c} with b,c in selected\{a}, b!=c.
    if n >= 2:
        for a in selected:
            # Iterate pairs from selected excluding a
            # We can optimize by breaking early
            for i in range(n):
                b = selected[i]
                if b == a:
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    c = selected[j]
                    if c == a:
                        continue
                    if (x + b + c) % a == 0:
                        return False
    return True

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

def _greedy(order, deadline):
    """Run greedy selection in the given order, respecting deadline."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Local search: try to increase size by removing one and adding two."""
    current = selected[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current)
        
        # Try removing each element
        for i in range(len(current)):
            if time.time() > deadline:
                break
            removed = current[i]
            reduced = current[:i] + current[i+1:]
            reduced_set = set(reduced)
            
            # Try adding elements not in reduced, high to low
            added_one = False
            for x in range(192, 0, -1):
                if time.time() > deadline:
                    break
                if x in reduced_set:
                    continue
                if _can_add(reduced, x):
                    new_set = reduced + [x]
                    new_set_set = set(new_set)
                    added_one = True
                    # Try adding one more
                    for y in range(192, 0, -1):
                        if time.time() > deadline:
                            break
                        if y in new_set_set:
                            continue
                        if _can_add(new_set, y):
                            final_set = new_set + [y]
                            if len(final_set) > len(current):
                                current = final_set
                                improved = True
                            break
                    break
            if improved:
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_selected = []
    best_size = 0
    
    def update_best(sel):
        nonlocal best_selected, best_size
        if len(sel) > best_size:
            best_selected = sel[:]
            best_size = len(best_selected)
    
    # Strategy 1: Deterministic Greedy (High to Low)
    selected = _greedy(range(192, 0, -1), deadline)
    update_best(selected)
    
    # Strategy 2: Deterministic Greedy (Low to High)
    if time.time() < deadline:
        selected = _greedy(range(1, 193), deadline)
        update_best(selected)
    
    # Strategy 3: Randomized Greedy - multiple runs
    random.seed(42)
    while time.time() < deadline - 10:  # Keep 10s for local search
        if time.time() > deadline:
            break
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        selected = _greedy(candidates, deadline)
        update_best(selected)
        
        if best_size == 48:
            break
    
    # Local search on the best found set
    if best_size > 0 and time.time() < deadline:
        current = _local_search(best_selected, deadline)
        update_best(current)
    
    # Additional randomized local search
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline:
                break
            candidates = list(range(1, 193))
            random.shuffle(candidates)
            selected = _greedy(candidates, deadline)
            if len(selected) >= best_size:
                current = _local_search(selected, deadline)
                update_best(current)
    
    return best_selected
# EVOLVE-BLOCK-END
