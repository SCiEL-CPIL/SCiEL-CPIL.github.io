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
    for a in selected:
        others = [v for v in selected if v != a]
        m = len(others)
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if (others[i] + others[j] + others[k]) % a == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_selected = []
    best_size = 0
    
    # Helper to update best with strict validity check
    def update_best(sel):
        nonlocal best_selected, best_size
        if len(sel) > best_size:
            if _is_valid(sel):
                best_selected = sel[:]
                best_size = len(best_selected)
    
    # Strategy 1: Deterministic Greedy (High to Low) - Baseline
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    update_best(selected)
    
    # Strategy 2: Deterministic Greedy (Low to High)
    selected = []
    for x in range(1, 193):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    update_best(selected)
    
    # Strategy 3: Randomized Greedy
    num_random_runs = 0
    while time.time() < deadline - 5:
        if time.time() > deadline:
            break
        
        selected = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        for x in candidates:
            if _can_add(selected, x):
                selected.append(x)
        
        update_best(selected)
        num_random_runs += 1
        
        if best_size == 48:
            break
    
    # Local Search on the best found set
    if best_size > 0 and time.time() < deadline:
        current = best_selected[:]
        current_set = set(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                reduced = current[:i] + current[i+1:]
                reduced_set = set(reduced)
                
                for x in range(192, 0, -1):
                    if x in reduced_set:
                        continue
                    if _can_add(reduced, x):
                        new_set = reduced + [x]
                        new_set_set = set(new_set)
                        for y in range(192, 0, -1):
                            if y in new_set_set:
                                continue
                            if _can_add(new_set, y):
                                final_set = new_set + [y]
                                if len(final_set) > best_size and _is_valid(final_set):
                                    best_selected = final_set
                                    best_size = len(best_selected)
                                    current = final_set[:]
                                    current_set = set(current)
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
                    
    return best_selected
# EVOLVE-BLOCK-END
