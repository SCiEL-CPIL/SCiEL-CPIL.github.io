"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add(selected, x):
    """Check if x can be added to selected without violating the condition."""
    n = len(selected)
    if n < 3:
        # x cannot divide sum of 3 others if there are fewer than 3 others
        # But existing elements might be affected? No, existing elements only divide sums of 3 others.
        # If n < 3, no triple of others exists for x.
        # For existing a, can a divide sum of 3 others including x?
        # Others for a are from selected U {x} excluding a.
        # If n=0, others={x}. Size 1. No triples.
        # If n=1, others={x, other}. Size 2. No triples.
        # If n=2, others={x, o1, o2}. Size 3. One triple: x+o1+o2.
        # So we need to check if any a in selected divides (x + b + c) for distinct b,c in selected\{a}.
        
        # Check 1: Does x divide sum of any 3 distinct in selected?
        if n >= 3:
            for t in combinations(selected, 3):
                if sum(t) % x == 0:
                    return False
        else:
            # n < 3, no triples from selected
            pass
        
        # Check 2: Does any a in selected divide sum of 3 distinct in selected U {x} excluding a?
        # The triple must include x?
        # If the triple does NOT include x, it was already valid in 'selected'.
        # So we only need to check triples that include x.
        # Triple is {x, b, c} where b, c in selected, b != a, c != a, b != c.
        if n >= 2:
            for a in selected:
                # Find pairs (b, c) from selected \ {a}
                # If n=2, and a is one of them, only 1 left. No pair.
                # If n=2, and a is not... wait a IS in selected.
                # Let's just iterate pairs from selected.
                for b in selected:
                    if b == a:
                        continue
                    for c in selected:
                        if c == a or c == b:
                            continue
                        if (x + b + c) % a == 0:
                            return False
        return True
    
    # General case n >= 3
    
    # Check 1: Does x divide sum of any 3 distinct in selected?
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            return False
    
    # Check 2: Does any a in selected divide sum of 3 distinct in selected U {x} excluding a?
    # Only need to check triples including x: {x, b, c} with b,c in selected\{a}, b!=c.
    for a in selected:
        # Iterate pairs from selected excluding a
        # Optimization: precompute? No, n is small.
        # To avoid duplicate pairs (b,c) and (c,b), we can just use indices or sets.
        # Since we just need existence, simple loops are fine.
        found = False
        # We can break early if found
        # Let's use a set of pairs to avoid checking same pair twice? Or just indices.
        sel_list = selected
        for i in range(len(sel_list)):
            b = sel_list[i]
            if b == a:
                continue
            for j in range(i+1, len(sel_list)):
                c = sel_list[j]
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

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    best_selected = []
    best_size = 0
    
    # Helper to update best
    def update_best(sel):
        nonlocal best_selected, best_size
        if len(sel) > best_size:
            # Optional: verify validity to be safe, though _can_add should guarantee it
            # if _is_valid(sel): 
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
    # Inspiration 3 suggests random shuffles can find different local optima.
    # We will run multiple random shuffles.
    # To speed up, we might restrict the pool or just shuffle all 192.
    # 192 elements, O(n^3) check is ~ 192^3 / 6 ~ 1.2M ops per add? No, combinations of 3 from n.
    # n grows to ~40. C(40,3) = 9880. 40 * 9880 ~ 400k ops.
    # 192 adds * 400k = 76M ops. In Python, this is slow.
    # _can_add is O(n^3).
    # Let's estimate time for one full greedy run.
    # If it takes ~1s, we can do ~50 runs in 55s.
    
    num_random_runs = 0
    while time.time() < deadline - 5: # Keep 5s for local search if possible
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
        
        # If we hit 48, stop
        if best_size == 48:
            break
            
    # Local Search on the best found set
    # Try to swap out one element for another to see if we can increase size?
    # Swapping maintains size. To increase size, we need to add without removing.
    # But if greedy got stuck, maybe removing one "bad" element allows adding two "good" ones?
    # Or maybe the greedy order was bad.
    
    # A better local search: 
    # 1. Take best_selected.
    # 2. Try removing each element.
    # 3. Try adding any element not in set.
    # 4. If we can add more than 1 element (i.e., size increases), keep it.
    
    if best_size > 0 and time.time() < deadline:
        current = best_selected[:]
        current_set = set(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing one element
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                reduced = current[:i] + current[i+1:]
                reduced_set = set(reduced)
                
                # Try adding elements not in reduced
                # Prioritize larger numbers? Or random?
                # Let's try adding from high to low, as larger numbers are less likely to divide sums.
                for x in range(192, 0, -1):
                    if x in reduced_set:
                        continue
                    if _can_add(reduced, x):
                        new_set = reduced + [x]
                        # If we added x, size is same as current.
                        # Can we add another one?
                        # Try to add one more to new_set
                        new_set_set = set(new_set)
                        for y in range(192, 0, -1):
                            if y in new_set_set:
                                continue
                            if _can_add(new_set, y):
                                # Size increased by 1 (removed 1, added 2)
                                final_set = new_set + [y]
                                if len(final_set) > best_size:
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
