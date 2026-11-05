"""Reciprocal weight of three-progression-free sets."""
import time
import random
from fractions import Fraction
import numpy as np

_N = 256
_DEADLINE_FRAC = 0.88  # use 88% of 90s = 79.2s safety margin

def _precompute_conflicts():
    """Precompute for each element x, the set of elements that conflict with x.
    
    Two elements a, b conflict if there exists c in [1, _N] such that
    {a, b, c} forms a 3-term AP. That is, for the triple (a, b, c) sorted,
    the middle element is the average of the other two.
    
    For a given pair (a, b), they conflict if:
    - c = 2*b - a is in [1, _N] (then a, b, c is AP with b as middle)
    - c = 2*a - b is in [1, _N] (then b, a, c is AP with a as middle) 
    - a = 2*c - b i.e. c = (a+b)/2 if a+b even (then a, c, b is AP with c as middle)
    - b = 2*c - a i.e. c = (a+b)/2 if a+b even (same as above)
    
    Actually, for any two distinct elements a, b, they are in conflict if there
    exists a third element c such that {a, b, c} is a 3-term AP.
    
    The three cases for a 3-AP containing both a and b:
    1. a, b, c: c = 2b - a (b is middle)
    2. b, a, c: c = 2a - b (a is middle)  
    3. c, a, b: c = 2a - b... wait, let me think more carefully.
    
    A 3-term AP is (x, y, z) with y - x = z - y, so 2y = x + z.
    If a and b are both in the AP, the possibilities are:
    - a is first, b is second: c = 2b - a (third term)
    - a is second, b is first: c = 2a - b (third term)
    - a is first, b is third: c = (a+b)/2 if a+b even (middle term)
    - a is third, b is first: c = (a+b)/2 if a+b even (middle term)
    - a is second, b is third: c = 2a - b (first term)
    - a is third, b is second: c = 2b - a (first term)
    
    So conflicts for pair (a, b) exist if:
    - 2b - a is in [1, _N] and 2b - a != a and 2b - a != b
    - 2a - b is in [1, _N] and 2a - b != a and 2a - b != b
    - (a + b) % 2 == 0 and (a+b)//2 is in [1, _N] and (a+b)//2 != a and (a+b)//2 != b
    """
    conflicts = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            # Case 1: a, b, c where c = 2b - a
            c = 2 * b - a
            if 1 <= c <= _N and c != a and c != b:
                conflicts[a].add(b)
                conflicts[b].add(a)
                # Also c conflicts with a and b
                conflicts[c].add(a)
                conflicts[a].add(c)
                conflicts[c].add(b)
                conflicts[b].add(c)
            
            # Case 2: b, a, c where c = 2a - b
            c = 2 * a - b
            if 1 <= c <= _N and c != a and c != b:
                conflicts[b].add(a)
                conflicts[a].add(b)
                conflicts[c].add(b)
                conflicts[b].add(c)
                conflicts[c].add(a)
                conflicts[a].add(c)
            
            # Case 3: c, a, b or c, b, a where c = (a+b)/2
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if 1 <= c <= _N and c != a and c != b:
                    conflicts[c].add(a)
                    conflicts[a].add(c)
                    conflicts[c].add(b)
                    conflicts[b].add(c)
                    conflicts[a].add(b)
                    conflicts[b].add(a)
    
    return conflicts

def _is_valid_3ap_free(selected_set):
    """Check if a set has no 3-term AP using precomputed conflicts."""
    # For each pair in selected, check if they conflict
    selected_list = list(selected_set)
    for i in range(len(selected_list)):
        for j in range(i + 1, len(selected_list)):
            a, b = selected_list[i], selected_list[j]
            # Check if adding both creates a 3-AP
            # They conflict if there's a third element that would complete the AP
            # But we only need to check if the set itself contains a 3-AP
            # Simpler: for each element in selected, check if it forms AP with two others
            pass
    
    # Better approach: for each triple, check
    s = selected_set
    for a in s:
        for b in s:
            if a >= b:
                continue
            c = 2 * b - a
            if c <= _N and c in s:
                return False
    return True

def _conflicts_with_new_element(x, selected_set, conflicts):
    """Check if adding x to selected_set creates any 3-AP.
    
    x creates a 3-AP if there exist a, b in selected_set such that
    {a, b, x} is a 3-term AP.
    """
    for a in selected_set:
        if a == x:
            continue
        # Check if a, x, c forms AP: c = 2x - a
        c = 2 * x - a
        if c <= _N and c in selected_set and c != a and c != x:
            return True
        # Check if x, a, c forms AP: c = 2a - x
        c = 2 * a - x
        if c <= _N and c in selected_set and c != a and c != x:
            return True
        # Check if c, x, a or c, a, x forms AP: c = (x+a)/2
        if (x + a) % 2 == 0:
            c = (x + a) // 2
            if c <= _N and c in selected_set and c != x and c != a:
                return True
    return False

def _greedy_initial(order, conflicts=None):
    """Greedy: add elements in given order if valid."""
    selected = set()
    for x in order:
        if not _conflicts_with_new_element(x, selected, conflicts):
            selected.add(x)
    return selected

def _reciprocal_sum(selected):
    """Compute sum of reciprocals."""
    return sum(1.0 / x for x in selected)

def _reciprocal_sum_exact(selected):
    """Compute exact sum of reciprocals as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _local_search(selected_set, conflicts, time_limit):
    """Local search: try to improve by adding, removing, or swapping elements."""
    selected = set(selected_set)
    best_score = _reciprocal_sum(selected)
    best_selected = set(selected)
    
    while True:
        t0 = time.time()
        if t0 > time_limit:
            break
        
        improved = False
        
        # Try adding elements (prefer small ones for high reciprocal)
        candidates = sorted([x for x in range(1, _N + 1) if x not in selected], 
                           key=lambda x: x)  # Small first for high 1/x
        for x in candidates:
            if time.time() > time_limit:
                break
            if not _conflicts_with_new_element(x, selected, conflicts):
                new_score = best_score + 1.0 / x
                if new_score > best_score + 1e-15:
                    selected.add(x)
                    best_score = new_score
                    best_selected = set(selected)
                    improved = True
        
        # Try removing one element and adding two (or one) to improve
        if not improved:
            # Try single removals that allow better additions
            for remove_x in sorted(selected, key=lambda x: -1.0/x):  # Remove low-value first
                if time.time() > time_limit:
                    break
                trial = selected - {remove_x}
                # Try adding elements to trial
                for add_x in range(1, _N + 1):
                    if add_x in trial:
                        continue
                    if not _conflicts_with_new_element(add_x, trial, conflicts):
                        new_set = trial | {add_x}
                        new_score = _reciprocal_sum(new_set)
                        if new_score > best_score + 1e-15:
                            selected = new_set
                            best_score = new_score
                            best_selected = set(selected)
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_selected, best_score

def _multi_start_greedy(conflicts, time_limit):
    """Try multiple greedy orderings and local search."""
    best_selected = set()
    best_score = 0.0
    
    # Ordering 1: increasing (small first)
    t0 = time.time()
    if time.time() < time_limit:
        order1 = list(range(1, _N + 1))
        sel1 = _greedy_initial(order1, conflicts)
        sel1, score1 = _local_search(sel1, conflicts, min(time_limit, time.time() + (time_limit - t0) * 0.4))
        if score1 > best_score:
            best_score = score1
            best_selected = sel1
    
    # Ordering 2: decreasing (large first) - might allow more small ones later
    t0 = time.time()
    if time.time() < time_limit:
        order2 = list(range(_N, 0, -1))
        sel2 = _greedy_initial(order2, conflicts)
        sel2, score2 = _local_search(sel2, conflicts, min(time_limit, time.time() + (time_limit - t0) * 0.3))
        if score2 > best_score:
            best_score = score2
            best_selected = sel2
    
    # Ordering 3: random with small bias
    t0 = time.time()
    if time.time() < time_limit:
        random.seed(42)
        order3 = list(range(1, _N + 1))
        # Shuffle but bias toward small numbers
        for i in range(len(order3) - 1, 0, -1):
            j = random.randint(0, i)
            # With 70% probability, swap toward smaller
            if random.random() < 0.7 and order3[j] < order3[i]:
                order3[i], order3[j] = order3[j], order3[i]
        sel3 = _greedy_initial(order3, conflicts)
        sel3, score3 = _local_search(sel3, conflicts, min(time_limit, time.time() + (time_limit - t0) * 0.3))
        if score3 > best_score:
            best_score = score3
            best_selected = sel3
    
    # More random starts if time allows
    remaining = time_limit - time.time()
    if remaining > 5:
        for seed in [1, 2, 3]:
            if time.time() > time_limit - 2:
                break
            random.seed(seed + 100)
            order = list(range(1, _N + 1))
            random.shuffle(order)
            # Bias: put smaller numbers earlier
            order.sort(key=lambda x: (x > 50, x))  # Small numbers first, then random among large
            sel = _greedy_initial(order, conflicts)
            sel, score = _local_search(sel, conflicts, min(time_limit, time.time() + 5))
            if score > best_score:
                best_score = score
                best_selected = sel
    
    return best_selected, best_score

def _fix_3aps(selected_set):
    """Remove elements to eliminate all 3-APs. Greedy: remove the element with smallest reciprocal."""
    selected = set(selected_set)
    changed = True
    while changed:
        changed = False
        # Find a 3-AP
        ap_found = None
        s_list = sorted(selected)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a, b = s_list[i], s_list[j]
                c = 2 * b - a
                if c <= _N and c in selected:
                    ap_found = (a, b, c)
                    break
            if ap_found:
                break
        if ap_found:
            # Remove the element with smallest reciprocal (largest value)
            a, b, c = ap_found
            candidates = [(a, 1.0/a), (b, 1.0/b), (c, 1.0/c)]
            candidates.sort(key=lambda x: x[1])  # Sort by reciprocal, remove smallest
            remove_x = candidates[0][0]
            selected.discard(remove_x)
            changed = True
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 90 * _DEADLINE_FRAC
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    # Multi-start greedy with local search
    best_selected, best_score = _multi_start_greedy(conflicts, deadline)
    
    # Ensure validity
    if not _is_valid_3ap_free(best_selected):
        best_selected = _fix_3aps(best_selected)
        best_score = _reciprocal_sum(best_selected)
    
    # Final check
    if not best_selected:
        # Fallback: just {1}
        best_selected = {1}
        best_score = 1.0
    
    return sorted(best_selected)
