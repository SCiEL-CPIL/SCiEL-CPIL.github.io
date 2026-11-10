"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _reciprocal_weight(selected):
    """Return the exact sum of reciprocals as a Fraction."""
    s = Fraction(0, 1)
    for x in selected:
        s += Fraction(1, x)
    return s

def _is_3ap_free(selected):
    """Check if selected set has no nonconstant 3-term AP."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                c = 2*b - a
                if c in s:
                    return False
    return True

def _greedy_initial():
    """Greedy construction: iterate 1..256, include if no 3-AP formed."""
    selected = []
    sel_set = set()
    for x in range(1, 257):
        # Check if adding x creates a 3-AP
        conflict = False
        for a in selected:
            # x could be the middle: a, x, c => c = 2x - a
            c = 2*x - a
            if c in sel_set and c != x:
                conflict = True
                break
            # x could be the end: a, b, x => b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in sel_set and b != x and b != a:
                    conflict = True
                    break
        if not conflict:
            selected.append(x)
            sel_set.add(x)
    return selected

def _backtrack_optimize(selected, time_limit=85.0):
    """
    Backtracking search to improve the reciprocal weight.
    We try to swap out elements with lower reciprocal value to allow
    inclusion of elements with higher reciprocal value.
    
    Strategy: Start from greedy solution. Try local moves:
    1. Remove an element, see if we can add two or more elements.
    2. Try different orderings.
    """
    best = list(selected)
    best_weight = _reciprocal_weight(best)
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without forming 3-AP."""
        for a in current_set:
            # x as middle: a, x, c
            c = 2*x - a
            if c in current_set and c != x:
                return False
            # x as end: a, b, x
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in current_set and b != x and b != a:
                    return False
        return True
    
    def try_improve(current, current_set, current_weight):
        """Try local improvements."""
        nonlocal best, best_weight
        if elapsed() > time_limit - 2:
            return
        
        # Try removing each element and see if we can add others
        for i in range(len(current)):
            if elapsed() > time_limit - 2:
                break
            removed = current[i]
            new_set = current_set - {removed}
            new_current = [x for x in current if x != removed]
            
            # Try adding each excluded element
            for x in range(1, 257):
                if x in new_set:
                    continue
                if not can_add(x, new_set):
                    continue
                
                new_set2 = new_set | {x}
                new_current2 = new_current + [x]
                new_weight = current_weight - Fraction(1, removed) + Fraction(1, x)
                
                if new_weight > best_weight:
                    # Further try to add more elements
                    further_improved = False
                    for y in range(1, 257):
                        if y in new_set2:
                            continue
                        if not can_add(y, new_set2):
                            continue
                        new_set3 = new_set2 | {y}
                        new_current3 = new_current2 + [y]
                        new_weight3 = new_weight - Fraction(1, y) + Fraction(1, y)  # same as new_weight + 1/y
                        new_weight3 = new_weight + Fraction(1, y)
                        if new_weight3 > best_weight:
                            best = list(new_current3)
                            best_weight = new_weight3
                            further_improved = True
                    if not further_improved:
                        best = list(new_current2)
                        best_weight = new_weight
    
    try_improve(list(selected), set(selected), best_weight)
    
    return best, best_weight

def _hill_climbing(selected, time_limit=85.0):
    """
    More systematic hill climbing: try all single-element swaps and 
    double-element swaps.
    """
    best = list(selected)
    best_set = set(best)
    best_weight = _reciprocal_weight(best)
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def can_add(x, s):
        """Check if x can be added to set s without forming 3-AP."""
        for a in s:
            c = 2*x - a
            if c in s and c != x:
                return False
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in s and b != x and b != a:
                    return False
        return True
    
    def try_single_swap(s, weight):
        """Try removing one element and adding one or more."""
        nonlocal best, best_set, best_weight
        for i in range(len(s)):
            if elapsed() > time_limit - 2:
                break
            removed = s[i]
            new_s = s - {removed}
            
            # Try adding each possible element
            gains = []
            for x in range(1, 257):
                if x in new_s:
                    continue
                if can_add(x, new_s):
                    gains.append(x)
            
            # Try all subsets of gains (small ones first)
            for x in gains:
                new_s2 = new_s | {x}
                new_w = weight - Fraction(1, removed) + Fraction(1, x)
                if new_w > best_weight:
                    best = list(new_s2)
                    best_set = set(new_s2)
                    best_weight = new_w
            
            # Try pairs from gains
            for i2 in range(len(gains)):
                if elapsed() > time_limit - 2:
                    break
                for j2 in range(i2+1, len(gains)):
                    x, y = gains[i2], gains[j2]
                    if not can_add(y, new_s | {x}):
                        continue
                    new_s3 = new_s | {x, y}
                    new_w = weight - Fraction(1, removed) + Fraction(1, x) + Fraction(1, y)
                    if new_w > best_weight:
                        best = list(new_s3)
                        best_set = set(new_s3)
                        best_weight = new_w
    
    # Iterate until no improvement
    improved = True
    while improved and elapsed() < time_limit - 3:
        improved = False
        try_single_swap(best_set, best_weight)
        if best_weight > _reciprocal_weight(best):
            # This shouldn't happen, but just in case
            pass
        # Check if we actually improved
        # The function updates best directly, so check if best changed
        # Actually, let's track it differently
        old_best_weight = best_weight
        try_single_swap(best_set, best_weight)
        if best_weight > old_best_weight:
            improved = True
    
    return best, best_weight

def _simulated_annealing(selected, time_limit=85.0):
    """
    Simulated annealing to escape local optima.
    """
    import random
    best = list(selected)
    best_set = set(best)
    best_weight = _reciprocal_weight(best)
    current = list(best)
    current_set = set(current)
    current_weight = best_weight
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def can_add(x, s):
        for a in s:
            c = 2*x - a
            if c in s and c != x:
                return False
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in s and b != x and b != a:
                    return False
        return True
    
    T = 0.1
    cooling = 0.999
    
    while elapsed() < time_limit - 3 and T > 0.001:
        if random.random() < 0.5:
            # Try to add a random element
            candidates = [x for x in range(1, 257) if x not in current_set]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            if can_add(x, current_set):
                new_set = current_set | {x}
                new_w = current_weight + Fraction(1, x)
                delta = float(new_w - current_weight)
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_set = new_set
                    current_weight = new_w
                    if new_w > best_weight:
                        best_set = set(new_set)
                        best_weight = new_w
                        best = list(new_set)
        else:
            # Try to remove a random element
            if not current:
                T *= cooling
                continue
            x = random.choice(current)
            new_set = current_set - {x}
            new_w = current_weight - Fraction(1, x)
            delta = float(new_w - current_weight)
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = new_set
                current_weight = new_w
                if new_w > best_weight:
                    best_set = set(new_set)
                    best_weight = new_w
                    best = list(new_set)
        
        T *= cooling
    
    return best, best_weight

def construct():
    """
    Construct a 3-AP-free subset of [1,256] maximizing sum of reciprocals.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with greedy
    selected = _greedy_initial()
    best = list(selected)
    best_weight = _reciprocal_weight(best)
    
    # Hill climbing
    if time.time() < deadline - 10:
        best, best_weight = _hill_climbing(best, time_limit=deadline - start_time - 5)
    
    # Simulated annealing
    if time.time() < deadline - 3:
        best, best_weight = _simulated_annealing(best, time_limit=deadline - start_time - 2)
    
    # Final verification
    assert _is_3ap_free(best), "Result contains a 3-term AP!"
    assert all(1 <= x <= 256 for x in best), "Elements out of range!"
    assert len(best) > 0, "Empty result!"
    
    return best
# EVOLVE-BLOCK-END
