"""Improved program: maximize reciprocal weight of 3-AP-free subset of [1,256]."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

def has_3ap(s_set):
    """Check if set contains a nonconstant 3-term AP. s_set should be a set."""
    for a in s_set:
        for b in s_set:
            if a == b:
                continue
            c = 2 * b - a
            if c != a and c != b and c in s_set:
                return True
    return False

def reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def is_3ap_free(selected_set):
    """Check if a set is 3-AP-free."""
    return not has_3ap(selected_set)

def greedy_construct():
    """Greedy: include x if it doesn't create a 3-AP with previously included numbers."""
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 3-AP
        ok = True
        for a in selected:
            for b in selected:
                if a == b:
                    continue
                c = 2 * b - a
                if c == x:
                    ok = False
                    break
                # Also check if x is the middle or first term
                # If x is middle: x = (a+c)/2 => c = 2x - a
                # If x is first: x, b, c => c = 2b - x
                # We need to check all three positions
                # Actually, the condition "no three distinct elements form a 3-AP" 
                # means: for any a,b in selected, 2b-a should not be in selected∪{x}
                # and 2a-b should not be in selected∪{x}, etc.
                # Simpler: check all pairs in selected∪{x}
            if not ok:
                break
        if ok:
            # Need to check properly
            new_set = selected | {x}
            if is_3ap_free(new_set):
                selected = new_set
    return selected

def local_search(initial_set, time_limit):
    """Local search to improve the 3-AP-free set."""
    start_time = time.time()
    current = set(initial_set)
    current_score = reciprocal_sum(current)
    best = set(current)
    best_score = current_score
    
    # Try to add elements
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try adding each element not in current
        for x in range(1, N + 1):
            if x in current:
                continue
            if time.time() - start_time > time_limit:
                break
            new_set = current | {x}
            if is_3ap_free(new_set):
                new_score = current_score + Fraction(1, x)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best = set(current)
                        best_score = current_score
        
        if not improved:
            break
    
    # Try swaps: remove one element, try to add two others
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try removing each element
        for x in list(current):
            if time.time() - start_time > time_limit:
                break
            reduced = current - {x}
            reduced_score = current_score - Fraction(1, x)
            
            # Try to add elements to reduced
            for y in range(1, N + 1):
                if y in reduced:
                    continue
                if time.time() - start_time > time_limit:
                    break
                new_set1 = reduced | {y}
                if not is_3ap_free(new_set1):
                    continue
                score1 = reduced_score + Fraction(1, y)
                
                # Try to add one more
                for z in range(y + 1, N + 1):
                    if z in new_set1:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    new_set2 = new_set1 | {z}
                    if not is_3ap_free(new_set2):
                        continue
                    score2 = score1 + Fraction(1, z)
                    
                    if score2 > current_score:
                        current = new_set2
                        current_score = score2
                        improved = True
                        if current_score > best_score:
                            best = set(current)
                            best_score = current_score
                        break  # Found improvement, move to next x
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    greedy_set = greedy_construct()
    greedy_score = reciprocal_sum(greedy_set)
    
    # Phase 2: Local search
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 2.0:
        improved_set = local_search(greedy_set, remaining_time - 1.0)
        improved_score = reciprocal_sum(improved_set)
        if improved_score > greedy_score:
            greedy_set = improved_set
            greedy_score = improved_score
    
    # Convert to sorted list
    result = sorted(greedy_set)
    return result
# EVOLVE-BLOCK-END
