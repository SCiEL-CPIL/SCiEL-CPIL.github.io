import time
import random
from fractions import Fraction

def _has_3ap(S):
    """Check if set S has any 3-term arithmetic progression."""
    s = set(S)
    for a in s:
        for b in s:
            if a < b:
                c = 2*b - a
                if c in s and c != a:
                    return True
    return False

def _is_valid(S):
    """Check if S is a valid 3-AP-free set."""
    return not _has_3ap(S)

def _reciprocal_sum(S):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _greedy_by_reciprocal():
    """Greedy: add elements in order of largest reciprocal (smallest x) if valid."""
    selected = set()
    # Try adding from 1 to 256 (smallest first = largest reciprocal)
    for x in range(1, 257):
        if _is_valid(selected | {x}):
            selected.add(x)
    return selected

def _greedy_by_value_then_size():
    """Greedy with different ordering strategies."""
    # Strategy 1: smallest first
    s1 = _greedy_by_reciprocal()
    
    # Strategy 2: try to maximize count first, then optimize
    # Use a different greedy: process in order but skip if it creates AP
    selected = set()
    for x in range(1, 257):
        # Check if adding x creates a 3-AP
        valid = True
        for a in selected:
            for b in selected:
                if a < b:
                    c = 2*b - a
                    if c == x:
                        valid = False
                        break
                # Also check if x is the middle or first
                # x, b, c: 2b - x = c => c = 2b - x
                # a, x, c: 2x - a = c
                # a, b, x: 2b - a = x
            if not valid:
                break
        # More careful check
        if valid:
            valid = True
            for a in selected:
                # Check if x, a, 2a-x is in selected
                c1 = 2*a - x
                if c1 in selected and c1 != x:
                    valid = False
                    break
                # Check if a, x, 2x-a is in selected
                c2 = 2*x - a
                if c2 in selected and c2 != x:
                    valid = False
                    break
                # Check if a, b, x where 2b-a=x, so b=(a+x)/2
                if (a + x) % 2 == 0:
                    b = (a + x) // 2
                    if b in selected and b != a and b != x:
                        valid = False
                        break
        if valid:
            selected.add(x)
    
    s2 = selected
    
    # Strategy 3: random restarts
    best = max(s1, s2, key=lambda S: _reciprocal_sum(S))
    
    return best

def _local_search(start_set, max_iter=5000):
    """Local search: try swaps and additions to improve reciprocal sum."""
    S = set(start_set)
    current_sum = _reciprocal_sum(S)
    best_set = S.copy()
    best_sum = current_sum
    
    # Precompute all possible elements not in S
    all_elements = set(range(1, 257))
    
    for iteration in range(max_iter):
        improved = False
        
        # Try adding elements
        candidates = list(all_elements - S)
        random.shuffle(candidates)
        
        for x in candidates:
            new_S = S | {x}
            if _is_valid(new_S):
                new_sum = current_sum + Fraction(1, x)
                if new_sum > current_sum:
                    S = new_S
                    current_sum = new_sum
                    improved = True
                    break
        
        # Try removing and re-adding
        if not improved:
            # Try swap: remove one, add another
            removed_candidates = list(S)
            random.shuffle(removed_candidates)
            
            for a in removed_candidates:
                # Try to find an element to add that gives better sum
                without_a = S - {a}
                base_sum = _reciprocal_sum(without_a)
                
                for x in (all_elements - without_a):
                    new_S = without_a | {x}
                    if _is_valid(new_S):
                        new_sum = base_sum + Fraction(1, x)
                        if new_sum > current_sum:
                            S = new_S
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
        
        if _reciprocal_sum(S) > best_sum:
            best_sum = _reciprocal_sum(S)
            best_set = S.copy()
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Start with greedy
    best_set = _greedy_by_reciprocal()
    best_sum = _reciprocal_sum(best_set)
    
    # Try multiple local search restarts with different initial sets
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Create a perturbed initial set
        S = set(best_set)
        
        # Randomly remove some elements
        n_remove = random.randint(0, min(5, len(S)))
        to_remove = random.sample(list(S), n_remove)
        for x in to_remove:
            S.discard(x)
        
        # Add some random elements
        all_elements = set(range(1, 257))
        available = list(all_elements - S)
        n_add = random.randint(0, min(3, len(available)))
        if available:
            to_add = random.sample(available, n_add)
            for x in to_add:
                if _is_valid(S | {x}):
                    S.add(x)
        
        # Local search
        result = _local_search(S, max_iter=2000)
        result_sum = _reciprocal_sum(result)
        
        if result_sum > best_sum:
            best_set = result
            best_sum = result_sum
    
    return list(best_set)
