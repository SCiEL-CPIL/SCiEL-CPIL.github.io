"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 128}

def _can_add(x, selected):
    """Check if adding x to selected creates a forbidden triple."""
    s = set(selected)
    # For each pair (a, b) in selected, check if a+b, a+x, b+x are all in s ∪ {x}
    # Forbidden: distinct a, b, c in set such that a+b, a+c, b+c are all in set.
    # If we add x, the new triples involving x are: (x, a, b) for a,b in selected
    #   where x+a, x+b, a+b are all in selected ∪ {x}
    # Also triples where x replaces one of the pair-sums:
    #   Actually, the condition is: three distinct members a,b,c such that a+b, a+c, b+c are all selected.
    #   When we add x, new forbidden triples can involve x as one of a,b,c.
    #   Case 1: x is one of {a,b,c}. WLOG a=x. Then need x+b, x+c, b+c all in set.
    #     b+c must be in selected (since x is new, b+c != x unless b+c=x, but b+c > x for b,c >= 1... not necessarily)
    #     x+b and x+c must be in selected ∪ {x}. Since x+b > x and x+c > x (for b,c >= 1), they must be in selected.
    #   So: for distinct b, c in selected: if (x+b) in s and (x+c) in s and (b+c) in s: forbidden.
    #   Case 2: x is one of the pair sums. WLOG a+b = x. Then need a+c, b+c in set.
    #     a and c distinct, b and c distinct, a and b distinct. c in selected, a,b in selected.
    #     a+c and b+c must be in selected (they can't be x since a+c > a and a+b=x, so c vs b... 
    #     a+c = x would mean a+c = a+b so c=b, contradiction. Similarly b+c != x.
    #     So a+c, b+c must be in selected.
    #     So: for distinct a, b in selected with a+b = x: for c in selected, c != a, c != b:
    #       if (a+c) in s and (b+c) in s: forbidden.
    
    sel = list(selected)
    n = len(sel)
    
    # Case 1: x is a member of the triple
    for i in range(n):
        b = sel[i]
        if (x + b) not in s:
            continue
        for j in range(i+1, n):
            c = sel[j]
            if (x + c) not in s:
                continue
            if (b + c) in s:
                return False
    
    # Case 2: x is a pair sum (a+b = x)
    for i in range(n):
        a = sel[i]
        if a >= x:
            continue
        b = x - a
        if b not in s or b <= a:
            continue
        # a and b are distinct and both in selected, a+b=x
        for j in range(n):
            c = sel[j]
            if c == a or c == b:
                continue
            if (a + c) in s and (b + c) in s:
                return False
    
    return True

def _greedy_construct(order, base_set):
    """Try adding elements in given order to base_set."""
    selected = set(base_set)
    for x in order:
        if x in selected:
            continue
        if _can_add(x, selected):
            selected.add(x)
    return selected

def _local_improve(selected, deadline):
    """Try to add more elements or swap to improve."""
    selected = set(selected)
    n = P['n']
    candidates = [x for x in range(1, n+1) if x not in selected]
    
    # Try adding each candidate
    for x in candidates:
        if _can_add(x, selected):
            selected.add(x)
    
    # Try swaps: remove one, add two
    sel_list = list(selected)
    for i in range(len(sel_list)):
        if time.time() > deadline:
            break
        a = sel_list[i]
        temp_selected = selected - {a}
        added = 0
        for x in candidates:
            if x in selected:
                continue
            if _can_add(x, temp_selected):
                temp_selected.add(x)
                added += 1
                if added >= 2:
                    selected = temp_selected
                    break
        if added >= 2:
            # Re-evaluate candidates
            candidates = [x for x in range(1, n+1) if x not in selected]
            sel_list = list(selected)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline with 10s safety margin
    
    n = P['n']
    base = list(range(n//2 + 1, n + 1))  # 65..128, size 64
    
    best_set = set(base)
    
    # Try multiple orderings for adding lower elements
    lower = list(range(1, n//2 + 1))  # 1..64
    
    # Ordering 1: descending (larger elements first, less likely to create forbidden triples)
    orders = [
        sorted(lower, reverse=True),
        sorted(lower),
        list(reversed(lower)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(20):
        shuffled = lower[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy_construct(order, base)
        if len(result) > len(best_set):
            best_set = result
    
    # Local improvement
    best_set = _local_improve(best_set, deadline)
    
    # More random restarts with different bases
    for trial in range(50):
        if time.time() > deadline:
            break
        # Try with a slightly different base
        base_copy = list(base)
        # Remove a random element from base
        idx = random.Random(trial).randint(0, len(base_copy)-1)
        removed = base_copy.pop(idx)
        
        order = lower[:]
        random.Random(trial+100).shuffle(order)
        
        result = _greedy_construct(order, base_copy)
        # Try to add back the removed element
        if _can_add(removed, result):
            result.add(removed)
        
        if len(result) > len(best_set):
            best_set = result
    
    # Final local improvement
    best_set = _local_improve(best_set, deadline)
    
    return list(best_set)
# EVOLVE-BLOCK-END
