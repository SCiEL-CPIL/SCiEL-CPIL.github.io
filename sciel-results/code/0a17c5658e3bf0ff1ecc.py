import time
from itertools import combinations

def _can_add(selected, x):
    """
    Check if x can be added to selected while maintaining the constraint.
    Constraint: No element a in (selected + [x]) divides the sum of any 3 distinct OTHER elements.
    """
    if not selected:
        return True
    
    n = len(selected)
    
    # 1. Check if x divides the sum of any 3 distinct elements in selected
    # Sum of 3 max elements is 3 * max(selected).
    # If x > 3 * max(selected), it's safe.
    if n >= 3:
        if x <= 3 * max(selected):
            # Iterate over triples
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if (selected[i] + selected[j] + selected[k]) % x == 0:
                            return False
        # else: x is larger than any possible sum of 3 elements, so safe
    
    # 2. Check if any a in selected divides a sum involving x
    # Sum must be x + s_i + s_j where s_i, s_j are distinct elements from selected.
    # So for each a in selected, we check if (x + s_i + s_j) % a == 0 for any distinct s_i, s_j in selected.
    
    for a in selected:
        if n < 2:
            continue
            
        target = (-x) % a
        
        # We need distinct u, v from selected.
        # Let's collect residues of all elements in selected mod a.
        # But we need to ensure u and v are distinct elements.
        
        # Iterate over pairs of indices in selected.
        # n is small (<50), so O(n^2) is fast.
        for i in range(n):
            for j in range(i+1, n):
                if (x + selected[i] + selected[j]) % a == 0:
                    return False
    return True

def _greedy(order):
    """Greedy construction following a given order of candidates."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best = []
    
    # Strategy 1: Descending order (original)
    best = _greedy(range(192, 0, -1))
    
    if time.time() > deadline:
        return best
        
    # Strategy 2: Try various heuristic orderings
    import random
    
    # Strategy 3: Random restarts with different seeds
    # Mix of large numbers first with some randomness
    for seed in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        # Keep top 50 mostly in order, shuffle the rest
        candidates = list(range(192, 0, -1))
        top = candidates[:50]
        rest = candidates[50:]
        rng.shuffle(rest)
        order = top + rest
        result = _greedy(order)
        if len(result) > len(best):
            best = result
            
    # Strategy 4: Try ascending then descending hybrid / weighted
    if time.time() < deadline:
        for seed in range(30):
            if time.time() > deadline:
                break
            rng = random.Random(1000 + seed)
            candidates = list(range(192, 0, -1))
            # Weight toward larger numbers
            weighted = []
            for c in candidates:
                weight = c / 192.0
                for _ in range(int(weight * 3) + 1):
                    weighted.append(c)
            # Remove duplicates preserving order
            seen = set()
            unique_weighted = []
            for w in weighted:
                if w not in seen:
                    seen.add(w)
                    unique_weighted.append(w)
            rng.shuffle(unique_weighted)
            result = _greedy(unique_weighted)
            if len(result) > len(best):
                best = result
                
    # Strategy 5: Try to add more elements by trying different insertion points
    if time.time() < deadline:
        # For the best solution, try adding each excluded element
        best_set = set(best)
        for x in range(192, 0, -1):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            if _can_add(best, x):
                best.append(x)
                best_set.add(x)
                
    return best
