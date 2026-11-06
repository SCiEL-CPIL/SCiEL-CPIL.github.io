from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2
    left = denominators[:mid]
    right = denominators[mid:]
    
    # Generate all subset sums for left half
    left_sums = [(Fraction(), frozenset())]
    for d in left:
        new_sums = []
        for s, idx in left_sums:
            new_s = s + Fraction(1, d)
            new_idx = idx | frozenset([left.index(d)]) # This is inefficient
        # Better: use bitmasks or lists of indices
    # Let's use a list of (sum, indices)
    # Actually, generating 2^23 sums with Fraction might be slow.
    # Let's try a heuristic search instead.
    
    # Heuristic: Local Search
    # Start with the greedy solution
    selected = []
    total = Fraction(0, 1)
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    
    best_selected = selected[:]
    best_total = total
    
    # Local search: try to improve
    # We want to increase the sum (reduce deficit) without exceeding 1.
    # We can try adding a new element, or swapping an element.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding an element not in selected
        for i in range(n):
            if i not in selected:
                new_total = total + Fraction(1, denominators[i])
                if new_total < 1:
                    # This is a valid improvement (sum is larger)
                    selected.append(i)
                    total = new_total
                    best_selected = selected[:]
                    best_total = total
                    improved = True
                    break # Restart search from new state
        if improved:
            continue
            
        # Try swapping: remove one, add two? Or remove one, add one?
        # Since we are maximizing sum, we want to replace a small fraction with a larger one?
        # No, we want to get closer to 1.
        # If we remove a fraction, sum decreases. Then we can add others.
        
        # Let's try a different strategy: 
        # Start with a set that sums to < 1.
        # Try to find a subset that sums to exactly 1 - epsilon.
        
        # Let's use a DP-like approach for the remaining capacity.
        # Capacity = 1 - total.
        # We want to fill this capacity as much as possible with the remaining denominators.
        
        remaining_denoms = [d for i, d in enumerate(denominators) if i not in selected]
        # Sort remaining by value descending
        remaining_denoms.sort(key=lambda d: -1/d)
        
        # Greedy fill the capacity
        cap = 1 - total
        if cap > 0:
            # Try to add fractions to fill cap
            # This is a knapsack problem
            # Since n is small, we can try all subsets of the top K remaining
            # But let's just try adding the largest possible fraction that fits
            for d in remaining_denoms:
                if Fraction(1, d) <= cap:
                    selected.append(denominators.index(d))
                    total += Fraction(1, d)
                    cap -= Fraction(1, d)
                    best_selected = selected[:]
                    best_total = total
                    improved = True
                    break
            
    return best_selected
