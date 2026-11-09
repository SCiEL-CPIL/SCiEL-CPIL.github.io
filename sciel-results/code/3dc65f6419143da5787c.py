import time
import random
import numpy as np

def construct():
    N = 51975
    # Get all divisors > 1 of N
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    # Filter to odd moduli only (all divisors of odd N are odd, but be safe)
    divisors = [d for d in divisors if d > 1 and d % 2 == 1]
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Greedy initialization with a given order
    def greedy_init(order):
        cov = [0] * N
        classes = {}  # m -> r
        uncovered = N
        
        for m in order:
            if time.time() > deadline:
                break
            best_r = 0
            max_gain = -1
            for r in range(m):
                # Count how many currently-uncovered residues fall in class (m, r)
                gain = 0
                for x in range(r, N, m):
                    if cov[x] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_r = r
            
            # Apply best_r
            classes[m] = best_r
            for x in range(best_r, N, m):
                cov[x] += 1
                if cov[x] == 1:
                    uncovered -= 1
        
        return uncovered, classes, cov
    
    # Local search: try changing residue for one modulus
    def local_search(cov, classes, uncovered, max_iterations=5000):
        if not classes:
            return uncovered, classes
        
        moduli = list(classes.keys())
        improved = True
        iters = 0
        
        while improved and iters < max_iterations and time.time() < deadline:
            improved = False
            iters += 1
            
            # Shuffle moduli for randomization
            random.shuffle(moduli)
            
            for m in moduli:
                if time.time() > deadline:
                    return uncovered, classes
                
                old_r = classes[m]
                best_r = old_r
                best_uncovered = uncovered
                
                # Try all other residues for this modulus
                for r in range(m):
                    if r == old_r:
                        continue
                    
                    # Calculate new uncovered if we change (m, old_r) to (m, r)
                    # First, remove coverage from old class
                    # Then add coverage from new class
                    # Net change: residues covered only by old class become uncovered,
                    # residues not covered by new class but were uncovered become covered
                    
                    # Efficient approach: count the difference
                    # Gain = (# newly covered) - (# newly uncovered)
                    # newly covered: x in new class where cov[x] == 1 (only covered by old)
                    # newly uncovered: x in old class where cov[x] == 1 (only covered by old) 
                    # Wait, that's not right.
                    
                    # Let me think more carefully:
                    # Current state: cov[x] >= 0 for all x
                    # Remove old class: for x in range(old_r, N, m): cov[x] -= 1
                    # Add new class: for x in range(r, N, m): cov[x] += 1
                    # Net effect on uncovered count:
                    # A residue x becomes uncovered if:
                    #   - x is in old class but not in new class, AND cov[x] was 1
                    # A residue x becomes covered if:
                    #   - x is in new class but not in old class, AND cov[x] was 0 (after removing old)
                    
                    # More precisely:
                    # After removing old class, cov'[x] = cov[x] - 1 if x in old else cov[x]
                    # After adding new class, cov''[x] = cov'[x] + 1 if x in new else cov'[x]
                    
                    # x becomes uncovered: cov''[x] == 0 and cov[x] > 0
                    #   This happens when: x in old, x not in new, cov[x] == 1
                    # x becomes covered: cov''[x] > 0 and cov[x] == 0
                    #   This happens when: x not in old, x in new, cov[x] == 0
                    
                    # But we also need to handle overlap: x in both old and new
                    # If x in both: cov''[x] = cov[x] - 1 + 1 = cov[x], no change
                    
                    # So:
                    # newly_uncovered = #{x : x in old, x not in new, cov[x] == 1}
                    # newly_covered = #{x : x not in old, x in new, cov[x] == 0}
                    # delta = newly_covered - newly_uncovered
                    # new_uncovered = uncovered + delta
                    
                    # Compute this efficiently
                    # x in old: x = old_r + k*m
                    # x in new: x = r + k*m
                    # x in both: x ≡ old_r (mod m) and x ≡ r (mod m) => old_r ≡ r (mod m), i.e., old_r == r
                    # Since we skip r == old_r, there's no overlap!
                    
                    # So:
                    # newly_uncovered = #{x in range(old_r, N, m) : cov[x] == 1}
                    # newly_covered = #{x in range(r, N, m) : cov[x] == 0}
                    
                    newly_uncovered = 0
                    for x in range(old_r, N, m):
                        if cov[x] == 1:
                            newly_uncovered += 1
                    
                    newly_covered = 0
                    for x in range(r, N, m):
                        if cov[x] == 0:
                            newly_covered += 1
                    
                    new_uncovered = uncovered + newly_uncovered - newly_covered
                    
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r
                
                # Apply best change if improvement
                if best_r != old_r:
                    # Remove old class
                    for x in range(old_r, N, m):
                        cov[x] -= 1
                        if cov[x] == 0:
                            uncovered += 1
                    # Add new class
                    for x in range(best_r, N, m):
                        cov[x] += 1
                        if cov[x] == 1:
                            uncovered -= 1
                    classes[m] = best_r
                    uncovered = best_uncovered
                    improved = True
        
        return uncovered, classes
    
    # Run multiple greedy initializations and local searches
    best_uncovered = N
    best_classes = {}
    
    # Deterministic orders
    orders = [
        sorted(divisors),
        sorted(divisors, reverse=True),
        sorted(divisors, key=lambda d: N // d),  # Smaller classes first
        sorted(divisors, key=lambda d: -N // d),  # Larger classes first
    ]
    
    # Add random orders
    for _ in range(15):
        random_order = divisors[:]
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        unc, classes, cov = greedy_init(order)
        
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_cov = cov[:]
        
        # Run local search on this solution
        if time.time() < deadline - 10:  # Leave some time
            unc_ls, classes_ls = local_search(cov, classes, unc)
            
            if unc_ls < best_uncovered:
                best_uncovered = unc_ls
                best_classes = classes_ls
    
    # Convert to list of [m, r] pairs
    result = [[m, r] for m, r in best_classes.items()]
    return result
