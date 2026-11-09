"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute weights as integers: weight[i] = LCM(2..48) / denominator[i]
    # LCM of 2..48 is a large number, but we can use a common denominator approach
    # Actually, let's use a smarter approach: find a common denominator that's manageable
    # LCM(2,3,...,48) = product of prime powers: 2^5 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # That's huge. Instead, let's use the fact that we only need to compare sums.
    
    # Better approach: use floating point for speed, with careful tie-breaking
    # Or use a smaller common denominator based on the actual denominators we use
    
    # For now, let's use integer arithmetic with a common denominator
    # The LCM of 2..48 can be computed, but it's very large
    # Let's use a different approach: keep track of sums as (numerator, denominator) pairs
    # and compare using cross-multiplication
    
    # Actually, for speed, let's use floating point with a very small epsilon for comparison
    # and use integer arithmetic only for final verification
    
    weights = [1.0 / d for d in denominators]
    
    # Greedy initialization: sort by weight descending, include if sum stays < 1
    order = sorted(range(n), key=lambda i: -weights[i])
    selected = set()
    total = 0.0
    for i in order:
        if total + weights[i] < 1.0 - 1e-15:
            selected.add(i)
            total += weights[i]
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1.0 - total
    
    # Local search with multiple move types
    def get_deficit(sel, tot):
        return 1.0 - tot
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_deficit = best_deficit
        
        # Try adding each unselected fraction
        for i in range(n):
            if i not in best_selected:
                new_total = best_total + weights[i]
                if new_total < 1.0 - 1e-15:
                    new_deficit = 1.0 - new_total
                    if new_deficit < current_deficit:
                        best_selected.add(i)
                        best_total = new_total
                        best_deficit = new_deficit
                        current_deficit = new_deficit
                        improved = True
        
        # Try 1-for-1 swaps
        for i in range(n):
            if i in best_selected:
                for j in range(n):
                    if j not in best_selected:
                        new_total = best_total - weights[i] + weights[j]
                        if new_total < 1.0 - 1e-15:
                            new_deficit = 1.0 - new_total
                            if new_deficit < current_deficit:
                                best_selected.remove(i)
                                best_selected.add(j)
                                best_total = new_total
                                best_deficit = new_deficit
                                current_deficit = new_deficit
                                improved = True
        
        # Try 1-for-2 swaps: remove one, add two
        for i in range(n):
            if i in best_selected:
                for j in range(n):
                    if j not in best_selected:
                        for k in range(n):
                            if k not in best_selected and k != j:
                                new_total = best_total - weights[i] + weights[j] + weights[k]
                                if new_total < 1.0 - 1e-15:
                                    new_deficit = 1.0 - new_total
                                    if new_deficit < current_deficit:
                                        best_selected.remove(i)
                                        best_selected.add(j)
                                        best_selected.add(k)
                                        best_total = new_total
                                        best_deficit = new_deficit
                                        current_deficit = new_deficit
                                        improved = True
                                        break
                if improved:
                    break
        
        # Try 2-for-1 swaps: remove two, add one
        for i in range(n):
            if i in best_selected:
                for j in range(n):
                    if j in best_selected and j != i:
                        for k in range(n):
                            if k not in best_selected:
                                new_total = best_total - weights[i] - weights[j] + weights[k]
                                if new_total < 1.0 - 1e-15:
                                    new_deficit = 1.0 - new_total
                                    if new_deficit < current_deficit:
                                        best_selected.remove(i)
                                        best_selected.remove(j)
                                        best_selected.add(k)
                                        best_total = new_total
                                        best_deficit = new_deficit
                                        current_deficit = new_deficit
                                        improved = True
                                        break
                if improved:
                    break
        
        # Try 2-for-2 swaps
        for i in range(n):
            if i in best_selected:
                for j in range(n):
                    if j in best_selected and j != i:
                        for k in range(n):
                            if k not in best_selected:
                                for l in range(n):
                                    if l not in best_selected and l != k:
                                        new_total = best_total - weights[i] - weights[j] + weights[k] + weights[l]
                                        if new_total < 1.0 - 1e-15:
                                            new_deficit = 1.0 - new_total
                                            if new_deficit < current_deficit:
                                                best_selected.remove(i)
                                                best_selected.remove(j)
                                                best_selected.add(k)
                                                best_selected.add(l)
                                                best_total = new_total
                                                best_deficit = new_deficit
                                                current_deficit = new_deficit
                                                improved = True
                                                break
                                if improved:
                                    break
                        if improved:
                            break
                if improved:
                    break
        
        # Random restarts if stuck
        if not improved and time.time() < deadline - 10:
            for _ in range(100):
                if time.time() >= deadline:
                    break
                # Random subset
                rand_selected = set()
                rand_total = 0.0
                for i in range(n):
                    if random.random() < 0.3:
                        if rand_total + weights[i] < 1.0 - 1e-15:
                            rand_selected.add(i)
                            rand_total += weights[i]
                if rand_selected and (1.0 - rand_total) < best_deficit:
                    best_selected = rand_selected
                    best_total = rand_total
                    best_deficit = 1.0 - rand_total
                    improved = True
                    break
    
    # Verify with exact arithmetic
    from fractions import Fraction
    exact_total = Fraction(0)
    for i in best_selected:
        exact_total += Fraction(1, denominators[i])
    
    if exact_total >= 1:
        # Need to fix: remove items until sum < 1
        # This shouldn't happen with our epsilon, but just in case
        for i in sorted(best_selected, key=lambda x: -weights[x]):
            exact_total -= Fraction(1, denominators[i])
            best_selected.remove(i)
            if exact_total < 1:
                break
    
    return sorted(list(best_selected))
# EVOLVE-BLOCK-END
