"""Solve closest strict underpacking of 1 by unit fractions from [2,3,...,48]."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    # Denominators: index i -> denominator i+2, so d ranges 2..48
    # We want to maximize sum of selected 1/d subject to sum < 1
    # Equivalently, minimize deficit 1 - sum > 0

    # LCM of 2..48
    def lcm(a, b):
        from math import gcd
        return a * b // gcd(a, b)
    
    L = 1
    for d in range(2, 49):
        L = lcm(L, d)
    
    # Convert each 1/d to numerator over L
    vals = [L // d for d in range(2, 49)]  # vals[i] = L // (i+2)
    target = L  # we want sum < target
    
    # Split indices into A (large values, small denominators) and B (small values, large denominators)
    # A: indices 0..26 (denoms 2..28)
    # B: indices 27..46 (denoms 29..48)
    # We enumerate subsets of B and greedily fill from A.
    
    best_sum = -1
    best_indices = []
    
    # Pre-sort A indices by value descending (which is index ascending 0..26)
    # vals[0] = 1/2, vals[1] = 1/3, etc.
    
    # Enumerate subsets of B (20 items)
    num_b = 20
    # We can iterate masks from 0 to 2^20 - 1
    
    # Optimization: We can pre-calculate sums for B subsets? No, just iterate.
    # But 2^20 is ~1M.
    
    # To speed up, we can use a recursive or iterative approach with pruning if we sorted B by value?
    # B values are vals[27]..vals[46].
    # They are small.
    
    # Let's just iterate masks.
    for mask in range(1 << num_b):
        if elapsed() > DEADLINE:
            break
            
        sum_b = 0
        selected_b = []
        
        # Calculate sum and indices for B
        # Unroll loop for speed? Or just iterate.
        for k in range(num_b):
            if mask & (1 << k):
                idx = 27 + k
                sum_b += vals[idx]
                selected_b.append(idx)
                
        if sum_b >= target:
            continue
            
        # Greedily fill from A
        current_sum = sum_b
        selected_a = []
        
        for i in range(27):
            if current_sum + vals[i] < target:
                current_sum += vals[i]
                selected_a.append(i)
            # Note: If we skip i, we don't necessarily stop, because later i might be smaller?
            # No, vals[i] is decreasing with i. So if vals[i] doesn't fit, vals[i+1] is smaller and MIGHT fit.
            # Wait, vals[i] = L/(i+2).
            # i=0 -> 1/2
            # i=1 -> 1/3
            # So vals[0] > vals[1] > ...
            # If vals[i] doesn't fit, vals[i+1] is smaller, so it *might* fit.
            # So we must continue checking all i.
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = selected_a + selected_b
            
    # Local search to polish
    # 1-exchange and 2-exchange
    selected = set(best_indices)
    current_sum = best_sum
    
    def try_add(i):
        nonlocal current_sum, best_sum, best_indices, selected
        if i in selected:
            return False
        new_sum = current_sum + vals[i]
        if new_sum < target and new_sum > best_sum:
            selected.add(i)
            current_sum = new_sum
            best_sum = new_sum
            best_indices = sorted(selected)
            return True
        return False
        
    def try_swap(i, j):
        nonlocal current_sum, best_sum, best_indices, selected
        if i not in selected or j in selected:
            return False
        new_sum = current_sum - vals[i] + vals[j]
        if new_sum < target and new_sum > best_sum:
            selected.discard(i)
            selected.add(j)
            current_sum = new_sum
            best_sum = new_sum
            best_indices = sorted(selected)
            return True
        return False

    # Run a few rounds of local search
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        for i in range(47):
            if try_add(i):
                improved = True
                break
        if elapsed() > DEADLINE: break
        
        for i in list(selected):
            for j in range(47):
                if try_swap(i, j):
                    improved = True
                    break
            if improved or elapsed() > DEADLINE: break
            
    # Final check
    final_sum = sum(vals[i] for i in best_indices)
    if final_sum >= target:
         # Fallback to a known valid solution if something went wrong
         # Greedy from scratch
         selected = set()
         s = 0
         for i in range(47):
             if s + vals[i] < target:
                 selected.add(i)
                 s += vals[i]
         best_indices = sorted(selected)
         best_sum = s

    return best_indices
# EVOLVE-BLOCK-END
