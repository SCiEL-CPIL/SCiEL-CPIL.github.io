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
    
    # Initial solution: greedy by smallest denominator first
    selected = set()
    current_sum = 0
    
    for i in range(47):
        if current_sum + vals[i] < target:
            selected.add(i)
            current_sum += vals[i]
    
    best_sum = current_sum
    best_indices = sorted(selected)
    
    # Also try the known good solution: [1, 2, 3, 4, 35, 42] -> denominators [3,4,5,6,37,44]
    known = [1, 2, 3, 4, 35, 42]
    known_sum = sum(vals[i] for i in known)
    if known_sum < target and known_sum > best_sum:
        best_sum = known_sum
        best_indices = known
        selected = set(known)
        current_sum = known_sum
    
    # Local search with 1-exchange and 2-exchange moves
    def try_add(i):
        nonlocal best_sum, best_indices, selected, current_sum
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
        """Remove i, add j"""
        nonlocal best_sum, best_indices, selected, current_sum
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
    
    # Iterative improvement
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        
        # Try adding any unselected item
        for i in range(47):
            if try_add(i):
                improved = True
                break
        
        if elapsed() > DEADLINE:
            break
        
        # Try swaps
        for i in list(selected):
            for j in range(47):
                if try_swap(i, j):
                    improved = True
                    break
            if improved or elapsed() > DEADLINE:
                break
        
        # Try 2-exchanges: remove two, add two
        if not improved and elapsed() < DEADLINE:
            sel_list = sorted(selected)
            for idx1 in range(len(sel_list)):
                if elapsed() > DEADLINE:
                    break
                for idx2 in range(idx1 + 1, len(sel_list)):
                    if elapsed() > DEADLINE:
                        break
                    i1, i2 = sel_list[idx1], sel_list[idx2]
                    for j1 in range(47):
                        if j1 in selected:
                            continue
                        for j2 in range(j1 + 1, 47):
                            if j2 in selected:
                                continue
                            new_sum = (current_sum 
                                      - vals[i1] - vals[i2]
                                      + vals[j1] + vals[j2])
                            if new_sum < target and new_sum > best_sum:
                                selected.discard(i1)
                                selected.discard(i2)
                                selected.add(j1)
                                selected.add(j2)
                                current_sum = new_sum
                                best_sum = new_sum
                                best_indices = sorted(selected)
                                improved = True
                                break
                        if improved or elapsed() > DEADLINE:
                            break
                    if improved or elapsed() > DEADLINE:
                        break
                if improved or elapsed() > DEADLINE:
                    break
    
    # Final verification and return
    # Ensure the result is valid
    final_sum = sum(vals[i] for i in best_indices)
    if final_sum >= target:
        # Fallback: use the known good solution
        best_indices = [1, 2, 3, 4, 35, 42]
    
    return best_indices
# EVOLVE-BLOCK-END
