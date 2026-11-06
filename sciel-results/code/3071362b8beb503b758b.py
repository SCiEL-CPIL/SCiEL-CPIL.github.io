import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # We want subset sum < 1, maximize sum
    # Use fractions to avoid floating point errors in comparison
    # But for speed, we can use LCM or just careful float with epsilon?
    # LCM of 2..48 is large but fits in Python int.
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    values = [L // d for d in denominators]
    
    # Greedy initialization
    selected = [False] * n
    current_sum = 0
    
    # Sort indices by value descending (denominator ascending)
    # Since denominators are 2..48, index 0 is 1/2, index 46 is 1/48
    # Values are already in descending order of fraction size.
    for i in range(n):
        if current_sum + values[i] < L:
            selected[i] = True
            current_sum += values[i]
    
    # Local Search: 1-swap and 1-add
    # We track best solution found
    best_sum = current_sum
    best_selected = selected[:]
    
    while True:
        if time.time() - start_time > timeout:
            break
        
        improved = False
        
        # Try Add moves
        for j in range(n):
            if not selected[j]:
                if current_sum + values[j] < L:
                    selected[j] = True
                    current_sum += values[j]
                    improved = True
                    break # Restart search from scratch after change
        
        if improved:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = selected[:]
            continue
        
        # Try Swap moves
        # To optimize, we can iterate selected i and unselected j
        # If swap improves sum, do it.
        swapped = False
        for i in range(n):
            if not selected[i]:
                continue
            for j in range(n):
                if selected[j]:
                    continue
                # Check if swapping i (out) and j (in) improves sum
                # new_sum = current_sum - values[i] + values[j]
                # Condition: new_sum > current_sum AND new_sum < L
                if values[j] > values[i]:
                    new_sum = current_sum - values[i] + values[j]
                    if new_sum < L:
                        # Perform swap
                        selected[i] = False
                        selected[j] = True
                        current_sum = new_sum
                        swapped = True
                        break
            if swapped:
                break
        
        if swapped:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = selected[:]
            continue
        
        # No add or swap improved the sum.
        # We might be in a local optimum.
        # For this specific problem (unit fractions), greedy + 1-swap is often optimal or very close.
        # If we want to be safer, we could try 2-swap, but it's O(N^4) or O(N^2 * N^2).
        # Given N=47, N^2 is ~2200. N^4 is ~23M. Might be tight in Python.
        # Let's stick to 1-swap for speed and reliability.
        
        break
        
    # Reconstruct indices
    result = []
    for i in range(n):
        if best_selected[i]:
            result.append(i)
    
    # Ensure non-empty
    if not result:
        result = [0]
        
    return result
