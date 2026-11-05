import time
from fractions import Fraction
import numpy as np

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Precompute all 3-term AP constraints.
    2. Use a bitmask-based representation for fast validity checks.
    3. Initial greedy construction (smallest first).
    4. Local search:
       - Try adding missing elements.
       - Try swapping elements (remove one, add another with higher reciprocal value).
       - Try 2-for-1 and 1-for-2 moves (remove k, add m) if beneficial.
    5. Deadline guard to ensure completion within time limit.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # 5 second safety margin before 60s kill
    
    # Precompute all 3-term APs: (a, b, c) with a < b < c and a + c = 2b
    # For each element x, precompute the list of pairs that would form an AP with x.
    # Alternatively, for fast checking, precompute for each x:
    #   - mid_pairs[x]: list of (a, c) such that a + c = 2x, a < x < c
    #   - first_pairs[x]: list of (b, c) such that x + c = 2b, x < b < c
    #   - last_pairs[x]: list of (a, b) such that a + x = 2b, a < b < x
    
    mid_pairs = [[] for _ in range(n + 1)]
    first_pairs = [[] for _ in range(n + 1)]
    last_pairs = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x as middle
        for a in range(1, x):
            c = 2 * x - a
            if c <= n and c > x:
                mid_pairs[x].append((a, c))
        
        # x as first
        for b in range(x + 1, n + 1):
            c = 2 * b - x
            if c <= n and c > b:
                first_pairs[x].append((b, c))
        
        # x as last
        for b in range(1, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                last_pairs[x].append((a, b))
    
    def is_valid_addition(bitmask, x):
        """Check if x can be added to set represented by bitmask."""
        if (bitmask >> (x - 1)) & 1:
            return False
        
        # Check mid conflicts
        for a, c in mid_pairs[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
                return False
        
        # Check first conflicts
        for b, c in first_pairs[x]:
            if ((bitmask >> (b - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
                return False
        
        # Check last conflicts
        for a, b in last_pairs[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (b - 1)) & 1):
                return False
        
        return True
    
    def get_elements(bitmask):
        """Get list of elements from bitmask."""
        return [x for x in range(1, n + 1) if (bitmask >> (x - 1)) & 1]
    
    def reciprocal_sum(bitmask):
        """Compute exact reciprocal sum from bitmask."""
        total = Fraction(0, 1)
        for x in get_elements(bitmask):
            total += Fraction(1, x)
        return total
    
    def reciprocal_sum_float(bitmask):
        """Compute float reciprocal sum for fast comparison."""
        total = 0.0
        for x in get_elements(bitmask):
            total += 1.0 / x
        return total
    
    # Step 1: Greedy construction (add small numbers first to maximize reciprocal sum)
    bitmask = 0
    for x in range(1, n + 1):
        if is_valid_addition(bitmask, x):
            bitmask |= (1 << (x - 1))
    
    current_sum = reciprocal_sum_float(bitmask)
    best_bitmask = bitmask
    best_sum = current_sum
    
    # Precompute reciprocal values for fast comparison
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Step 2: Local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 1. Try adding missing numbers
        for x in range(1, n + 1):
            if time.time() > deadline - 5.0:
                break
            if (bitmask >> (x - 1)) & 1:
                continue
            if is_valid_addition(bitmask, x):
                new_bitmask = bitmask | (1 << (x - 1))
                new_sum = current_sum + recip[x]
                if new_sum > best_sum + 1e-15:
                    best_sum = new_sum
                    best_bitmask = new_bitmask
                    bitmask = new_bitmask
                    current_sum = new_sum
                    improved = True
                    break  # Restart to re-evaluate
        
        # 2. Try swaps: remove y, add x where x < y (so 1/x > 1/y)
        if not improved:
            elements = get_elements(bitmask)
            for y in elements:
                if time.time() > deadline - 3.0:
                    break
                # Temporarily remove y
                temp_bitmask = bitmask & ~(1 << (y - 1))
                
                # Try to add x where x < y
                for x in range(1, y):
                    if (temp_bitmask >> (x - 1)) & 1:
                        continue
                    if is_valid_addition(temp_bitmask, x):
                        new_bitmask = temp_bitmask | (1 << (x - 1))
                        new_sum = current_sum - recip[y] + recip[x]
                        if new_sum > best_sum + 1e-15:
                            best_sum = new_sum
                            best_bitmask = new_bitmask
                            bitmask = new_bitmask
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
        
        # 3. Try 2-for-1 moves: remove y, add x1 and x2 where recip[x1] + recip[x2] > recip[y]
        if not improved and time.time() < deadline - 10.0:
            elements = get_elements(bitmask)
            for y in elements:
                if time.time() > deadline - 10.0:
                    break
                temp_bitmask = bitmask & ~(1 << (y - 1))
                
                # Try adding pairs x1, x2
                for x1 in range(1, n + 1):
                    if (temp_bitmask >> (x1 - 1)) & 1:
                        continue
                    if not is_valid_addition(temp_bitmask, x1):
                        continue
                    temp_bitmask2 = temp_bitmask | (1 << (x1 - 1))
                    
                    for x2 in range(x1 + 1, n + 1):
                        if (temp_bitmask2 >> (x2 - 1)) & 1:
                            continue
                        if not is_valid_addition(temp_bitmask2, x2):
                            continue
                        new_sum = current_sum - recip[y] + recip[x1] + recip[x2]
                        if new_sum > best_sum + 1e-15:
                            new_bitmask = temp_bitmask2 | (1 << (x2 - 1))
                            best_sum = new_sum
                            best_bitmask = new_bitmask
                            bitmask = new_bitmask
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # 4. Try 1-for-2 moves: remove y1 and y2, add x where recip[x] > recip[y1] + recip[y2]
        if not improved and time.time() < deadline - 10.0:
            elements = get_elements(bitmask)
            for i in range(len(elements)):
                if time.time() > deadline - 10.0:
                    break
                y1 = elements[i]
                for j in range(i + 1, len(elements)):
                    y2 = elements[j]
                    temp_bitmask = bitmask & ~(1 << (y1 - 1)) & ~(1 << (y2 - 1))
                    
                    # Try adding x
                    for x in range(1, n + 1):
                        if (temp_bitmask >> (x - 1)) & 1:
                            continue
                        if not is_valid_addition(temp_bitmask, x):
                            continue
                        new_sum = current_sum - recip[y1] - recip[y2] + recip[x]
                        if new_sum > best_sum + 1e-15:
                            new_bitmask = temp_bitmask | (1 << (x - 1))
                            best_sum = new_sum
                            best_bitmask = new_bitmask
                            bitmask = new_bitmask
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Return the best set found
    return get_elements(best_bitmask)

# EVOLVE-BLOCK-START
# The code above is the main logic.
# Key improvements:
# 1. Bitmask representation for fast validity checks.
# 2. Precomputed conflict pairs for O(1) lookups.
# 3. Multiple local search moves: add, swap, 2-for-1, 1-for-2.
# 4. Deadline guard to ensure completion.
# EVOLVE-BLOCK-END
