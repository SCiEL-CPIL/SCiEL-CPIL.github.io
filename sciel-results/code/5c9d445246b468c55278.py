"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2*max(U) + 1)
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Compute sum and difference sets
    # For efficiency, we can use numpy if the range is manageable, 
    # but for small U, set operations are fine.
    # Let's stick to sets for simplicity and correctness first, 
    # as N is likely small (<= 100-200).
    
    sum_set = set()
    diff_set = set()
    
    # Optimization: Use nested loops
    # This is O(N^2)
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    # The exponent lower bound is 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    # Note: The task says "lower bound ... on C6". 
    # The formula is: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses a greedy/local search approach to maximize the score.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # Initial set: Small sets often have good ratios because |U-U| is large relative to |U+U|
    # For {0, 1, ..., n}, |U+U| = 2n+1, |U-U| = 2n+1. Ratio 1. Score = 1.
    # For sparse sets, |U+U| can be larger than |U-U|? No, |U-U| >= |U+U| for sets containing 0?
    # Actually, if U is symmetric, |U-U| = |U+U|. But U is non-negative.
    # Let U = {0, a, b, ...}
    # Sums: a, b, a+b, 2a, 2b...
    # Diffs: a, b, a-b, b-a, 2a, 2b...
    # Generally |U-U| >= |U+U| because differences are symmetric (d and -d), while sums are non-negative.
    # So the ratio |U-U|/|U+U| is usually > 1.
    # We want to maximize this ratio while keeping max(U) small.
    
    # Let's try a few initial candidates
    candidates = [
        [0, 1, 3, 9, 27, 81], # Geometric
        [0, 1, 3, 9],
        [0, 1, 3, 10, 30],
        [0, 1, 2, 4, 8, 16, 32], # Powers of 2
    ]
    
    best_U = candidates[0]
    best_score = -1
    
    for c in candidates:
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_U = list(c)
    
    current_U = list(best_U)
    current_score = best_score
    
    # Local Search / Greedy Improvement
    # Strategies:
    # 1. Add a new element
    # 2. Remove an element (except 0)
    # 3. Replace an element
    
    while time.time() < deadline:
        improved = False
        current_max = max(current_U) if current_U else 0
        
        # Try adding a new element
        # Candidate new elements: small integers, or large gaps
        new_elements_to_try = set()
        
        # Small range around current max
        for x in range(1, min(50, int(current_max * 1.5) + 10)):
            if x not in current_U:
                new_elements_to_try.add(x)
                
        # Some specific geometric progressions
        if current_max > 0:
            for multiplier in [2, 3, 4, 5]:
                candidate = current_max * multiplier
                if candidate < 10000: # Keep size reasonable
                    new_elements_to_try.add(candidate)
        
        # Try removing an element
        elements_to_remove = [x for x in current_U if x != 0]
        
        best_move = None
        best_move_score = current_score
        best_move_type = None
        best_move_arg = None
        
        # Check additions
        if len(current_U) < 50: # Limit size
            for x in new_elements_to_try:
                new_U = sorted(list(set(current_U + [x])))
                s = compute_score(new_U)
                if s > best_move_score:
                    best_move_score = s
                    best_move = new_U
                    best_move_type = 'add'
                    best_move_arg = x
        
        # Check removals
        for x in elements_to_remove:
            new_U = [y for y in current_U if y != x]
            if len(new_U) < 2:
                continue
            s = compute_score(new_U)
            if s > best_move_score:
                best_move_score = s
                best_move = new_U
                best_move_type = 'remove'
                best_move_arg = x
        
        # Check replacements (swap x with y)
        # This is more expensive, only do if we have time
        if time.time() < deadline - 20:
            for x in elements_to_remove:
                # Try replacing x with a nearby number
                for delta in [-2, -1, 1, 2]:
                    y = x + delta
                    if y < 0 or y == 0 or y == x:
                        continue
                    if y in current_U:
                        continue
                    new_U = sorted(list(set([y] + [z for z in current_U if z != x])))
                    s = compute_score(new_U)
                    if s > best_move_score:
                        best_move_score = s
                        best_move = new_U
                        best_move_type = 'replace'
                        best_move_arg = (x, y)
        
        if best_move:
            current_U = best_move
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
            improved = True
        
        if not improved:
            # No simple improvement found, try a random restart or perturb
            # Let's try a few random sets
            for _ in range(5):
                if time.time() > deadline:
                    break
                size = random.randint(3, 10)
                max_val = random.randint(10, 100)
                random_set = set([0])
                while len(random_set) < size:
                    random_set.add(random.randint(1, max_val))
                random_U = sorted(list(random_set))
                s = compute_score(random_U)
                if s > best_score:
                    best_score = s
                    best_U = random_U
                    current_U = random_U
                    current_score = s
                    break
            
            # If still no improvement, break to avoid infinite loop
            # Or we could randomize current_U
            # Let's just break if we've been stuck for a while? 
            # For now, break if no improvement in local search.
            break
            
    return best_U

# EVOLVE-BLOCK-END
