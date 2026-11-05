"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Use a boolean array for fast membership testing
    in_S = [False] * (N + 1)
    S_list = []
    S_set = set()
    
    def can_add(x):
        """Check if x can be added to the current set without creating a 3-AP."""
        if x < 1 or x > N:
            return False
        if in_S[x]:
            return True # Already in set, trivially "can add" (no change)
            
        # For each a in S, check if a, x, c or a, b, x or x, b, a forms AP
        # 1. x is middle: a, x, c => c = 2x - a
        # 2. x is end: a, b, x or x, b, a => b = (a+x)/2
        # Note: We only need to check a in S.
        
        for a in S_list:
            # Case 1: a, x, c. c = 2x - a
            c = 2 * x - a
            if 1 <= c <= N and c != a and in_S[c]:
                return False
            
            # Case 2: a, b, x (x is largest) or x, b, a (x is smallest)
            # b = (a + x) / 2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if 1 <= b <= N and b != a and b != x and in_S[b]:
                    return False
        return True

    def add(x):
        in_S[x] = True
        S_list.append(x)
        S_set.add(x)

    def remove(x):
        in_S[x] = False
        S_list.remove(x)
        S_set.remove(x)

    def get_score():
        return sum(Fraction(1, x) for x in S_set)

    # Initial Greedy Construction
    # Try to add numbers from 1 to N in increasing order (maximizes weight)
    for x in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        if can_add(x):
            add(x)
    
    current_score = get_score()
    best_score = current_score
    best_set = set(S_set)
    
    # Local Search
    # We want to maximize sum(1/x). 
    # Strategy:
    # 1. Try to add any missing small number.
    # 2. Try to swap a large number x with a smaller number y if valid and 1/y > 1/x (always true if y<x).
    #    Actually, if we remove x and add y, score changes by 1/y - 1/x.
    #    If y < x, 1/y > 1/x, so score increases.
    #    So we want to remove large elements and replace with smaller ones if possible.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1. Try adding missing numbers (prioritize small ones)
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not in_S[x]:
                if can_add(x):
                    add(x)
                    current_score += Fraction(1, x)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(S_set)
        
        # 2. Try swaps: Remove a large element, try to add a smaller one
        # Sort S_list in descending order to target large elements first
        # But we need to be careful with modifying S_list while iterating.
        # We iterate over a copy of sorted keys.
        large_elements = sorted([x for x in S_set if x > 1], reverse=True)
        
        for x in large_elements:
            if time.time() > deadline - 2:
                break
            if not in_S[x]:
                continue
            
            # Temporarily remove x
            remove(x)
            old_score = current_score
            current_score -= Fraction(1, x)
            
            found_better = False
            # Try to add any y < x that is not in S
            # We prioritize small y because they give larger weight gain
            # But any y < x gives a gain.
            for y in range(1, x):
                if time.time() > deadline - 2:
                    break
                if in_S[y]:
                    continue
                if can_add(y):
                    add(y)
                    current_score += Fraction(1, y)
                    if current_score > old_score:
                        found_better = True
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(S_set)
                        break
                    else:
                        # Should not happen if y < x, but safe to revert
                        remove(y)
                        current_score -= Fraction(1, y)
            
            if not found_better:
                # Revert removal of x
                add(x)
                current_score += Fraction(1, x)
            # If found_better, x is gone, y is added. Loop continues.
            # Note: The list `large_elements` is stale, but `in_S` is updated.
            # We just skip if x is no longer in S.

        # 3. Random Perturbation to escape local optima
        # If stuck (no improvement in last few iterations? or just periodically)
        # Let's do a simple random restart of the local search phase by shuffling?
        # Or just random removal of 1-3 elements and refill.
        
        # Check if we are making progress. If not, perturb.
        # For simplicity, we will just rely on the swap/add logic which is quite powerful.
        # However, the "multiple randomized greedy orderings" inspiration suggests diversity.
        # Let's add a perturbation step if we haven't improved in a while.
        
        # To keep it simple and robust within time:
        # If no improvement in this full pass, try a random perturbation.
        if not improved:
             # Randomly remove 1-3 elements
             num_remove = random.randint(1, 3)
             elements_to_remove = random.sample(list(S_set), min(num_remove, len(S_set)))
             for x in elements_to_remove:
                 remove(x)
                 current_score -= Fraction(1, x)
             
             # Greedy refill
             for x in range(1, N + 1):
                 if not in_S[x]:
                     if can_add(x):
                         add(x)
                         current_score += Fraction(1, x)
                         if current_score > best_score:
                             best_score = current_score
                             best_set = set(S_set)
             
             # Continue loop to try further improvements

    # Return the best set found
    return list(best_set)
# EVOLVE-BLOCK-END
