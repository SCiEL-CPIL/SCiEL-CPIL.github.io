import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin for 60s limit
    
    n = 192
    max_k = 48
    candidates = list(range(n, 0, -1))
    
    best_set = []
    best_size = 0
    
    # Heuristic: Try to find a large set using backtracking
    # We prioritize larger numbers as they are less likely to be divisors of sums of smaller numbers
    
    def check_validity(x, S):
        """
        Check if adding x to set S results in a valid set.
        S is a list of integers.
        Conditions:
        1. x does not divide sum of any 3 distinct elements in S.
        2. For each a in S, a does not divide x + b + c for any distinct b, c in S \ {a}.
        """
        S_list = list(S)
        
        # Check 1: x divides sum of 3 distinct from S
        if len(S_list) >= 3:
            for comb in combinations(S_list, 3):
                if sum(comb) % x == 0:
                    return False
        
        # Check 2: For each a in S, a divides x + b + c
        # We need to check pairs (b, c) from S excluding a
        for i, a in enumerate(S_list):
            # Create list of S excluding a
            others = [v for j, v in enumerate(S_list) if j != i]
            if len(others) < 2:
                continue
            for b_idx in range(len(others)):
                b = others[b_idx]
                for c_idx in range(b_idx + 1, len(others)):
                    c = others[c_idx]
                    if (x + b + c) % a == 0:
                        return False
        return True

    def backtrack(idx, current_set, size):
        nonlocal best_set, best_size, timeout, start_time
        
        # Time check
        if time.time() - start_time > timeout:
            return
        
        # Update best
        if size > best_size:
            best_size = size
            best_set = list(current_set)
            
        # Pruning: If we can't beat best_size, stop
        # Max possible size is size + (number of remaining candidates)
        remaining = n - idx
        if size + remaining <= best_size:
            return
            
        # If we reached max_k, we can stop exploring further additions
        # But we might still find a different set of same size? No, we want max size.
        # If size == max_k, we found a perfect solution.
        if size == max_k:
            return

        # Try to include candidates[idx]
        x = candidates[idx]
        if check_validity(x, current_set):
            current_set.append(x)
            backtrack(idx + 1, current_set, size + 1)
            current_set.pop()
            
        # Try to exclude candidates[idx]
        backtrack(idx + 1, current_set, size)

    # Start backtracking
    backtrack(0, [], 0)
    
    # If backtracking didn't find a good set quickly, or to ensure we have a valid result
    # We can also try a simple greedy if backtracking times out very early
    if best_size == 0:
        greedy_set = []
        for x in candidates:
            if check_validity(x, greedy_set):
                greedy_set.append(x)
        if len(greedy_set) > best_size:
            best_set = greedy_set
            best_size = len(greedy_set)

    return best_set
