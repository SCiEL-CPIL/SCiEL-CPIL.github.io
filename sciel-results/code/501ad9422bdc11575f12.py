import time
import random

def construct():
    deadline = time.time() + 85
    n = 512
    max_size = 48

    def _subset_sum_exists(lst, target, size):
        """Check if target can be formed by exactly 'size' elements from lst."""
        if size == 0:
            return target == 0
        if size > len(lst):
            return False
        if target < 0:
            return False
        if size == len(lst):
            return sum(lst) == target
        sorted_lst = sorted(lst)
        min_sum = sum(sorted_lst[:size])
        max_sum = sum(sorted_lst[-size:])
        if target < min_sum or target > max_sum:
            return False
        # DP with early termination
        dp = [set() for _ in range(size + 1)]
        dp[0] = {0}
        for val in lst:
            for k in range(size, 0, -1):
                if dp[k-1]:
                    new_sums = set()
                    for s in dp[k-1]:
                        ns = s + val
                        if ns <= target:
                            new_sums.add(ns)
                    dp[k].update(new_sums)
                    if target in dp[k]:
                        return True
        return target in dp[size]

    def is_valid_addition(S_list, x):
        """Check if x can be added to S_list (list of ints) without violating constraints."""
        # Check 1: x is not mean of any subset of S_list of size >= 2
        for k in range(2, len(S_list) + 1):
            if _subset_sum_exists(S_list, k * x, k):
                return False
        # Check 2: For each a in S_list, a is not mean of any subset of (S_list\{a})∪{x} of size >= 2
        for i, a in enumerate(S_list):
            others = S_list[:i] + S_list[i+1:]
            for k in range(2, len(S_list) + 1):
                # k elements from others ∪ {x} summing to k*a
                # If x included: k-1 from others summing to k*a - x
                if k - 1 >= 1:
                    if _subset_sum_exists(others, k * a - x, k - 1):
                        return False
        return True

    # Initial greedy construction
    random.seed(123)
    S = []
    for x in range(1, n + 1):
        if len(S) >= max_size:
            break
        if is_valid_addition(S, x):
            S.append(x)
    
    best_S = list(S)
    best_size = len(S)
    
    # Local search with random restarts and moves
    current_S = list(S)
    
    while time.time() < deadline:
        # Try random add
        improved = False
        for _ in range(200):
            x = random.randint(1, n)
            if x in current_S:
                continue
            if is_valid_addition(current_S, x):
                current_S.append(x)
                improved = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = list(current_S)
                break
        if improved:
            continue
        
        # Try swap
        if current_S:
            swap_done = False
            for _ in range(100):
                idx = random.randint(0, len(current_S) - 1)
                a = current_S[idx]
                x = random.randint(1, n)
                if x in current_S:
                    continue
                # Remove a, try add x
                temp_S = current_S[:idx] + current_S[idx+1:]
                if is_valid_addition(temp_S, x):
                    current_S = temp_S + [x]
                    swap_done = True
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_S = list(current_S)
                    break
            if swap_done:
                continue
        
        # Try remove one, add two
        if len(current_S) >= 3:
            idx = random.randint(0, len(current_S) - 1)
            a = current_S[idx]
            temp_S = current_S[:idx] + current_S[idx+1:]
            added = 0
            new_elements = []
            for _ in range(300):
                x = random.randint(1, n)
                if x in temp_S or x in new_elements:
                    continue
                if is_valid_addition(temp_S + new_elements, x):
                    new_elements.append(x)
                    added += 1
                    if added >= 2:
                        break
            if added >= 2:
                current_S = temp_S + new_elements
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = list(current_S)
            elif added == 1 and len(current_S) < best_size:
                # Keep the improvement if it helps
                current_S = temp_S + new_elements
    
    return best_S
