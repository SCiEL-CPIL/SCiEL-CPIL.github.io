import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n_max = 512
    target_size = 48

    def check_full_validity(S_list, x):
        """
        Check if adding x to set S_list (where S_list is already valid) keeps the set valid.
        Returns True if S_list U {x} is valid.
        """
        if x in S_list:
            return False
        
        n = len(S_list)
        
        # Check 1: x is not the mean of any subset of S_list of size >= 2
        # sum(T) = k * x for some T subset of S_list, |T| = k >= 2
        if n >= 2:
            max_sum = sum(S_list)
            max_k = min(n, max_sum // x)
            if max_k >= 2:
                # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
                dp = [0] * (n + 1)
                dp[0] = 1
                for val in S_list:
                    for k in range(n - 1, -1, -1):
                        if dp[k]:
                            dp[k+1] |= (dp[k] << val)
                for k in range(2, max_k + 1):
                    target = k * x
                    if (dp[k] >> target) & 1:
                        return False
        
        # Check 2: No element a in S_list is the mean of any subset of (S_list \ {a}) U {x} of size >= 2
        for a in S_list:
            others = [s for s in S_list if s != a]
            m = len(others)
            if m < 1:
                continue
            
            # Case B: subset contains x. M = {x} U T, T subset of others, |T| >= 1.
            # sum(M) = x + sum(T) = (|T| + 1) * a
            # sum(T) = (j + 1) * a - x where j = |T|, j >= 1
            
            if m >= 1:
                dp = [0] * (m + 1)
                dp[0] = 1
                for val in others:
                    for k in range(m - 1, -1, -1):
                        if dp[k]:
                            dp[k+1] |= (dp[k] << val)
                
                max_sum_o = sum(others)
                for j in range(1, m + 1):
                    target = (j + 1) * a - x
                    if target < 0:
                        continue
                    if target > max_sum_o:
                        break
                    if (dp[j] >> target) & 1:
                        return False
        
        return True

    def greedy_conflict_aware(order, S_set):
        """
        Greedy independent set with conflict-aware ordering.
        """
        S = []
        S_set = set(S_set) if S_set else set()
        
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if check_full_validity(S, x):
                S.append(x)
                S_set.add(x)
        
        return S, S_set

    def conflict_degree(x):
        """Count the number of pairs (a, b) with a < b, a + b = 2x, 1 <= a, b <= n_max."""
        a_min = max(1, 2 * x - n_max)
        a_max = x - 1
        if a_min > a_max:
            return 0
        return a_max - a_min + 1

    # Strategy 1: Greedy with conflict-aware ordering (lowest degree first)
    best_set = []
    best_size = 0
    
    ordered_by_degree = sorted(range(1, n_max + 1), key=lambda x: conflict_degree(x))
    S, S_set = greedy_conflict_aware(ordered_by_degree, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Strategy 2: Descending degree
    ordered_by_degree_desc = sorted(range(1, n_max + 1), key=lambda x: -conflict_degree(x))
    S, S_set = greedy_conflict_aware(ordered_by_degree_desc, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Strategy 3: Random orders
    rng = random.Random(42)
    for i in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        S, S_set = greedy_conflict_aware(order, set())
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Local search: try to add or swap elements
    if best_size < target_size and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try to add any available element
            for x in available:
                if len(S) >= target_size:
                    break
                if check_full_validity(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= target_size:
                break
            
            # If no add, try swaps
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = set(S_temp)
                    
                    for x in available:
                        if x in S_temp_set:
                            continue
                        if check_full_validity(S_temp, x):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(old)
                            improved = True
                            break
                    if improved:
                        break
    
    return best_set
