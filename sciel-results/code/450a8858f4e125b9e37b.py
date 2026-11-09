import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    max_elements = 48
    n = 192
    
    def is_safe_incremental(a, S_list, S_set_local):
        """Check if adding a to S_list keeps validity. Only checks new constraints involving a."""
        L = len(S_list)
        if L >= 3:
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % a] += 1
            
            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        
        for b in S_list:
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue
            
            target_rem = (-a) % b
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1
            
            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    return False
        
        return True
    
    def is_safe_set(S_list):
        """Check if S_list is valid: no element divides sum of 3 distinct others."""
        for idx, a in enumerate(S_list):
            others = [v for i, v in enumerate(S_list) if i != idx]
            L_len = len(others)
            if L_len < 3:
                continue
            
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % a] += 1
            
            for i in range(L_len):
                u = others[i]
                for j in range(i + 1, L_len):
                    v = others[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    def compute_conflict_degree(x, S_list):
        """Compute how many conflicts x would have with existing set S_list."""
        conflicts = 0
        L = len(S_list)
        
        # Count pairs (u,v) in S_list where u+v+w ≡ 0 (mod x) for some w in S_list
        if L >= 3:
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % x] += 1
            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % x
                    cnt = rem_count.get(target_rem, 0)
                    if u % x == target_rem:
                        cnt -= 1
                    if v % x == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        conflicts += cnt
        
        # Count for each b in S_list, pairs (u,v) in S_list\{b} where a+u+v ≡ 0 (mod b)
        for b in S_list:
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue
            target_rem = (-x) % b
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1
            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    conflicts += cnt
        
        return conflicts
    
    # Strategy: Degree-based greedy with multiple random restarts
    best_S = []
    best_size = 0
    
    def try_greedy_order(order, S_init=None):
        """Try greedy selection given an initial set and an ordering of candidates."""
        if S_init is not None:
            S = S_init[:]
            S_set = set(S_init)
        else:
            S = []
            S_set = set()
        
        for x in order:
            if len(S) >= max_elements:
                break
            if time.time() - start_time > deadline - 5:
                break
            if x in S_set:
                continue
            if is_safe_incremental(x, S, S_set):
                S.append(x)
                S_set.add(x)
        
        return S
    
    # Precompute a static conflict score for each number (how "conflicting" it is overall)
    # A number x is more conflicting if it divides many sums of 3 other numbers
    static_score = {}
    for x in range(1, n + 1):
        # Count how many triples (a,b,c) from [1,n]\{x} have a+b+c divisible by x
        # This is expensive, so use a heuristic: smaller numbers tend to be more constraining
        # because they divide more sums
        static_score[x] = x  # Smaller x = more likely to divide sums = more restrictive
        
    # Multiple restarts with different strategies
    strategies = []
    
    # Strategy 1: Largest to smallest (original)
    strategies.append(list(range(n, 0, -1)))
    
    # Strategy 2: Smallest to largest
    strategies.append(list(range(1, n + 1)))
    
    # Strategy 3: Random order
    for _ in range(5):
        rand_order = list(range(1, n + 1))
        random.shuffle(rand_order)
        strategies.append(rand_order)
    
    # Strategy 4: Degree-based - sort by "conflict potential" (smaller numbers first as they're more constraining)
    # Actually, we want to pick numbers that have FEWER conflicts first
    # Heuristic: larger numbers have fewer divisors, so they conflict less
    degree_order = sorted(range(1, n + 1), key=lambda x: -x)  # Largest first = least constraining
    strategies.append(degree_order)
    
    # Strategy 5: Mixed - alternate large and small
    mixed = []
    large = list(range(n, 0, -1))
    small = list(range(1, n + 1))
    for i in range(n):
        if i < len(large):
            mixed.append(large[i])
        if i < len(small):
            mixed.append(small[i])
    strategies.append(mixed)
    
    for strat_idx, order in enumerate(strategies):
        if time.time() - start_time > deadline - 10:
            break
        
        S_result = try_greedy_order(order)
        
        if len(S_result) > best_size:
            best_size = len(S_result)
            best_S = S_result[:]
        
        # Try to improve with local search
        if len(S_result) < max_elements and time.time() - start_time < deadline - 3:
            improved = True
            while improved and len(S_result) < max_elements and time.time() - start_time < deadline - 3:
                improved = False
                S_copy = S_result[:]
                S_set_copy = set(S_copy)
                
                for i in range(len(S_copy)):
                    if time.time() - start_time > deadline - 3:
                        break
                    S_temp = [v for j, v in enumerate(S_copy) if j != i]
                    S_set_temp = set(S_temp)
                    
                    for x1 in range(n, 0, -1):
                        if len(S_temp) >= max_elements:
                            break
                        if x1 in S_set_temp:
                            continue
                        if is_safe_incremental(x1, S_temp, S_set_temp):
                            S_temp2 = S_temp + [x1]
                            S_set_temp2 = S_set_temp | {x1}
                            
                            for x2 in range(n, 0, -1):
                                if len(S_temp2) >= max_elements:
                                    break
                                if x2 in S_set_temp2:
                                    continue
                                if is_safe_incremental(x2, S_temp2, S_set_temp2):
                                    S_temp3 = S_temp2 + [x2]
                                    if len(S_temp3) > len(S_copy):
                                        S_result = S_temp3
                                        S_set_copy = set(S_result)
                                        S_copy = S_result[:]
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
                
                if len(S_result) > best_size:
                    best_size = len(S_result)
                    best_S = S_result[:]
    
    # Final local search on best_S
    if len(best_S) < max_elements and time.time() - start_time < deadline - 1:
        improved = True
        while improved and len(best_S) < max_elements and time.time() - start_time < deadline - 1:
            improved = False
            S_copy = best_S[:]
            S_set_copy = set(S_copy)
            
            for i in range(len(S_copy)):
                if time.time() - start_time > deadline - 1:
                    break
                S_temp = [v for j, v in enumerate(S_copy) if j != i]
                S_set_temp = set(S_temp)
                
                for x1 in range(n, 0, -1):
                    if len(S_temp) >= max_elements:
                        break
                    if x1 in S_set_temp:
                        continue
                    if is_safe_incremental(x1, S_temp, S_set_temp):
                        S_temp2 = S_temp + [x1]
                        S_set_temp2 = S_set_temp | {x1}
                        
                        for x2 in range(n, 0, -1):
                            if len(S_temp2) >= max_elements:
                                break
                            if x2 in S_set_temp2:
                                continue
                            if is_safe_incremental(x2, S_temp2, S_set_temp2):
                                S_temp3 = S_temp2 + [x2]
                                if len(S_temp3) > len(S_copy):
                                    best_S = S_temp3
                                    S_copy = best_S[:]
                                    S_set_copy = set(S_copy)
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
    
    return best_S
