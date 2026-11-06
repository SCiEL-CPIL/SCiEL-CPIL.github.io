import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    best_set = []
    
    # Helper to check if a set is valid
    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        if n == 0:
            return True
        S_set = set(S_list)
        for a in S_list:
            T = [x for x in S_list if x != a]
            # Bitset DP for subset sums of T
            # dp[k] is a bitset of possible sums using k elements
            # Max sum is 48 * 512 = 24576
            # We use Python integers as bitsets
            max_sum = 25000
            dp = [0] * (len(T) + 1)
            dp[0] = 1 # bit 0 is set
            
            for x in T:
                # Update dp in reverse to avoid using x multiple times
                # We only care about sums up to max_sum
                for k in range(len(T) - 1, -1, -1):
                    if dp[k]:
                        # dp[k+1] |= dp[k] << x
                        # Mask to max_sum to keep integers small
                        dp[k+1] |= (dp[k] << x)
                        
            # Check if k*a is in dp[k] for k >= 2
            for k in range(2, len(T) + 1):
                target = k * a
                if target > max_sum:
                    continue
                if dp[k] & (1 << target):
                    return False
        return True

    # Greedy construction
    def greedy_construct():
        S = set()
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_S = S | {x}
            if is_valid_set(new_S):
                S = new_S
        return S

    # Local search to improve
    def local_search(S, max_iters=1000):
        S = set(S)
        current_size = len(S)
        all_vals = set(range(1, 513))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add an element
            added = False
            candidates_to_add = [x for x in all_vals if x not in S]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add[:20]: # Limit checks
                new_S = S | {x}
                if is_valid_set(new_S):
                    S = new_S
                    current_size += 1
                    added = True
                    break
            
            if added:
                continue
                
            # Try a swap
            if S:
                a = random.choice(list(S))
                candidates_to_swap = [x for x in all_vals if x not in S]
                random.shuffle(candidates_to_swap)
                for x in candidates_to_swap[:20]:
                    new_S = S - {a} | {x}
                    if is_valid_set(new_S):
                        S = new_S
                        break

        return S

    # Main loop
    best_size = 0
    while time.time() < deadline:
        # Greedy start
        S = greedy_construct()
        
        # Local search
        S = local_search(S, max_iters=200)
        
        # Verify and update best
        if is_valid_set(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
            if best_size == 48:
                break

    return list(best_set)
