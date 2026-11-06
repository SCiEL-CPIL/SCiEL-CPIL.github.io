import random
import time

def construct():
    max_n = 512
    max_size = 48
    best_set = []
    
    # Fast validity check for a set S
    def is_valid(S):
        S_set = set(S)
        n = len(S)
        if n < 2:
            return True
        
        # Precompute subset sums DP for S
        # dp[k] is a bitmask of sums achievable by k elements
        # Max sum is 512 * 48 = 24576
        # We need to check for each x in S, if k*x is in dp[k] formed by S\{x}
        # This requires removing x from DP, which is hard.
        # Alternative: For each x, run a DP on S\{x}.
        # This is O(N^2 * DP_cost). N=50, DP_cost is small.
        
        for x in S:
            S_minus_x = [y for y in S if y != x]
            m = len(S_minus_x)
            if m < 2:
                continue
            
            # DP for S_minus_x
            # We only need to check up to size m
            # And sums up to m * max(S)
            max_sum = m * max(S_minus_x)
            # Use integer bitmasks
            dp = [0] * (m + 1)
            dp[0] = 1
            for y in S_minus_x:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << y)
            
            # Check if x is a mean
            for k in range(2, m + 1):
                target = k * x
                if (dp[k] >> target) & 1:
                    return False
        return True

    # Initialize with a good seed
    # Use the greedy logic from the original program to get a starting point
    def greedy_init():
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        candidates = list(range(1, max_n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= max_size:
                break
            # Check if x can be added
            # Reuse the logic from current program but simplified for init
            # Just use is_valid on S + [x]
            # This is O(N^2) per insertion, so O(N^3) total. N=48. 48^3 ~ 100k. Fast.
            if is_valid(S + [x]):
                S.append(x)
        return S

    best_set = greedy_init()
    best_len = len(best_set)
    
    # Local Search
    start_time = time.time()
    timeout = 55 # Leave 5s margin
    
    while time.time() - start_time < timeout:
        S = best_set[:]
        improved = False
        
        # Try additions
        unused = set(range(1, max_n + 1)) - set(S)
        if unused:
            random.shuffle(list(unused)) # Can't shuffle set, make list
            unused_list = list(unused)
            random.shuffle(unused_list)
            for x in unused_list:
                if len(S) >= max_size:
                    break
                new_S = S + [x]
                if is_valid(new_S):
                    S = new_S
                    improved = True
                    break # Restart search with new state
        
        # Try swaps (remove 1, add 1)
        if not improved and len(S) > 0:
            S_list = S[:]
            random.shuffle(S_list)
            for u in S_list:
                unused = set(range(1, max_n + 1)) - set(S)
                if not unused:
                    break
                unused_list = list(unused)
                random.shuffle(unused_list)
                for v in unused_list:
                    if u == v:
                        continue
                    new_S = [y for y in S if y != u] + [v]
                    if is_valid(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        
        # Try 2-for-1 swaps (remove 2, add 1) - increases density? No, decreases size.
        # We want to maximize size. So 1-for-2 swaps (remove 1, add 2)
        if not improved and len(S) < max_size:
            S_list = S[:]
            random.shuffle(S_list)
            for u in S_list:
                unused = set(range(1, max_n + 1)) - set(S)
                if len(unused) < 2:
                    break
                unused_list = list(unused)
                random.shuffle(unused_list)
                # Pick 2 random from unused
                if len(unused_list) >= 2:
                    v1, v2 = unused_list[0], unused_list[1]
                    if v1 != v2:
                        new_S = [y for y in S if y != u] + [v1, v2]
                        if is_valid(new_S):
                            S = new_S
                            improved = True
                            break
            if improved:
                break

        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            if best_len == max_size:
                break

        if not improved:
            # Stuck? Try random restart or perturb
            # Perturb: remove 2 random elements
            if len(S) > 2:
                S_list = S[:]
                random.shuffle(S_list)
                S = S_list[2:]
            else:
                S = greedy_init()

    return best_set
