import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    best_set = []
    
    # Helper to check if adding x to S is valid
    def is_valid_addition(S, x):
        S_list = list(S) + [x]
        # Check each element a in S_list
        for a in S_list:
            others = [y - a for y in S_list if y != a]
            # We need a subset of 'others' summing to 0 with size >= 2
            # DP: dp[sum] = bitset of sizes? Or just reachable sums?
            # We need to know if 0 is reachable with size >= 2.
            # dp[sum] = min_size? No, we need existence of size >= 2.
            # Let's use dp[sum] = bitmask of possible sizes?
            # Or simply: can we reach sum 0 with exactly k elements?
            # Range of sums:
            # Max abs sum: len(others) * 512
            # Let's shift sums by offset.
            
            n = len(others)
            if n < 2:
                continue
                
            max_sum = sum(abs(v) for v in others)
            offset = max_sum
            # dp[s] = set of sizes? Or just boolean?
            # If we just check if 0 is reachable, we might get size 1 (which is invalid) or size 0 (invalid).
            # We need size >= 2.
            
            # dp[sum] = min number of elements to reach sum?
            # If min_size[0] >= 2, then valid? No, if min is 1, there might be another path with size 2.
            # We need to know if there is ANY path with size >= 2.
            
            # Let's use dp[sum] = bitmask of sizes.
            # Size up to 20. Bitmask fits in int.
            # dp[s] is an integer where bit k is 1 if sum s is reachable with k elements.
            
            size = 2 * max_sum + 1
            dp = [0] * size
            dp[offset] = 1 # 0 elements, sum 0 (shifted)
            
            for v in others:
                new_dp = dp[:] # Not efficient, but n is small
                # Actually, we need to update in place or use new array.
                # Since we are adding one element, we can iterate backwards or use a copy.
                # But we want to combine with previous states.
                # Standard subset sum DP.
                
                # Optimization: Only keep reachable sums.
                # But array is small enough.
                
                # To avoid using x multiple times, we iterate over sums.
                # But we are adding 'v' to existing subsets.
                
                # Let's just do a simple DFS/BFS for small n?
                # n <= 20. 2^20 = 1M.
                # For each a, we do this. |S| <= 20.
                # 20 * 1M = 20M ops. Might be slow in Python.
                
                # Let's stick to DP.
                # dp[s] is bitmask.
                # When adding v, new_sum = s + v.
                # new_dp[new_sum] |= dp[s] << 1
                # We must be careful not to use v twice.
                # So we iterate s from current_max down to current_min?
                # Or just create new_dp from scratch.
                
                new_dp = [0] * size
                for s in range(size):
                    if dp[s]:
                        # Add v
                        ns = s + v
                        if 0 <= ns < size:
                            new_dp[ns] |= (dp[s] << 1)
                        # Not add v
                        new_dp[s] |= dp[s]
                dp = new_dp
                
            # Check if sum 0 (index offset) is reachable with size >= 2
            mask = dp[offset]
            # mask has bits set for reachable sizes.
            # We want bits 2, 3, ... to be set.
            # mask & ~3 checks if any bit >= 2 is set.
            if mask & ~3:
                return False
                
        return True

    # Try to find a good set
    # Strategy: Greedy with random restarts
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        S = set()
        # Randomize order
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # Bias towards small numbers? Or powers of 2?
        # Let's try a mix.
        
        for x in candidates:
            if len(S) >= 48:
                break
            if is_valid_addition(S, x):
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = list(S)
            
    # Also try the powers of 2 explicitly
    p2 = [2**i for i in range(10) if 2**i <= 512]
    if len(p2) > len(best_set):
        best_set = p2
        
    # Try a few more specific structures?
    # Geometric progression with ratio 3?
    g3 = []
    val = 1
    while val <= 512:
        g3.append(val)
        val *= 3
    if len(g3) > len(best_set):
        best_set = g3

    return best_set
