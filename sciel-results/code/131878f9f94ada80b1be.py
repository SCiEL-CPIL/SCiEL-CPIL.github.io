import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # Greedy construction with fast DP-based validation
    selected = []
    
    def is_valid_addition(x, current_set):
        """Check if adding x to current_set keeps the set valid."""
        new_set = current_set + [x]
        n = len(new_set)
        if n < 3:
            return True
        
        # For each element a in new_set, check if a is the mean
        # of any subset of size k>=2 from new_set\{a}
        # i.e., check if k*a is achievable as sum of exactly k distinct elements from new_set\{a}
        
        for a in new_set:
            others = [y for y in new_set if y != a]
            m = len(others)
            if m < 2:
                continue
            
            # DP: dp[i] = set of achievable sums using exactly i elements
            # Use boolean array: dp[i][s] = True if sum s achievable with i elements
            max_sum = sum(others)
            # We only care about sums that are multiples of a: k*a for k in [2, m]
            # So we can limit the sum range
            
            # dp[i] is a boolean array of achievable sums with i elements
            # Initialize
            dp = [set() for _ in range(m + 1)]
            dp[0].add(0)
            
            for y in others:
                # Process in reverse to avoid using y multiple times
                for i in range(min(m, len(others)), 0, -1):
                    for s in list(dp[i-1]):
                        new_s = s + y
                        if new_s <= max_sum:
                            dp[i].add(new_s)
            
            # Check if any k*a is in dp[k] for k >= 2
            for k in range(2, m + 1):
                if k * a in dp[k]:
                    return False
        
        return True
    
    # Try adding numbers in order, but be smart about which to try
    # Start with a good initial set, then try to add more
    # Greedy: try each number from 1 to 512, add if valid
    
    # Better: try to maximize count by being selective
    # Let's try numbers in increasing order first, then try to fill gaps
    
    candidates = list(range(1, 513))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if is_valid_addition(x, selected):
            selected.append(x)
    
    return selected
