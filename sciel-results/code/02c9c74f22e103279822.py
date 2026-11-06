import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s margin
    
    def _is_valid_set(arr):
        """Check if no element in arr is the mean of any subset of >=2 other elements."""
        n = len(arr)
        if n < 3:
            return True
        arr = sorted(arr)
        for idx, a in enumerate(arr):
            others = [arr[j] for j in range(n) if j != idx]
            m = len(others)
            if m < 2:
                continue
            # For each k >= 2, check if k*a can be formed as sum of exactly k elements from others
            # Use bitset DP: dp[i] is a bitset where bit s is set if sum s is achievable with i elements
            # Max sum of m elements each <= 512 is m*512
            max_sum = sum(others)
            # dp[i] as a Python int (arbitrary precision bitset)
            dp = [0] * (m + 1)
            dp[0] = 1  # bit 0 set: sum 0 with 0 elements
            
            for y in others:
                # Process in reverse to avoid reusing y
                for i in range(m, 0, -1):
                    # dp[i] |= dp[i-1] << y
                    dp[i] |= (dp[i-1] << y)
            
            # Check if any k*a is achievable with exactly k elements
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    continue
                if (dp[k] >> target) & 1:
                    return False
        return True
    
    def _can_add(x, current_set):
        """Check if adding x to current_set keeps the set valid."""
        new_set = current_set + [x]
        return _is_valid_set(new_set)
    
    # Greedy construction: try numbers in decreasing order (larger numbers less likely to be means)
    selected = []
    
    # Start with a few large numbers that are unlikely to be means
    # Then try to fill in with more numbers
    
    candidates = list(range(512, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _can_add(x, selected):
            selected.append(x)
    
    # Local improvement: try to swap out smaller numbers for larger ones if possible
    if time.time() < deadline - 10:
        improved = True
        while improved and len(selected) < 48 and time.time() < deadline - 5:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                for x in range(512, max(selected), -1):
                    if x in selected:
                        continue
                    # Try swapping selected[i] with x
                    new_set = selected[:i] + [x] + selected[i+1:]
                    if _is_valid_set(new_set):
                        selected = new_set
                        improved = True
                        break
                if improved:
                    break
    
    # Try to add more elements
    if time.time() < deadline - 10 and len(selected) < 48:
        for x in range(1, 513):
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            if x not in selected:
                if _can_add(x, selected):
                    selected.append(x)
    
    return selected
