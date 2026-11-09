import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s out of 60s

    # Known valid solution: {66, ..., 100}
    # Size 35
    base_set = list(range(66, 101))
    best_size = len(base_set)
    best_set = base_set

    # Try to find a larger set (size 36)
    # Heuristic: Start with {64..100} and remove minimal conflicts
    # Or random restarts
    
    def is_valid(S):
        # Check if any subset sums to 200
        # Since max element is 100, subset size <= 2 for sum < 200? No.
        # But if min element is high, subset size is small.
        # For general set, use DP or bitmask if small.
        # Here, we can just check sums of size 1, 2, 3, 4.
        # If min element > 50, size 4 sum > 200.
        # So check 1, 2, 3.
        S = sorted(S)
        if len(S) == 0: return True
        
        # Size 1
        if 200 in S: return False
        
        # Size 2
        seen = set()
        for x in S:
            if 200 - x in seen:
                return False
            seen.add(x)
            
        # Size 3
        # O(N^2) check
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                rem = 200 - S[i] - S[j]
                # Check if rem is in S and > S[j]
                # Binary search or set lookup
                # We need distinct indices
                # Since S is sorted, we can use set
                # But we must ensure rem is not S[i] or S[j]
                # And rem must be in S
                if rem in S:
                    if rem != S[i] and rem != S[j]:
                        return False
        return True

    # Try to construct size 36
    # Strategy: Take {64..100} and remove 1 element? No, we proved need 3.
    # Take {63..100}?
    
    # Randomized local search
    # Start with a random set of size 36 from [1..100]
    # Prefer larger numbers
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Generate candidate
        # Bias towards larger numbers
        candidates = list(range(1, 101))
        random.shuffle(candidates)
        # Sort by value descending, then pick top 36?
        # Or pick 36 random, ensuring min is high
        # Let's try picking 36 from 60..100
        pool = list(range(60, 101))
        if len(pool) < 36:
            pool = list(range(1, 101))
        
        random.shuffle(pool)
        cand = pool[:36]
        
        if is_valid(cand):
            if len(cand) > best_size:
                best_size = len(cand)
                best_set = cand
            # Found one, try to improve?
            # Or just return?
            # Let's keep searching for 37
    
    return best_set
