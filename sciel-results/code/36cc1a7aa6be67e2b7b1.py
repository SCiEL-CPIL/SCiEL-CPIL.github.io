import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 192
    best_set = []
    best_size = 0
    
    # Candidate pool: Focus on larger numbers first as they are "safer"
    # But we should consider all numbers in case a mix is better.
    # However, theoretical max is 48. 
    # If we can get 48 from [100, 192], that's great.
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        S_list = list(S)
        # 1. x does not divide sum of any 3 distinct others in S
        if len(S_list) >= 3:
            for a, b, c in combinations(S_list, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. No a in S divides x + b + c for distinct b, c in S \ {a}
        if len(S_list) >= 2:
            for a in S_list:
                # We need 2 distinct others from S \ {a}
                # The sum is x + b + c
                # Check if a divides (x + b + c)
                others = [v for v in S_list if v != a]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (x + b + c) % a == 0:
                        return False
        return True

    # Randomized Greedy Construction
    def greedy_random(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
        return list(S)

    # Try different orders
    attempts = 0
    max_attempts = 500 # Limit attempts to allow time check
    
    # Strategies:
    # 1. Descending (Large to Small)
    # 2. Random
    # 3. Biased Random (More large numbers)
    # 4. Ascending (Small to Large) - likely worse but try
    
    while time.time() < deadline - 2:
        attempts += 1
        
        if attempts % 5 == 0:
            # Descending
            order = list(range(n, 0, -1))
        elif attempts % 5 == 1:
            # Ascending
            order = list(range(1, n + 1))
        elif attempts % 5 == 2:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempts % 5 == 3:
            # Biased: Large numbers first
            large = list(range(n, n//2, -1))
            small = list(range(n//2, 0, -1))
            random.shuffle(large)
            random.shuffle(small)
            order = large + small
        else:
            # Mixed Random
            order = list(range(1, n + 1))
            random.shuffle(order)
            # Sort by value with some noise?
            # Or just shuffle again
            
        S = greedy_random(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
            if best_size == 48:
                break
                
    return best_set
