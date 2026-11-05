import random
import time

def construct():
    # We will use a randomized greedy approach with multiple restarts.
    # The goal is to find a large subset of {1, ..., 1000} such that
    # every sum a+b (a<=b) has at most 2 representations.
    
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for overhead
    
    best_set = []
    best_size = 0
    
    # Helper function to run greedy given an order
    def run_greedy(order):
        current = []
        counts = {}
        # To speed up conflict checking, we can maintain a set of 'full' sums (count == 2)
        full_sums = set()
        
        for x in order:
            # Calculate new sums
            # Sums are x+y for y in current, and 2*x
            # Check if any new sum is already full
            valid = True
            new_sums = []
            
            # Optimization: Check conflicts first
            for y in current:
                s = x + y
                if s in full_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                s_double = 2 * x
                if s_double in full_sums:
                    valid = False
                else:
                    new_sums.append(s_double)
            
            if not valid:
                continue
            
            # Check if any new sum collides with another new sum?
            # x+y1 == x+y2 => y1==y2. Distinct y gives distinct sums.
            # x+y == 2x => y=x. But y in current, x not in current. So distinct.
            # So new_sums are all distinct.
            
            # Check if adding these sums causes any count to exceed 2
            # We already checked against full_sums.
            # Now we need to check if any s in new_sums is already in counts with count 1?
            # If count is 1, it becomes 2 (full). That's allowed.
            # If count is 0, it becomes 1. That's allowed.
            # So we just need to update.
            
            # Update state
            current.append(x)
            for s in new_sums:
                c = counts.get(s, 0)
                if c == 1:
                    full_sums.add(s)
                    counts[s] = 2
                else:
                    counts[s] = 1
            
        return current

    # 1. Try a few deterministic orders
    base_orders = [
        list(range(1, 1001)),
        list(range(1000, 0, -1)),
        # Maybe a sparse order?
        list(range(1, 1001, 2)), # odds
        list(range(2, 1001, 2))  # evens
    ]
    
    for order in base_orders:
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    # 2. Randomized restarts
    while time.time() - start_time < time_limit:
        order = list(range(1, 1001))
        random.shuffle(order)
        
        res = run_greedy(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    return best_set
