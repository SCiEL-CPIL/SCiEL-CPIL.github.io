import random
import time

def check_distinct_sums(nums):
    sums = set()
    # Bitmask iteration for 12 elements
    n = len(nums)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial solution: Powers of 2
    current = [1 << i for i in range(12)]
    best = current[:]
    best_max = max(best)
    
    # Temperature for Simulated Annealing
    T = 1000.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        # Generate neighbor
        new_set = current[:]
        i = random.randint(0, 11)
        # Try to change current[i] to a smaller value, or random value
        # Bias towards smaller values to minimize max
        # Range: 1 to max(current)
        max_val = max(new_set)
        
        # Strategy: Try to reduce the largest element first
        # If the element to change is the largest, try smaller values
        # Otherwise, random swap or change
        
        # Pick a candidate value
        if new_set[i] == max_val:
            # Try to find a smaller valid value
            # Randomly pick a value from 1 to max_val - 1
            v = random.randint(1, max_val - 1)
        else:
            # Random change, maybe slightly biased to smaller
            v = random.randint(1, max_val + 10)
        
        if v in new_set:
            continue
            
        new_set[i] = v
        
        if not check_distinct_sums(new_set):
            continue
            
        new_max = max(new_set)
        
        if new_max < best_max:
            best = new_set[:]
            best_max = new_max
            current = new_set[:]
            T *= cooling_rate
        elif new_max == best_max:
            # Accept with some probability to explore
            if random.random() < 0.1:
                current = new_set[:]
                T *= cooling_rate
        else:
            # Accept worse solution with probability exp(-(delta)/T)
            delta = new_max - best_max
            prob = pow(2.71828, -delta / max(T, 1.0))
            if random.random() < prob:
                current = new_set[:]
        
        T *= cooling_rate
        if T < 1.0:
            T = 1.0

    return best
