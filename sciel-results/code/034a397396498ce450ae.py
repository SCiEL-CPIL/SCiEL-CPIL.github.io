import random
import time

def construct():
    # Calculate LCM of 1 to 22
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    D = 1
    for i in range(1, 23):
        D = lcm(D, i)
        
    def greedy(order):
        current_sums = {0}
        current_set = []
        for x in order:
            inc = D // x
            # Optimization: Check if inc is 0? No, D is multiple of x.
            # Generate new sums
            # Using list comprehension + set is often faster than loop+add for small sets
            # But for large sets, set update is better.
            # Let's stick to the loop for clarity and consistent overhead, or use set comprehension.
            # new_sums = {s + inc for s in current_sums}
            # This is faster in CPython.
            new_sums = {s + inc for s in current_sums}
            
            # Check for collision
            if new_sums.isdisjoint(current_sums):
                current_sums.update(new_sums)
                current_set.append(x)
        return current_set

    best_set = []
    
    # 1. Deterministic strategies
    # Descending x (small increments first)
    s1 = greedy(range(22, 0, -1))
    if len(s1) > len(best_set):
        best_set = s1
        
    # Ascending x (large increments first)
    s2 = greedy(range(1, 23))
    if len(s2) > len(best_set):
        best_set = s2

    # 2. Randomized strategies
    start_time = time.time()
    max_time = 55.0
    elements = list(range(1, 23))
    
    # Fix seed for reproducibility in testing? No, random is fine.
    # But if we want to beat the current score, we just need to find a better set.
    
    while time.time() - start_time < max_time:
        random.shuffle(elements)
        candidate = greedy(elements)
        if len(candidate) > len(best_set):
            best_set = candidate
            # If we hit the theoretical max or a very high number, we can stop early?
            # Max is 22. If we get 22, stop.
            if len(best_set) == 22:
                break

    return best_set
