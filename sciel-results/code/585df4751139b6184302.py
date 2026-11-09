import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 4096
    best_set = []
    
    def is_valid_addition(x, S, sums):
        """
        Checks if x can be added to S.
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        """
        new_sums = []
        # 3x
        new_sums.append(3*x)
        
        # 2x + y
        for y in S:
            new_sums.append(2*x + y)
            
        # x + 2y
        for y in S:
            new_sums.append(x + 2*y)
            
        # x + y + z
        for y, z in combinations_with_replacement(S, 2):
            new_sums.append(x + y + z)
            
        # Check for duplicates among new_sums
        if len(new_sums) != len(set(new_sums)):
            return False, None
            
        # Check for collisions with existing sums
        new_sums_set = set(new_sums)
        if new_sums_set & sums:
            return False, None
            
        return True, new_sums_set

    def greedy_build(order):
        S = []
        sums = set()
        for x in order:
            if x > N:
                continue
            valid, new_sums = is_valid_addition(x, S, sums)
            if valid:
                S.append(x)
                sums.update(new_sums)
        return S

    # 1. Try natural order
    best_set = greedy_build(range(1, N+1))
    
    # 2. Try random shuffles
    random.seed(42)
    candidates = list(range(1, N+1))
    
    # Run as many shuffles as possible before deadline
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = greedy_build(candidates)
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
