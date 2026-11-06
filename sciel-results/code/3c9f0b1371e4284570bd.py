import random
import math
import time

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    # Initialize with seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_sums = set()
    for a in best_A:
        for b in best_B:
            best_sums.add(a + b)
    best_cardinality = len(best_sums) # 5
    
    while time.time() < deadline:
        # Alternate or randomize between parity cases
        if random.random() < 0.5:
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        curr_cardinality = len(distinct_sums)
        
        # If we found a perfect 9, return immediately
        if curr_cardinality == 9:
            if is_pairwise_coprime(distinct_sums):
                return [A, B]
            # If not coprime, we can't use it, but maybe a different 9-sum set is?
            # Just continue searching.
            
        # If we found a better cardinality than seed
        if curr_cardinality > best_cardinality:
            if is_pairwise_coprime(distinct_sums):
                best_A = A
                best_B = B
                best_cardinality = curr_cardinality
                if curr_cardinality == 9:
                    return [A, B]

    return [best_A, best_B]
