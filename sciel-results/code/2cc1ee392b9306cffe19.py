import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    def get_conflicts(x, S):
        """Return set of elements in S that conflict with x"""
        fx = prime_factors[x]
        conflicts = set()
        for y in S:
            if fx & prime_factors[y]:
                conflicts.add(y)
        return conflicts

    def local_search(S):
        """
        Perform best-improvement local search.
        Move: Add x not in S, remove all conflicts of x in S.
        Gain: x - sum(conflicts)
        """
        S = set(S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            best_gain = 0
            best_x = None
            best_conflicts = None
            
            # Iterate over all x not in S
            for x in range(1, N + 1):
                if x in S:
                    continue
                conflicts = get_conflicts(x, S)
                gain = x - sum(conflicts)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_conflicts = conflicts
            
            if best_x is not None:
                S -= best_conflicts
                S.add(best_x)
                improved = True
                
        return S

    candidates = []
    
    # Candidate 1: All primes + 1
    primes = [i for i in range(2, N + 1) if len(prime_factors[i]) == 1]
    S1 = set([1] + primes)
    candidates.append(local_search(S1))
    
    # Candidate 2: Greedy descending
    S2 = set()
    for x in range(N, 0, -1):
        if not get_conflicts(x, S2):
            S2.add(x)
    candidates.append(local_search(S2))
    
    # Candidate 3: Greedy ascending
    S3 = set()
    for x in range(1, N + 1):
        if not get_conflicts(x, S3):
            S3.add(x)
    candidates.append(local_search(S3))
    
    # Candidate 4-20: Random restarts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        S_rand = set()
        nums = list(range(1, N + 1))
        random.shuffle(nums)
        for x in nums:
            if not get_conflicts(x, S_rand):
                S_rand.add(x)
        candidates.append(local_search(S_rand))
        
    # Find best candidate
    best_S = max(candidates, key=sum)
    
    # Final polish
    best_S = local_search(best_S)
    
    # Ensure return type is list of ints
    return list(best_S)
