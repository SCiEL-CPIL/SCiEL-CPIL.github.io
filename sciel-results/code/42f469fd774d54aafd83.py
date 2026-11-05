import time
import math
from heapq import heappush, heappop

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    
    # Fallback solution
    best_a, best_b = 1, target - 1
    
    # Generate primes up to a reasonable limit
    # We need primes to define the bounds B.
    # If the optimal B is large, we need primes up to that B.
    # Given target ~ 1e5, B won't be huge. Let's sieve up to 10000.
    limit_primes = 10000
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit_primes)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # We will iterate through primes as the upper bound B for smoothness.
    # For each B, we generate all B-smooth numbers <= target.
    
    # To speed up generation, we can maintain the list of primes <= B.
    # However, regenerating the smooth set for each B from scratch is O(|S| * pi(B)).
    # |S| for B=100 is small.
    # Let's try iterating B over primes.
    
    # Optimization: We can stop if B > target (trivial case).
    
    for idx, B in enumerate(primes):
        if time.time() > deadline:
            break
            
        # Get all primes <= B
        # Since primes list is sorted, primes[:idx+1] are <= B
        p_list = primes[:idx+1]
        
        # Generate B-smooth numbers <= target
        # Using a set and a queue (BFS)
        smooth_nums = set()
        queue = [1]
        smooth_nums.add(1)
        
        # To avoid using a slow queue with append/pop(0), use heapq or just a list with index
        # Given small size, a list with index is fine.
        q_idx = 0
        q = [1]
        
        while q_idx < len(q):
            n = q[q_idx]
            q_idx += 1
            for p in p_list:
                m = n * p
                if m > target:
                    break
                if m not in smooth_nums:
                    smooth_nums.add(m)
                    q.append(m)
                    
        # Check for solution
        # We need a in smooth_nums such that target - a in smooth_nums
        # To ensure a, b are positive, a must be in [1, target-1]
        # smooth_nums contains 1 and target (if target is smooth).
        # If a=1, b=target. P(1)=1, P(target)=target. max=target.
        # We want to minimize this.
        
        found = False
        # Iterate through smooth_nums
        # To avoid duplicate checks (a,b) and (b,a), we can check a <= target/2
        # But set lookup is fast.
        for a in smooth_nums:
            b = target - a
            if b in smooth_nums:
                # Found optimal solution for this B
                return [a, b]
                
    # If loop finishes without finding a solution (or timeout), return fallback
    return [best_a, best_b]
