import time
import heapq

def construct():
    S = 123457
    deadline = time.time() + 50
    
    # Sieve of Eratosthenes to get primes up to S
    # We need primes to test smoothness bounds
    is_prime = [True] * (S + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(S**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, S + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def check_smooth(B):
        # Generate all B-smooth numbers <= S
        # B is the max prime factor allowed
        # Primes to use: all primes <= B
        allowed_primes = [p for p in primes if p <= B]
        
        if not allowed_primes:
            # Only 1 is 0-smooth? P(1)=1. If B < 2, only 1 is smooth.
            # If B=1, allowed_primes is empty.
            # 1 is B-smooth.
            # Check if 1 + (S-1) works? S-1 must be B-smooth.
            # If B=1, S-1 must be 1-smooth, i.e., S-1=1 -> S=2.
            # For S=123457, B=1 yields no pair.
            pass

        smooths = {1}
        heap = [1]
        
        # To avoid duplicates, we can use the standard algorithm for generating smooth numbers
        # But a simple BFS/DFS with a set is easier to implement correctly in short time.
        # Given S is small, this should be fine.
        
        # Actually, a more efficient way is to iterate through the set and expand.
        # But heap is good for generating in order, though we don't strictly need order for the set check.
        
        # Let's use a queue/BFS style
        queue = [1]
        i = 0
        while i < len(queue):
            x = queue[i]
            i += 1
            for p in allowed_primes:
                nx = x * p
                if nx > S:
                    break
                if nx not in smooths:
                    smooths.add(nx)
                    queue.append(nx)
                    
        # Check for pair
        for x in smooths:
            if S - x in smooths:
                return True, x, S - x
        return False, 0, 0

    # Binary search on primes
    # The answer must be a prime (or 1)
    # Low: index of smallest prime (2), High: index of largest prime (S)
    # Actually, we can just iterate primes? 
    # If the optimal B is small, linear scan of primes is fast.
    # If optimal B is large, binary search is better.
    # Let's try binary search on the list of primes.
    
    lo = 0
    hi = len(primes) - 1
    best_a, best_b = 1, S - 1
    best_val = primes[-1] # Fallback
    
    # Special case: B=1
    # Only 1 is 1-smooth.
    # Pair (1, S-1). S-1 must be 1-smooth.
    # S-1 = 123456 != 1. So B=1 fails.
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        B = primes[mid]
        
        found, a, b = check_smooth(B)
        
        if found:
            best_a, best_b = a, b
            best_val = B
            hi = mid - 1
        else:
            lo = mid + 1
            
    return [best_a, best_b]
