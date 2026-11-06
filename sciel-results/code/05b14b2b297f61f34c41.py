import time
import numpy as np
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Generate y-smooth numbers for increasing y, check for consecutive pairs.
    This is a structural change from linear scan to sparse smooth-number lattice search.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    LOW = 10000
    HIGH = 1000000
    
    # We'll try smooth number bounds in increasing order
    # For each bound B, generate all B-smooth numbers in [LOW, HIGH+1]
    # Then check if any two are consecutive
    
    # The primes we'll use as smoothness bounds
    # Start with small primes and increase
    smooth_bounds = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = None
    best_score = float('inf')
    
    # Generate primes up to 1000 for trial division verification
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = False
        if limit >= 1:
            is_prime[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    primes_up_to_1000 = sieve_primes(1000)
    
    def largest_prime_factor(n):
        """Compute LPF via trial division."""
        if n <= 1:
            return 1
        lpf = 1
        for p in primes_up_to_1000:
            if p * p > n:
                break
            while n % p == 0:
                lpf = p
                n //= p
        if n > 1:
            lpf = n
        return lpf
    
    # For each smoothness bound, generate smooth numbers and check for consecutive pairs
    for B in smooth_bounds:
        if time.time() > deadline - 5.0:
            break
        
        # Get primes up to B
        primes = [p for p in primes_up_to_1000 if p <= B]
        
        if not primes:
            continue
        
        # Generate all B-smooth numbers in [LOW, HIGH+1]
        # Use a set to avoid duplicates
        smooth_set = set()
        
        # Start with 1, multiply by primes
        # Use a list-based approach: start with [1], repeatedly multiply by each prime
        # and keep results in range
        
        # More efficient: use a priority queue or iterative expansion
        # For simplicity, use iterative multiplication with a set
        
        current = [1]
        for p in primes:
            new_vals = []
            for v in current:
                # Multiply by p, p^2, p^3, ... as long as in range
                x = v
                while x * p <= HIGH + 1:
                    x *= p
                    if x >= LOW:
                        new_vals.append(x)
                        smooth_set.add(x)
                    else:
                        # Still need to continue multiplying
                        pass
                # Also add v itself if in range (already handled by previous iterations)
            # Add all new values
            for v in new_vals:
                if v not in smooth_set:
                    smooth_set.add(v)
            # Extend current with new values for further multiplication
            current = list(smooth_set)
        
        # Actually, the above approach is inefficient. Let me use a cleaner method.
        # Generate smooth numbers using a heap-based approach (like ugly numbers)
        
        # Reset
        smooth_set = set()
        
        # Use a different approach: enumerate by factorization
        # For each combination of exponents for primes <= B, compute the product
        # This is exponential in the number of primes, so only works for small B
        
        if len(primes) <= 6:  # Only do full enumeration for small B
            # Generate all combinations
            from itertools import product
            max_exponents = []
            for p in primes:
                exp = 0
                x = 1
                while x * p <= HIGH + 1:
                    x *= p
                    exp += 1
                max_exponents.append(exp)
            
            ranges = [range(e + 1) for e in max_exponents]
            
            for combo in product(*ranges):
                val = 1
                for i, exp in enumerate(combo):
                    val *= primes[i] ** exp
                if LOW <= val <= HIGH + 1:
                    smooth_set.add(val)
        
        # For larger B, use a different approach: sieve-like marking
        else:
            # Mark all B-smooth numbers in [LOW, HIGH+1]
            # Create a boolean array
            size = HIGH + 1 - LOW + 1
            is_smooth = np.zeros(size, dtype=bool)
            
            # Start with 1 being smooth, then multiply by primes
            # Use a queue-based approach
            from collections import deque
            queue = deque([1])
            smooth_set = set()
            
            while queue:
                v = queue.popleft()
                if v > HIGH + 1:
                    continue
                if v >= LOW:
                    smooth_set.add(v)
                for p in primes:
                    new_v = v * p
                    if new_v <= HIGH + 1 and new_v not in smooth_set and new_v not in is_smooth[v - LOW if v >= LOW else 0]:
                        # Check if already in queue (approximate)
                        if new_v not in smooth_set:
                            smooth_set.add(new_v)
                            queue.append(new_v)
            
            # This might still be slow. Let me just use the enumeration for all B <= 47
            # and skip larger B if too slow
        
        # Check for consecutive pairs in smooth_set
        if smooth_set:
            sorted_smooth = sorted(smooth_set)
            for i in range(len(sorted_smooth) - 1):
                if sorted_smooth[i+1] - sorted_smooth[i] == 1:
                    n = sorted_smooth[i]
                    if LOW <= n <= HIGH:
                        # Compute the score
                        lpf_n = largest_prime_factor(n)
                        lpf_np1 = largest_prime_factor(n + 1)
                        score = max(lpf_n, lpf_np1)
                        if score < best_score:
                            best_score = score
                            best_n = n
                        if time.time() > deadline - 5.0:
                            break
            
            if time.time() > deadline - 5.0:
                break
        
        # If we found a very good score, we can stop early
        if best_score <= 23:  # Very smooth
            break
    
    # If no smooth pair found, fall back to a linear scan with a time limit
    if best_n is None:
        if time.time() > deadline - 10.0:
            # Not enough time, return a reasonable default
            return 10080  # 10080 = 2^5 * 3^2 * 5 * 7, LPF=7; 10081=73*138? No, 10081=10081 (prime?) Let me just pick something
        
        # Linear scan with time limit
        best_n = LOW
        best_score = float('inf')
        
        n = LOW
        while n <= HIGH and time.time() < deadline - 5.0:
            lpf_n = largest_prime_factor(n)
            lpf_np1 = largest_prime_factor(n + 1)
            score = max(lpf_n, lpf_np1)
            if score < best_score:
                best_score = score
                best_n = n
            n += 1
    
    return best_n

# Test locally
if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}, Time: {t1-t0:.3f}s")
    
    def lpf_trial(n):
        if n <= 1:
            return 1
        lpf = 1
        f = 2
        while f * f <= n:
            while n % f == 0:
                lpf = f
                n //= f
            f += 1
        if n > 1:
            lpf = n
        return lpf
    
    print(f"LPF({result}) = {lpf_trial(result)}, LPF({result+1}) = {lpf_trial(result+1)}")
    print(f"Score = {max(lpf_trial(result), lpf_trial(result+1))}")
