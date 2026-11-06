import math
import time

def construct():
    deadline = time.time() + 115.0
    
    # Sieve primes up to 10^7
    # This covers p up to 10^7. If we find a good p, we return it.
    # If not, we might need to go higher, but 10^7 is a good balance.
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    limit = 10**7
    primes = sieve_primes(limit)
    
    best_k = 0
    best_p = 2

    # We will iterate through primes p.
    # For each p, we check the range [p^2+1, p^2+K_max]
    # K_max is limited by time and the fact that k <= 200.
    
    # To optimize, we can skip some primes or check in batches?
    # The check is fast, so we can check many primes.
    
    for idx, p in enumerate(primes):
        if time.time() > deadline - 5:
            break
            
        # If p is too large, p^2 might overflow? No, Python handles big ints.
        # But the range size is fixed (<= 200).
        
        # We want to find the max k such that p^2+1 .. p^2+k are p-smooth.
        # We test up to k=200.
        
        base = p * p
        k_test = 200
        
        # rem[i] corresponds to base + i + 1
        # rem[0] = base + 1
        rem = [base + i + 1 for i in range(k_test)]
        
        # For each prime q <= p, divide rem[i] by q as much as possible
        # We only need primes q such that q <= p.
        # Since primes is sorted, we use primes[:idx+1]
        
        for j in range(idx + 1):
            q = primes[j]
            # Find the first i in [0, k_test-1] such that (base + i + 1) % q == 0
            # Let N_i = base + i + 1
            # N_i = base + 1 + i
            # We want (base + 1 + i) % q == 0
            # i = (-base - 1) % q
            start_i = (-base - 1) % q
            
            # Iterate over i = start_i, start_i + q, ... < k_test
            for i in range(start_i, k_test, q):
                val = rem[i]
                while val % q == 0:
                    val //= q
                rem[i] = val
                # Early exit for this i if val becomes 1?
                # If val == 1, it's fully factored by primes <= p (so far).
                # But we still need to check if it remains 1 after all primes.
                # Actually, if it becomes 1, it stays 1.
        
        # Now rem[i] == 1 iff base + i + 1 is p-smooth
        # Find the longest prefix of 1s
        current_k = 0
        for i in range(k_test):
            if rem[i] == 1:
                current_k = i + 1
            else:
                break
        
        if current_k > best_k:
            best_k = current_k
            best_p = p
            
        # Optimization: If best_k is already 200, we can't do better.
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}
