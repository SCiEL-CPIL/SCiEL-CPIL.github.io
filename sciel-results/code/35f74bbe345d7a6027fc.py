import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    candidates = list(range(2, n + 1))
    
    def is_valid(S):
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(len(S_list)):
                if i == j: continue
                b = S_list[j]
                for k in range(len(S_list)):
                    if i == k or j == k: continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Initial set: all primes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    S = [i for i in range(2, n + 1) if is_prime[i]]
    S_set = set(S)
    best_S = S[:]
    best_size = len(S)
    
    # Local Search
    # We try to improve by swapping elements.
    # A move: remove x, add y.
    # If size increases (impossible with 1-for-1), or if we can add y without removing (gain).
    # Or if we remove x and add y, and then later add z...
    
    # Let's try to find any set larger than 53.
    # If not found, return 53.
    
    # Heuristic: Try to add composites by removing conflicting primes.
    # If removing 1 prime allows adding 2 composites, we gain 1.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding each non-member
        for c in candidates:
            if c in S_set:
                continue
            if time.time() > deadline - 5:
                break
            
            # Find conflicts: elements a in S such that S \ {a} U {c} is valid?
            # No, we want S U {c} to be valid? (Gain 1)
            # Or S \ {a} U {c} valid? (Swap, size same)
            # Or S \ {a,b} U {c,d} valid? (Complex)
            
            # Simple Gain: Can we add c without removing?
            # Check if c conflicts with any pair in S.
            valid_add = True
            for a in S:
                for b in S:
                    if a == b: continue
                    # Check a | c*b, c | a*b, b | c*a
                    if (c * b) % a == 0 or (a * b) % c == 0 or (c * a) % b == 0:
                        valid_add = False
                        break
                if not valid_add:
                    break
            
            if valid_add:
                S.append(c)
                S_set.add(c)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                improved = True
                continue
            
            # If no gain, try Swap: Remove a, Add c.
            # If we can swap, size is same. But might enable future gains.
            # To save time, only try swaps if we haven't found a gain recently?
            # Or just try a few random swaps.
            
            # Let's try to find a 'a' to remove such that S\{a} U {c} is valid.
            # This is expensive.
            # Approximation: Remove 'a' if 'a' is a prime factor of c?
            # If c = p*k, and p in S, then p | c*b for any b.
            # So p MUST be removed to add c.
            
            prime_factors_of_c = []
            temp_c = c
            for p in S:
                if p > temp_c: break
                if temp_c % p == 0:
                    prime_factors_of_c.append(p)
                    while temp_c % p == 0:
                        temp_c //= p
            
            if prime_factors_of_c:
                # Try removing one of the prime factors
                for p in prime_factors_of_c:
                    if p not in S_set: continue
                    # Tentative swap
                    S.remove(p)
                    S_set.remove(p)
                    S.append(c)
                    S_set.add(c)
                    
                    if is_valid(S):
                        # Valid swap. Size is same.
                        # Do we keep it?
                        # It might allow adding another composite later.
                        # For now, let's keep it if it doesn't reduce size.
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                        improved = True
                        break
                    else:
                        # Revert
                        S.remove(c)
                        S_set.remove(c)
                        S.append(p)
                        S_set.add(p)
            
    # Final check
    if is_valid(best_S):
        return best_S
    return S

# EVOLVE-BLOCK-END
