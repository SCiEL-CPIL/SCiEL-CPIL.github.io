"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt
from itertools import combinations, product

NMAX = 256

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = b"\x00" * len(sieve[p * p::p])
    return [i for i in range(2, n + 1) if sieve[i]]

def _is_valid_set(S):
    """Check if set S satisfies: no a in S divides b*c for distinct b,c in S."""
    L = list(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            b = L[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if (b * L[k]) % a == 0:
                    return False
    return True

def _valid_addition(x, S_list):
    """Check if x can be added to set S_list without violating the constraint."""
    for i in range(len(S_list)):
        b = S_list[i]
        for j in range(len(S_list)):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _max_valid_subset(candidates, deadline):
    """Find the largest valid subset from candidates using greedy + local search."""
    if not candidates:
        return []
    
    # Greedy initialization: sort by value, try to add
    # Larger numbers are less likely to divide products of others?
    # Actually, for composites formed by small primes, structure matters.
    # Let's just try all permutations of a few candidates if small, or greedy.
    
    best = []
    best_set = set()
    
    # Simple greedy: try to add each candidate
    for x in sorted(candidates, reverse=True):
        if _valid_addition(x, best):
            best.append(x)
            best_set.add(x)
            
    # Local search to improve
    S = best[:]
    S_set = set(S)
    rng = random.Random(123)
    for _ in range(100):
        if time.time() > deadline:
            break
        # Try swapping
        if S:
            i = rng.randrange(len(S))
            rem = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            # Try to add a candidate not in S_temp
            for x in candidates:
                if x in S_temp_set:
                    continue
                if _valid_addition(x, S_temp):
                    S = S_temp + [x]
                    S_set = set(S)
                    break
            else:
                S = S_temp + [rem]
                S_set = set(S)
        if len(S) > len(best):
            best = S[:]
            best_set = set(S)
            
    return best

def _search_prime_swap(deadline):
    """Search for improvements by removing primes and adding composites."""
    primes = _primes_upto(NMAX)
    base_set = primes[:]
    best_set = base_set[:]
    best_size = len(base_set)
    
    # Try removing 1, 2, or 3 primes
    for k in range(1, 4):
        if time.time() > deadline - 5:
            break
        for p_out in combinations(primes, k):
            if time.time() > deadline - 2:
                break
            
            # Generate composites from p_out
            p_set = set(p_out)
            candidates = []
            # Generate numbers <= 256 whose prime factors are all in p_set
            # and are composite
            for x in range(4, NMAX + 1):
                temp = x
                valid = True
                for p in p_set:
                    while temp % p == 0:
                        temp //= p
                if temp != 1:
                    valid = False
                if valid and x not in p_set: # Must be composite (not in p_out)
                    # Also must not be 1
                    candidates.append(x)
            
            if not candidates:
                continue
                
            # Find max valid subset of candidates
            C = _max_valid_subset(candidates, deadline - 1)
            
            if len(C) > k:
                # Construct new set
                new_set = [p for p in primes if p not in p_out] + C
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                    # Verify validity just in case
                    if _is_valid_set(best_set):
                        pass
                    else:
                        best_size = len(base_set)
                        best_set = base_set[:]
                        
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # 1. Get the set of all primes
    primes = _primes_upto(NMAX)
    best_set = primes[:]
    best_size = len(primes)
    
    # 2. Try the prime-swap strategy
    if time.time() < deadline - 10:
        swapped_set = _search_prime_swap(deadline - 10)
        if len(swapped_set) > best_size:
            # Verify
            if _is_valid_set(swapped_set):
                best_set = swapped_set
                best_size = len(swapped_set)
            else:
                # If invalid, fallback to a valid subset or just keep primes
                pass
    
    # 3. General local search on the best set found so far
    # This might find other structures
    S = best_set[:]
    S_set = set(S)
    rng = random.Random(42)
    
    for it in range(500):
        if time.time() > deadline - 2:
            break
        
        # Try adding a random element
        if len(S) < NMAX:
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best_set = S[:]
                        best_size = len(S)
                    continue
        
        # Try swapping
        if S:
            i = rng.randrange(len(S))
            rem = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                # Try a few
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:10]:
                    if _valid_addition(x, S_temp):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best_set = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [rem]
                    S_set = set(S)
            else:
                S = S_temp + [rem]
                S_set = set(S)
                
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
