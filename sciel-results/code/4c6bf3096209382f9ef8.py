"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

MAX_VAL = 10000
N = 12

# Precompute prime factors for all numbers up to 2*MAX_VAL
MAX_SUM = 2 * MAX_VAL
_pf_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_prime_factors():
    for i in range(2, MAX_SUM + 1):
        n = i
        f = 2
        while f * f <= n:
            while n % f == 0:
                _pf_cache[i].add(f)
                n //= f
            f += 1
        if n > 1:
            _pf_cache[i].add(n)

_init_prime_factors()

def count_distinct_primes(nums):
    primes = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            primes |= _pf_cache[nums[i] + nums[j]]
    return len(primes)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    # Strategy:
    # 1. Try to find a set of 12 numbers such that all pairwise sums have prime factors 
    #    from a small set P_k (first k primes).
    # 2. If successful, return that set. Score is <= k.
    # 3. If not, use a greedy construction that minimizes the number of new primes added,
    #    then polish with local search.
    
    best_solution = None
    best_score = float('inf')
    
    SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Helper: greedy construction for a specific prime set P
    def greedy_construct(prime_set):
        prime_set = set(prime_set)
        S = []
        
        # We need to pick 12 numbers.
        for restart in range(20):
            if time.time() - start_time > time_limit - 5:
                return None
                
            S = []
            # Pick first number: random
            x1 = random.randint(1, MAX_VAL)
            S.append(x1)
            
            success = True
            for step in range(1, N):
                # Find a candidate x not in S
                # such that for all s in S, (x+s) is smooth w.r.t prime_set
                
                found = False
                for _ in range(500): # Try 500 random candidates
                    if time.time() - start_time > time_limit - 5:
                        return None
                    x = random.randint(1, MAX_VAL)
                    if x in S:
                        continue
                    
                    valid = True
                    for s in S:
                        if not _pf_cache[x+s].issubset(prime_set):
                            valid = False
                            break
                    if valid:
                        S.append(x)
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                return S
        
        return None

    # Try increasing k
    for k in range(2, 12):
        if time.time() - start_time > time_limit - 10:
            break
            
        P_k = SMALL_PRIMES[:k]
        
        # Try greedy construction
        sol = greedy_construct(P_k)
        if sol:
            # Calculate actual score
            primes_used = set()
            for i in range(N):
                for j in range(i+1, N):
                    primes_used |= _pf_cache[sol[i] + sol[j]]
            score = len(primes_used)
            if score < best_score:
                best_score = score
                best_solution = sol[:]
                
            if score <= 3:
                return best_solution

    # If no smooth solution found, or time is up, use local search from a good heuristic start
    
    # Heuristic Start: Greedy minimization of distinct primes
    if best_solution is None:
        S = []
        for i in range(N):
            best_x = None
            best_new_primes = float('inf')
            # Try a few random candidates
            candidates = random.sample(range(1, MAX_VAL+1), 200)
            for x in candidates:
                if x in S: continue
                new_primes = set()
                for s in S:
                    new_primes |= _pf_cache[x+s]
                cnt = len(new_primes)
                if cnt < best_new_primes:
                    best_new_primes = cnt
                    best_x = x
            if best_x is not None:
                S.append(best_x)
        best_solution = S
        
    # Local Search
    current = best_solution[:]
    current_score = count_distinct_primes(current)
    
    best_sol = current[:]
    best_sc = current_score
    
    random.seed(42)
    while time.time() - start_time < time_limit:
        i = random.randint(0, N-1)
        old_val = current[i]
        new_val = random.randint(1, MAX_VAL)
        if new_val == old_val:
            continue
        # Check distinctness
        if new_val in current[:i] or new_val in current[i+1:]:
            continue
            
        current[i] = new_val
        
        new_score = count_distinct_primes(current)
        
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_sc:
                best_sc = new_score
                best_sol = current[:]
        else:
            current[i] = old_val
            
    return best_sol

# EVOLVE-BLOCK-END
