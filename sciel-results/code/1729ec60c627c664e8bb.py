import time
import random
import math

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _get_primes(limit):
    primes = []
    for i in range(2, limit + 1):
        if _is_prime(i):
            primes.append(i)
    return primes

def _greedy_order(order, primes, deadline):
    S = []
    S_set = set()
    # residue_sets[p] holds the set of residues mod p present in S
    residue_sets = {p: set() for p in primes}
    
    # We only need to check primes up to the maximum possible size of S
    # Max size is 5001. Primes up to 5001.
    
    for x in order:
        if time.time() > deadline:
            break
        
        new_size = len(S) + 1
        valid = True
        
        # Check constraints for all primes p <= new_size
        # Note: primes list is sorted. We can break early if p > new_size.
        for p in primes:
            if p > new_size:
                break
            r = x % p
            if r not in residue_sets[p]:
                # If this residue is new, adding x will increase the count of distinct residues for p.
                # The constraint is that we must NOT cover ALL residues.
                # So if len(residue_sets[p]) is already p-1, adding a new residue makes it p, which is invalid.
                if len(residue_sets[p]) == p - 1:
                    valid = False
                    break
        
        if valid:
            S.append(x)
            S_set.add(x)
            for p in primes:
                if p > len(S):
                    break
                r = x % p
                if r not in residue_sets[p]:
                    residue_sets[p].add(r)
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s safety margin for 300s limit
    
    N = 5000
    # Primes up to N+1
    primes = _get_primes(N + 1)
    
    best_S = []
    
    # Strategy: Try a few different orders.
    # 1. Natural order
    # 2. Reverse natural order
    # 3. Random orders
    
    orders = []
    base_order = list(range(N + 1))
    orders.append(base_order)
    orders.append(list(reversed(base_order)))
    
    # Add a few random permutations if time permits
    # We will run them in a loop, checking time.
    
    for i in range(100): # Max 100 attempts if time allows
        if time.time() > deadline:
            break
            
        if i < 2:
            order = orders[i]
        else:
            order = base_order.copy()
            random.shuffle(order)
            
        S = _greedy_order(order, primes, deadline)
        
        if len(S) > len(best_S):
            best_S = S
            
        # If we found a very large set, we might stop early?
        # The theoretical max is not known precisely but likely close to N.
        # Let's just keep running until deadline.
        
    return best_S
