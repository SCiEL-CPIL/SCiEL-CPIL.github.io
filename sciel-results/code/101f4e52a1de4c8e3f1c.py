import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

MAX_PRIME_FOR_PRECOMP = 1000
ALL_PRIMES = _primes(MAX_PRIME_FOR_PRECOMP)

prime_indices = {}
prime_indices_np = {}
for p in ALL_PRIMES:
    indices = [[] for _ in range(p)]
    for x in range(N + 1):
        r = x % p
        indices[r].append(x)
    prime_indices[p] = indices
    prime_indices_np[p] = [np.array(idx, dtype=np.int32) if idx else np.array([], dtype=np.int32) for idx in indices]

def _is_admissible(S_active):
    size = int(S_active.sum())
    for p in ALL_PRIMES:
        if p > size:
            break
        for r in range(p):
            idx = prime_indices_np[p][r]
            if len(idx) == 0 or not S_active[idx].any():
                break
        else:
            return False
    return True

def _greedy_sequential(prime_order, deadline):
    S_active = np.ones(N + 1, dtype=bool)
    current_size = N + 1
    
    for p in prime_order:
        if p > current_size:
            break
        if time.time() > deadline:
            break
        
        counts = np.zeros(p, dtype=np.int32)
        for r in range(p):
            idx = prime_indices_np[p][r]
            if len(idx) > 0:
                counts[r] = int(S_active[idx].sum())
        
        min_count = int(counts.min())
        
        if min_count > 0:
            candidates = [r for r in range(p) if counts[r] == min_count]
            r_to_remove = random.choice(candidates)
            
            for x in prime_indices[p][r_to_remove]:
                if S_active[x]:
                    S_active[x] = False
                    current_size -= 1
    
    return current_size, S_active

def _local_search_add(S_active, current_size, deadline):
    S_active = S_active.copy()
    current_size = int(S_active.sum())
    
    # Track empty residue classes per prime
    prime_empty = {}
    for p in ALL_PRIMES:
        if p > current_size:
            break
        empties = []
        for r in range(p):
            idx = prime_indices_np[p][r]
            if len(idx) == 0 or not S_active[idx].any():
                empties.append(r)
        prime_empty[p] = empties
    
    inactive_mask = ~S_active
    inactive_indices = np.where(inactive_mask)[0]
    
    if len(inactive_indices) == 0:
        return current_size, S_active
    
    np.random.shuffle(inactive_indices)
    
    for x in inactive_indices:
        if time.time() > deadline:
            break
        
        new_size = current_size + 1
        
        feasible = True
        for p in ALL_PRIMES:
            if p > new_size:
                break
            r = x % p
            empties = prime_empty.get(p, [])
            if len(empties) == 1 and r in empties:
                feasible = False
                break
        
        if feasible:
            S_active[x] = True
            current_size = new_size
            for p in ALL_PRIMES:
                if p > current_size:
                    break
                r = x % p
                empties = prime_empty.get(p, [])
                if r in empties:
                    prime_empty[p] = [e for e in empties if e != r]
    
    return current_size, S_active

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = 0
    best_set = []
    
    num_starts = 50
    results = []
    
    for trial in range(num_starts):
        if time.time() > deadline - 120:
            break
        
        primes_to_use = [p for p in ALL_PRIMES if p <= N + 1]
        
        if trial % 5 == 0:
            prime_order = primes_to_use
        elif trial % 5 == 1:
            prime_order = primes_to_use[::-1]
        elif trial % 5 == 2:
            prime_order = sorted(primes_to_use, key=lambda p: -p)
        elif trial % 5 == 3:
            # Smallest primes first (they constrain most)
            prime_order = primes_to_use[:]
        else:
            prime_order = primes_to_use[:]
            random.shuffle(prime_order)
        
        new_size, S_active = _greedy_sequential(prime_order, deadline - 120)
        
        if new_size > best_size:
            best_size = new_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
        
        results.append((new_size, S_active))
    
    results.sort(key=lambda x: -x[0])
    
    local_search_deadline = deadline - 15
    
    for size, S_active in results[:25]:
        if time.time() > local_search_deadline:
            break
        if size < best_size:
            continue
        
        new_size, S_active2 = _local_search_add(S_active, size, local_search_deadline)
        
        if new_size > best_size:
            best_size = new_size
            best_set = [int(x) for x in range(N + 1) if S_active2[x]]
    
    if not best_set:
        best_set = [0, 1]
    
    return sorted(best_set)
