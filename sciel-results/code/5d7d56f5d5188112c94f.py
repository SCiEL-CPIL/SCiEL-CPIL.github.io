import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

PRIMES = _primes(N)
PRIME_SET = set(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}
NUM_PRIMES = len(PRIMES)

# Precompute members globally
MEMBERS = []
for i, p in enumerate(PRIMES):
    res_members = [[] for _ in range(p)]
    for x in range(N):
        res_members[x % p].append(x)
    MEMBERS.append(res_members)

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time

    best_size = 0
    best_set = set()

    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break

        in_set = bytearray(b'\x01') * N
        current_set = set(range(N))
        k = N
        
        counts = []
        num_empty = []
        for i, p in enumerate(PRIMES):
            c = [0] * p
            for x in range(N):
                c[x % p] += 1
            counts.append(c)
            num_empty.append(sum(1 for val in c if val == 0))
            
        # Greedy Repair
        while True:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            violating_p_idx = -1
            for i, p in enumerate(PRIMES):
                if p > k:
                    break
                if num_empty[i] == 0:
                    violating_p_idx = i
                    break
            
            if violating_p_idx == -1:
                break
                
            p = PRIMES[violating_p_idx]
            min_count = min(counts[violating_p_idx])
            candidates_r = [r for r in range(p) if counts[violating_p_idx][r] == min_count]
            r = random.choice(candidates_r)
            
            found = False
            for x in MEMBERS[violating_p_idx][r]:
                if in_set[x]:
                    in_set[x] = 0
                    current_set.remove(x)
                    k -= 1
                    found = True
                    for j, pj in enumerate(PRIMES):
                        rj = x % pj
                        if counts[j][rj] == 1:
                            num_empty[j] += 1
                        counts[j][rj] -= 1
                    break
            if not found:
                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # Local Search
        candidates = [x for x in range(N) if not in_set[x]]
        random.shuffle(candidates)
        
        added_any = True
        while added_any:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            added_any = False
            for x in candidates:
                if in_set[x]:
                    continue
                    
                new_k = k + 1
                can_add = True
                
                for i, p in enumerate(PRIMES):
                    if p > k:
                        break
                    r_x = x % p
                    if counts[i][r_x] == 0:
                        if num_empty[i] == 1:
                            can_add = False
                            break
                
                if not can_add:
                    continue
                
                if new_k in PRIME_SET:
                    idx_new = PRIME_TO_IDX[new_k]
                    r_x_new = x % new_k
                    if counts[idx_new][r_x_new] == 0:
                        if num_empty[idx_new] <= 1:
                            can_add = False
                    else:
                        if num_empty[idx_new] == 0:
                            can_add = False
                            
                if can_add:
                    in_set[x] = 1
                    current_set.add(x)
                    k += 1
                    added_any = True
                    for j, pj in enumerate(PRIMES):
                        rj = x % pj
                        if counts[j][rj] == 0:
                            num_empty[j] -= 1
                        counts[j][rj] += 1
                    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

    return list(best_set)

construct()
