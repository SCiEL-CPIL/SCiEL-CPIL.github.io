import time
import random

# Precompute
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
prime_to_idx = {p: i for i, p in enumerate(primes)}
num_primes = len(primes)

def get_vector(n):
    v = [0] * num_primes
    for i, p in enumerate(primes):
        while n % p == 0:
            v[i] += 1
            n //= p
    return tuple(v)

vectors = {}
for n in range(2, 41):
    vectors[n] = get_vector(n)

def is_valid(S):
    k = len(S)
    if k == 0:
        return True
    
    sums = set()
    sums.add(tuple([0]*num_primes))
    
    for x in S:
        v = vectors[x]
        new_sums = set()
        for s in sums:
            new_s = tuple(s[i] + v[i] for i in range(num_primes))
            if new_s in sums or new_s in new_sums:
                return False
            new_sums.add(new_s)
        sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    base = list(primes)
    composites = [n for n in range(2, 41) if n not in base]
    all_numbers = list(range(2, 41))
    
    best = list(base)
    
    # Greedy
    for c in composites:
        if is_valid(best + [c]):
            best.append(c)
            
    # Local Search
    def local_search(s):
        s = list(s)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return s
            # Try add
            for c in all_numbers:
                if c not in s:
                    if is_valid(s + [c]):
                        s.append(c)
                        improved = True
                        break
            if not improved:
                # Try swap: remove i, add j, k
                found = False
                for i in range(len(s)):
                    if found: break
                    s_wo_i = [x for j, x in enumerate(s) if j != i]
                    # Try add 2
                    for c1 in all_numbers:
                        if c1 in s: continue
                        for c2 in all_numbers:
                            if c2 in s or c2 <= c1: continue
                            cand = s_wo_i + [c1, c2]
                            if is_valid(cand):
                                s = cand
                                improved = True
                                found = True
                                break
        return s

    best = local_search(best)
    
    # Random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(best) >= 18:
            break
        random.shuffle(composites)
        trial = list(base)
        for c in composites:
            if is_valid(trial + [c]):
                trial.append(c)
        trial = local_search(trial)
        if len(trial) > len(best):
            best = trial

    return [int(x) for x in best]
