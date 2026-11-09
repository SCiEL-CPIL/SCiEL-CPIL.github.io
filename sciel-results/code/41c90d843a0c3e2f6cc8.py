import numpy as np
import time
import random
import math

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    primes = get_primes(1000)
    L_limit = 100000
    
    # Generate candidate prime sets
    def gen_greedy_set(skip_set=None, max_primes=100):
        s = []
        rem = 1.0
        for p in primes:
            if skip_set and p in skip_set:
                continue
            if 1.0 / p <= rem + 1e-9:
                s.append(p)
                rem -= 1.0 / p
                if len(s) >= max_primes:
                    break
        return s

    sets_to_try = []
    
    # Set 1: Greedy smallest primes
    sets_to_try.append(gen_greedy_set())
    
    # Set 2: Greedy smallest primes, skip 2
    sets_to_try.append(gen_greedy_set(skip_set={2}))
    
    # Set 3: Greedy smallest primes, skip 3
    sets_to_try.append(gen_greedy_set(skip_set={3}))
    
    # Set 4: Greedy smallest primes, skip 2 and 3
    sets_to_try.append(gen_greedy_set(skip_set={2, 3}))
    
    # Set 5: Seed-like (primes >= 13)
    s_seed = [p for p in primes if p >= 13]
    # Trim if necessary (though sum of 1/p for p>=13 is < 1)
    # 1/13 + 1/17 + ... is small.
    sets_to_try.append(s_seed)
    
    # Set 6: Mixed - small primes + some medium
    # Just take first 10 primes, then fill with larger
    s_mixed = primes[:10]
    rem = 1.0 - sum(1.0/p for p in s_mixed)
    for p in primes:
        if p in s_mixed: continue
        if 1.0/p <= rem:
            s_mixed.append(p)
            rem -= 1.0/p
    sets_to_try.append(s_mixed)
    
    # Deduplicate
    unique_sets = []
    seen = set()
    for s in sets_to_try:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
            
    best_L = 0
    best_config = None
    
    # Pre-allocate covered array for reuse? 
    # Since we are changing residues, we have to rebuild or update.
    # Rebuilding is O(L) in numpy, which is fast.
    
    for p_set in unique_sets:
        if not p_set:
            continue
            
        k = len(p_set)
        current_res = [random.randint(0, p - 1) for p in p_set]
        
        def eval_L(res_list):
            covered = np.zeros(L_limit + 1, dtype=bool)
            for i, p in enumerate(p_set):
                r = res_list[i]
                if r == 0:
                    covered[p::p] = True
                else:
                    covered[r::p] = True
            # Find first False in covered[1:]
            # We want the largest L such that covered[1..L] is all True.
            # ~covered[1:] is True where not covered.
            not_covered = ~covered[1:]
            if not np.any(not_covered):
                return L_limit
            first_gap = np.argmax(not_covered)
            return first_gap
        
        current_L = eval_L(current_res)
        if current_L > best_L:
            best_L = current_L
            best_config = (p_set, current_res[:])
            
        # Simulated Annealing
        # Temperature decay
        T = 1.0
        cool_rate = 0.999
        min_T = 1e-4
        
        stuck_count = 0
        max_stuck = 1000
        restarts = 0
        max_restarts = 5
        
        while time.time() < deadline:
            # Check for restart
            if stuck_count > max_stuck:
                if restarts >= max_restarts:
                    break
                restarts += 1
                current_res = [random.randint(0, p - 1) for p in p_set]
                current_L = eval_L(current_res)
                if current_L > best_L:
                    best_L = current_L
                    best_config = (p_set, current_res[:])
                stuck_count = 0
                T = 1.0 # Reset temperature
                continue
                
            # Pick random index
            i = random.randint(0, k - 1)
            p = p_set[i]
            old_r = current_res[i]
            
            # Pick new residue
            # Local move or random jump
            if random.random() < 0.8:
                new_r = (old_r + random.choice([-1, 1])) % p
            else:
                new_r = random.randint(0, p - 1)
                
            if new_r == old_r:
                continue
                
            current_res[i] = new_r
            new_L = eval_L(current_res)
            
            delta = new_L - current_L
            
            if delta >= 0:
                current_L = new_L
                stuck_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_config = (p_set, current_res[:])
                    if best_L >= L_limit:
                        break
            else:
                # Accept worse move with probability
                # If delta is small, accept. If large, reject.
                # Using temperature
                if T > min_T:
                    # P = exp(delta / T) -- wait, delta is negative.
                    # P = exp(delta / T)
                    # If delta is very negative, P is small.
                    # We want to accept if random() < P
                    try:
                        prob = math.exp(delta / T)
                    except OverflowError:
                        prob = 0.0
                    if random.random() < prob:
                        current_L = new_L
                        stuck_count = 0
                    else:
                        # Revert
                        current_res[i] = old_r
                        stuck_count += 1
                else:
                    # Revert
                    current_res[i] = old_r
                    stuck_count += 1
                    
            T *= cool_rate
            
            # If we hit the cap, break
            if best_L >= L_limit:
                break
                
            # Time check
            if time.time() > deadline:
                break
                
        if time.time() > deadline:
            break

    if best_config is None:
        # Fallback to seed
        primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    p_set, res_list = best_config
    return [[p, r] for p, r in zip(p_set, res_list)]
