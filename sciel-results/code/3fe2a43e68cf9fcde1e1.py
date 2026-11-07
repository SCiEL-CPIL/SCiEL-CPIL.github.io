import time
import random

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    # Precompute vectors for numbers 2 to 40
    vectors = {}
    for n in range(2, 41):
        vectors[n] = get_vector(n)
        
    candidates = list(range(2, 41))
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    global_best = []
    
    # We try to find sets of size 18 down to 12
    # 18 is the theoretical max mentioned in prompt range
    for target_size in range(18, 11, -1):
        # If we already have a set of this size or larger, we don't need to search for smaller
        if len(global_best) >= target_size:
            continue
            
        found_target = False
        # Run random restarts
        while time.time() < deadline:
            if found_target:
                break
                
            random.shuffle(candidates)
            current_set = []
            sums = {tuple(0 for _ in range(num_primes))}
            
            for cand in candidates:
                if len(current_set) >= target_size:
                    break
                    
                v = vectors[cand]
                
                # Generate new sums
                new_sums = set()
                for s in sums:
                    ns = tuple(s[i] + v[i] for i in range(num_primes))
                    new_sums.add(ns)
                    
                # Check for collision
                if sums & new_sums:
                    continue
                    
                # No collision, add candidate
                current_set.append(cand)
                sums = sums.union(new_sums)
                
            if len(current_set) > len(global_best):
                global_best = current_set[:]
                
            if len(current_set) == target_size:
                found_target = True
                # If we found the maximum possible (18), we can stop everything
                if target_size == 18:
                    return [int(x) for x in global_best]
                    
    return [int(x) for x in global_best]
