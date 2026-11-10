import time
import random

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 45
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Precompute prime exponent vectors for 2..40
    # vector[i] corresponds to exponent of primes[i]
    vectors = {}
    for n in range(2, 41):
        v = [0] * 12
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = tuple(v)
    
    zero_vec = tuple([0] * 12)
    candidates = list(range(2, 41))
    
    best_set = []
    best_len = 0
    
    num_restarts = 1000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        sums = {zero_vec}
        
        for cand in shuffled:
            v = vectors[cand]
            
            # Check for collision
            # If any s + v is in sums, collision
            # We can break early
            collision = False
            # Optimization: iterate over sums
            for s in sums:
                if (s[0]+v[0], s[1]+v[1], s[2]+v[2], s[3]+v[3], s[4]+v[4], s[5]+v[5],
                    s[6]+v[6], s[7]+v[7], s[8]+v[8], s[9]+v[9], s[10]+v[10], s[11]+v[11]) in sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(cand)
                # Update sums
                # We need to add s+v for all s in sums
                # To avoid modifying while iterating, we can create a list of new sums
                # Or use a set update
                new_sums = set()
                for s in sums:
                    new_sums.add((s[0]+v[0], s[1]+v[1], s[2]+v[2], s[3]+v[3], s[4]+v[4], s[5]+v[5],
                                  s[6]+v[6], s[7]+v[7], s[8]+v[8], s[9]+v[9], s[10]+v[10], s[11]+v[11]))
                sums.update(new_sums)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    if best_len == 18:
                        # Max possible? 18 is the upper bound in prompt.
                        # Let's assume 18 is the target max.
                        # We can break early if we hit 18.
                        return list(best_set)

    # Fallback
    if not best_set:
        return primes
        
    return list(best_set)
