import time
import random
import math

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

def construct():
    start_time = time.time()
    TIME_LIMIT = 290
    
    all_primes = _primes(N + 1)
    
    best_size = 0
    best_set = []
    
    # Run multiple randomized greedy passes
    # We can run many because each pass is fast
    for run in range(2000):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        prime_strategy = random.choice(['smallest', 'largest', 'random'])
        residue_strategy = random.choice(['min_count', 'random_min'])
        
        active = [True] * (N + 1)
        L = N + 1
        
        res_counts = {}
        for p in all_primes:
            if p > L: break
            counts = [0] * p
            for v in range(N + 1):
                counts[v % p] += 1
            res_counts[p] = counts
            
        violated = set()
        for p in all_primes:
            if p > L: break
            if min(res_counts[p]) > 0:
                violated.add(p)
                
        while violated:
            if time.time() - start_time > TIME_LIMIT:
                break
                
            if prime_strategy == 'smallest':
                p = min(violated)
            elif prime_strategy == 'largest':
                p = max(violated)
            else:
                p = random.choice(list(violated))
                
            counts = res_counts[p]
            min_c = min(counts)
            
            if residue_strategy == 'min_count':
                candidates = [r for r in range(p) if counts[r] == min_c]
                r = candidates[0]
            else:
                candidates = [r for r in range(p) if counts[r] == min_c]
                r = random.choice(candidates)
                
            to_remove = [v for v in range(N + 1) if active[v] and v % p == r]
            
            if not to_remove:
                violated.discard(p)
                continue
                
            for v in to_remove:
                active[v] = False
                L -= 1
                for q in all_primes:
                    if q > L: continue
                    rq = v % q
                    res_counts[q][rq] -= 1
                    if res_counts[q][rq] == 0:
                        violated.discard(q)
                        
            violated.discard(p)
            
        current_size = L
        if current_size > best_size:
            best_size = current_size
            best_set = [v for v in range(N + 1) if active[v]]
            
    # Add-Back Phase on best_set
    if best_set:
        # Initialize counts for best_set
        L = len(best_set)
        active = [False] * (N + 1)
        for v in best_set:
            active[v] = True
            
        res_counts = {}
        hole_counts = {}
        for p in all_primes:
            if p > L: break
            counts = [0] * p
            for v in best_set:
                counts[v % p] += 1
            res_counts[p] = counts
            hole_counts[p] = counts.count(0)
            
        violated = set()
        for p in all_primes:
            if p > L: break
            if hole_counts[p] == 0:
                violated.add(p)
                
        # Candidates to add back
        candidates = [v for v in range(N + 1) if not active[v]]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > TIME_LIMIT:
                break
                
            # Check if adding x is valid
            new_L = L + 1
            valid = True
            
            # Check primes p <= new_L
            for p in all_primes:
                if p > new_L: break
                r = x % p
                # If this residue was a hole (count 0), it becomes non-hole
                # We need to check if this causes any prime to become violated
                # A prime becomes violated if hole_count becomes 0
                
                # We simulate the change
                # If res_counts[p][r] == 0, hole_count decreases by 1
                # If res_counts[p][r] > 0, hole_count unchanged
                
                # But we also need to consider NEW primes p where L < p <= new_L
                # For these, we need to check if all residues are covered by S U {x}
                
                if res_counts[p][r] == 0:
                    # This hole is filled
                    # Check if hole_count becomes 0
                    if hole_counts[p] == 1:
                        # If this was the only hole, it becomes violated
                        valid = False
                        break
                else:
                    # Hole count unchanged, so if it was > 0, it stays > 0
                    # If it was 0, it was already violated (which shouldn't happen for valid set)
                    # But wait, if hole_counts[p] == 0, it's already violated.
                    # Our set is valid, so hole_counts[p] > 0 for all p <= L.
                    pass
                    
            if valid:
                # Add x
                active[x] = True
                L = new_L
                for p in all_primes:
                    if p > L: break
                    r = x % p
                    res_counts[p][r] += 1
                    if res_counts[p][r] == 1:
                        hole_counts[p] -= 1
                # Update best_set
                best_set.append(x)
                best_size = L
                
    return best_set
