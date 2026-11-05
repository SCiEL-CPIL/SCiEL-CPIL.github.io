import time
import random

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _greedy_construct():
    points = list(range(N + 1))
    primes = _primes(N + 1)
    
    for p in primes:
        if p > len(points):
            break
        
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        
        if min(counts) > 0:
            min_idx = counts.index(min(counts))
            points = [v for v in points if v % p != min_idx]
            
    return points

def construct():
    start_time = time.time()
    
    # 1. Greedy Initial Solution
    points = _greedy_construct()
    in_set = set(points)
    
    # 2. Preparation for Local Search
    primes = _primes(N + 1)
    
    # Maintain counts and missed sets for all primes up to N+1
    counts = {}
    missed = {}
    
    for p in primes:
        counts[p] = [0] * p
        for v in points:
            counts[p][v % p] += 1
        missed[p] = set()
        for r in range(p):
            if counts[p][r] == 0:
                missed[p].add(r)
                
    # 3. Local Search with Multiple Restarts and Different Heuristics
    candidates = [i for i in range(N + 1) if i not in in_set]
    
    current_len = len(points)
    
    # We will try several passes with different random shuffles
    # Also, we can try a "greedy by rarity" heuristic instead of random
    # But let's stick to random with multiple passes for simplicity and robustness
    
    improved = True
    passes = 0
    max_passes = 100  # Safety bound, but we'll break on time or no improvement
    
    while improved and (time.time() - start_time < DEADLINE - 5) and passes < max_passes:
        passes += 1
        improved = False
        
        # Shuffle candidates for this pass
        random.shuffle(candidates)
        
        # Optional: Re-compute candidates if in_set changed significantly?
        # No, we just skip those in in_set.
        
        for cand in candidates:
            if cand in in_set:
                continue
            
            new_len = current_len + 1
            can_add = True
            
            # Check admissibility constraints
            # We only need to check primes p <= new_len
            for p in primes:
                if p > new_len:
                    break
                
                m_size = len(missed[p])
                
                if m_size == 0:
                    # This can only happen if p > current_len (since current set is admissible)
                    # If p == new_len, then adding cand keeps all residues covered -> invalid
                    can_add = False
                    break
                    
                elif m_size == 1:
                    # If cand fills the unique missed residue, it becomes invalid
                    if cand % p in missed[p]:
                        can_add = False
                        break
                # else m_size >= 2: safe
                
            if can_add:
                # Add cand
                points.append(cand)
                in_set.add(cand)
                current_len = new_len
                improved = True
                
                # Update counts and missed
                for p in primes:
                    if p > current_len + 1:
                        break
                    r = cand % p
                    counts[p][r] += 1
                    if r in missed[p]:
                        missed[p].discard(r)
                        
    return points

# EVOLVE-BLOCK-END
