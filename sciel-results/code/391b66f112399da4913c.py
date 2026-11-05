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

def _find_violating_prime(points, primes_list):
    """Find smallest prime p <= len(points) where all residues are covered."""
    k = len(points)
    if k == 0:
        return None
    for p in primes_list:
        if p > k:
            break
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return p
    return None

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    primes = _primes(N)
    
    best_set = None
    best_size = 0
    
    # Strategy: Start with full set, greedily remove ONE element per violating prime
    # Process primes in increasing order for efficiency
    
    # Multiple random restarts with different removal orderings
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        current_set = set(range(N))
        
        # Greedy repair: for each violating prime, remove one element
        # from the least-populated residue class
        max_iterations = 1000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            k = len(current_set)
            if k == 0:
                break
            
            # Find a violating prime (smallest one first for efficiency)
            violating_p = _find_violating_prime(list(current_set), primes)
            
            if violating_p is None:
                # Admissible!
                break
            
            # Find the residue class mod violating_p with fewest elements
            # Remove ONE element from that class
            counts = [0] * violating_p
            for v in current_set:
                counts[v % violating_p] += 1
            
            # Find minimum count residue
            min_count = min(counts)
            min_residues = [r for r in range(violating_p) if counts[r] == min_count]
            
            # Randomly pick from min residues for diversity
            chosen_residue = random.choice(min_residues)
            
            # Remove one element with that residue
            # Pick randomly among elements with that residue for diversity
            candidates = [v for v in current_set if v % violating_p == chosen_residue]
            if candidates:
                # Remove a random one (or the one that helps most)
                # For now, just remove one
                to_remove = random.choice(candidates)
                current_set.remove(to_remove)
        
        current_size = len(current_set)
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        
        # Try to add back elements
        if best_set is not None:
            candidates = [x for x in range(N) if x not in best_set]
            random.shuffle(candidates)
            
            # Limit checks to avoid timeout
            check_limit = min(len(candidates), 500)
            added_any = True
            passes = 0
            while added_any and passes < 5:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                added_any = False
                passes += 1
                for x in candidates[:check_limit]:
                    if time.time() - start_time > TIME_LIMIT - 3:
                        break
                    new_size = len(best_set) + 1
                    is_adm = True
                    for p in primes:
                        if p > new_size:
                            break
                        seen = bytearray(p)
                        count = 0
                        for v in best_set:
                            r = v % p
                            if not seen[r]:
                                seen[r] = 1
                                count += 1
                                if count == p:
                                    break
                        rx = x % p
                        if count == p:
                            is_adm = False
                            break
                        if not seen[rx]:
                            count += 1
                            if count == p:
                                is_adm = False
                                break
                    if is_adm:
                        best_set.add(x)
                        best_size = len(best_set)
                        added_any = True
                        candidates.remove(x)
                        break
            
            # Update best_set reference
            if best_set is not None:
                best_set = set(best_set)
    
    if best_set is None:
        return list(range(N))
    
    return list(best_set)

# Warm up
construct()
