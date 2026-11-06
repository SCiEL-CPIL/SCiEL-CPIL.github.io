import time
import random
from collections import Counter

N = 5000
DEADLINE = 300.0

def _sieve_primes(limit):
    if limit < 2: return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _greedy_construct(primes, n_max=N, randomize=False):
    points = set(range(n_max + 1))
    for p in primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            if randomize and len(min_classes) > 1:
                remove_class = random.choice(min_classes)
            else:
                remove_class = min(min_classes)
            points = {v for v in points if v % p != remove_class}
    return sorted(points)

def _local_search_safe_add(points, primes, n_max=N, deadline=None):
    points_set = set(points)
    best = list(points)
    best_size = len(best)
    
    start_time = time.time()
    
    # We'll try to add elements repeatedly
    # To speed up, we can maintain the missing residues
    # But recalculating is simpler and likely fast enough for N=5000
    
    while True:
        if deadline and time.time() > deadline - 5:
            break
            
        current_size = len(points_set)
        # Get missing residues for primes <= current_size
        missing = {}
        for p in primes:
            if p > current_size:
                break
            res = {x % p for x in points_set}
            missing[p] = set(range(p)) - res
            
        added_any = False
        # Try to add any valid element
        # We can shuffle candidates to find different local optima
        candidates = [x for x in range(n_max + 1) if x not in points_set]
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            # Check existing primes
            for p, miss_set in missing.items():
                if x % p in miss_set:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check new prime p = current_size + 1
            p_new = current_size + 1
            if p_new in primes:
                # Check if adding x makes p_new full
                # Get residues of points_set mod p_new
                res_new = {y % p_new for y in points_set}
                rx = x % p_new
                if rx in res_new:
                    # No new residue added. Check if res_new is already full
                    if len(res_new) == p_new:
                        valid = False
                else:
                    # New residue added. Check if res_new U {rx} is full
                    if len(res_new) + 1 == p_new:
                        valid = False
            
            if valid:
                points_set.add(x)
                added_any = True
                break # Restart to update missing sets
                
        if not added_any:
            break
            
    if len(points_set) > best_size:
        best = sorted(points_set)
        
    return best

def construct():
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    deadline = start_time + DEADLINE
    
    best = []
    best_size = 0
    
    # Try standard greedy
    g1 = _greedy_construct(primes, N, randomize=False)
    if len(g1) > best_size:
        best = g1
        best_size = len(best)
        
    # Local search on best
    if time.time() < deadline - 10:
        ls1 = _local_search_safe_add(best, primes, N, deadline)
        if len(ls1) > best_size:
            best = ls1
            best_size = len(best)
            
    # Try random greedy
    if time.time() < deadline - 20:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            g2 = _greedy_construct(primes, N, randomize=True)
            if len(g2) > best_size:
                best = g2
                best_size = len(best)
            # Local search on g2
            if time.time() < deadline - 5:
                ls2 = _local_search_safe_add(g2, primes, N, deadline)
                if len(ls2) > best_size:
                    best = ls2
                    best_size = len(best)
                    
    return best
