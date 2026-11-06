"""Rewritten program for powerful number sum representations."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin
    
    # Generate powerful numbers up to 10^10
    limit_gen = 10**10
    
    # Sieve primes up to 10^5 (since p^2 <= 10^10 => p <= 10^5)
    sieve_lim = 10**5
    sieve = [True] * (sieve_lim + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(sieve_lim) + 1):
        if sieve[i]:
            for j in range(i*i, sieve_lim+1, i):
                sieve[j] = False
    primes = [i for i in range(2, sieve_lim+1) if sieve[i]]
    
    # Generate powerful numbers
    pw_list = [1]
    for p in primes:
        p2 = p * p
        if p2 > limit_gen:
            break
        new_entries = []
        for x in pw_list:
            val = x
            while val * p2 <= limit_gen:
                val *= p2
                new_entries.append(val)
        pw_list.extend(new_entries)
    
    powerful_set = set(pw_list)
    powerful_sorted = sorted(powerful_set)
    n_pw = len(powerful_sorted)
    
    # Count sums using a dictionary
    sum_count = {}
    for i in range(n_pw):
        a = powerful_sorted[i]
        for j in range(i, n_pw):
            b = powerful_sorted[j]
            s = a + b
            if s > 10**18:
                break
            if s in sum_count:
                sum_count[s].append((a, b))
            else:
                sum_count[s] = [(a, b)]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Find the N with the most pairs
    best_n = 2
    best_pairs = []
    for n, pairs in sum_count.items():
        if len(pairs) > len(best_pairs):
            best_n = n
            best_pairs = pairs
    
    if not best_pairs:
        best_n = 2
        best_pairs = [(1, 1)]
    
    return {'N': best_n, 'pairs': [[a, b] for a, b in best_pairs]}
# EVOLVE-BLOCK-END
