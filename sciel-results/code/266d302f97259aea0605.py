import time
import bisect
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes[:]
    if n == 2:
        return primes[:]
    
    best_seq = primes[:2]
    best_len = 2
    
    for i in range(n):
        pi = primes[i]
        for j in range(i+1, n):
            pj = primes[j]
            g = pj - pi
            
            # Greedily extend from (pi, pj) with initial gap g
            seq = [pi, pj]
            last_gap = g
            last_prime = pj
            
            while True:
                target = last_prime + last_gap
                pos = bisect.bisect_left(primes, target)
                if pos >= n:
                    break
                next_p = primes[pos]
                new_gap = next_p - last_prime
                # new_gap >= last_gap is guaranteed by construction
                seq.append(next_p)
                last_gap = new_gap
                last_prime = next_p
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return best_seq
