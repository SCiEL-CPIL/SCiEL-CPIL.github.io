# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _squarefree_signature(n):
    """Return frozenset of primes with odd exponent in n."""
    sig = set()
    d = 2
    while d * d <= n:
        cnt = 0
        while n % d == 0:
            n //= d
            cnt ^= 1
        if cnt:
            sig.add(d)
        d += 1
    if n > 1:
        sig.add(n)
    return frozenset(sig)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10210
    MAX_SIZE = 100
    N = 210
    
    sig_210 = frozenset({2, 3, 5, 7})
    
    best_result = None
    best_max = MAX_VAL + 1
    
    # Precompute signatures for all numbers in (210, MAX_VAL]
    sigs = [None] * (MAX_VAL + 1)
    for i in range(211, MAX_VAL + 1):
        sigs[i] = _squarefree_signature(i)
    
    # Build lookup: signature -> list of numbers with that signature
    sig_to_nums = defaultdict(list)
    for i in range(211, MAX_VAL + 1):
        sig_to_nums[sigs[i]].append(i)
    
    # Search for pairs: sig(a) XOR sig(b) XOR sig_210 = empty
    # i.e., sig(a) XOR sig(b) = sig_210
    # For each M (as the max), need a < M with sigs[a] = sigs[M] XOR sig_210
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        # Single element: sig(M) XOR sig_210 = empty => sig(M) = sig_210
        if sigs[M] == sig_210:
            if M < best_max:
                best_max = M
                best_result = [M]
            continue
        
        # Pair: need a with sigs[a] = sigs[M] XOR sig_210, 210 < a < M
        target_sig = sigs[M] ^ sig_210
        if target_sig in sig_to_nums:
            for a in sig_to_nums[target_sig]:
                if a < M:
                    if M < best_max:
                        best_max = M
                        best_result = [a, M]
                    break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Search for triples: sig(a) XOR sig(b) XOR sig(c) = sig_210
    # For each M (as max), need a < b < M with sigs[a] XOR sigs[b] = sig_210 XOR sigs[M]
    for M in range(211, 500):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        target = sig_210 ^ sigs[M]
        
        seen = {}
        for a in range(211, M):
            if time.time() > deadline:
                break
            needed = sigs[a] ^ target
            if needed in seen:
                b = seen[needed]
                if b != a:
                    if M < best_max:
                        best_max = M
                        best_result = [a, b, M]
                    break
            else:
                seen[sigs[a]] = a
        
        if best_result is not None and best_max == M:
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback
    return [840]

# EVOLVE-BLOCK-END
