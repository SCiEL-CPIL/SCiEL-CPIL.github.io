"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _is_valid_set(s):
    """Check if set s satisfies: for all distinct a,b,c in s, a does not divide b*c."""
    s_list = sorted(s)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if j == i:
                continue
            b = s_list[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _count_violations(s):
    """Count how many (a,b,c) triples violate the condition."""
    s_list = sorted(s)
    n = len(s_list)
    count = 0
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if j == i:
                continue
            b = s_list[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                c = s_list[k]
                if (b * c) % a == 0:
                    count += 1
    return count

def _can_add(s, candidate):
    """Check if adding candidate to set s keeps it valid."""
    if candidate in s:
        return False
    # Check: for all distinct a,b,c in s∪{candidate}, a ∤ b*c
    # New violations can only involve candidate
    s_list = sorted(s)
    # Case 1: candidate is a, so candidate ∤ b*c for all b,c in s, b≠c
    for i in range(len(s_list)):
        b = s_list[i]
        for j in range(i+1, len(s_list)):
            c = s_list[j]
            if (b * c) % candidate == 0:
                return False
    # Case 2: candidate is b or c, so for all a in s, a ∤ candidate * other
    for a in s_list:
        for other in s_list:
            if other == a:
                continue
            if (candidate * other) % a == 0:
                return False
    return True

def _can_remove_and_add(s, remove, add):
    """Check if removing 'remove' and adding 'add' keeps validity."""
    new_s = set(s)
    new_s.discard(remove)
    new_s.add(add)
    return _is_valid_set(new_s)

def construct():
    deadline = time.time() + 80  # safety margin
    
    # Generate primes in 2..256
    def sieve(n):
        is_prime = [True] * (n+1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n)+1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(256)
    current = set(primes)
    best = set(current)
    best_size = len(best)
    
    all_candidates = list(range(2, 257))
    
    # Phase 1: Greedily add composites
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in all_candidates:
            if cand in current:
                continue
            if _can_add(current, cand):
                current.add(cand)
                improved = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
    
    # Phase 2: Local search - try swaps
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        # Try adding any missing element
        for cand in all_candidates:
            if cand in current:
                continue
            if _can_add(current, cand):
                current.add(cand)
                improved = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
                break
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for remove in list(current):
            if time.time() > deadline:
                break
            for add in all_candidates:
                if add in current:
                    continue
                if add == remove:
                    continue
                new_s = set(current)
                new_s.discard(remove)
                new_s.add(add)
                if _is_valid_set(new_s):
                    current = new_s
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(best)
                    break
            if improved:
                break
        
        # Try removing one element and adding two
        if not improved:
            for remove in list(current):
                if time.time() > deadline:
                    break
                for add1 in all_candidates:
                    if add1 in current or add1 == remove:
                        continue
                    new_s = set(current)
                    new_s.discard(remove)
                    new_s.add(add1)
                    for add2 in all_candidates:
                        if add2 in new_s or add2 == remove or add2 == add1:
                            continue
                        new_s2 = set(new_s)
                        new_s2.add(add2)
                        if _is_valid_set(new_s2):
                            current = new_s2
                            improved = True
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(best)
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    # Return the best found
    return list(best)
# EVOLVE-BLOCK-END
