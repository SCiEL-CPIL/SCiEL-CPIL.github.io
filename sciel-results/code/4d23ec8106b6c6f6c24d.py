import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute primes for heuristic
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [x for x in candidates if is_prime[x]]
    
    # Strategy 1: All Primes
    best_set = primes[:]
    
    def is_valid_addition(a, current_set):
        if len(current_set) < 2:
            return True
        cs = current_set
        n = len(cs)
        for i in range(n):
            b = cs[i]
            for j in range(i + 1, n):
                c = cs[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True

    def greedy_search(order):
        s = []
        for a in order:
            if is_valid_addition(a, s):
                s.append(a)
        return s

    # Run greedy with different orders
    # 1. Primes first, then composites
    comp_order = sorted([x for x in candidates if not is_prime[x]], reverse=True)
    order1 = primes + comp_order
    s1 = greedy_search(order1)
    if len(s1) > len(best_set):
        best_set = s1

    # 2. Random shuffles
    for _ in range(50):
        if time.time() > deadline:
            break
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s

    # Local Search on best_set
    # Try to add any remaining element
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current = best_set[:]
        set_current = set(current)
        for v in candidates:
            if v not in set_current:
                if is_valid_addition(v, current):
                    current.append(v)
                    set_current.add(v)
                    changed = True
                    break # Restart loop with new set
        if changed:
            best_set = current

    # Try 1-for-2 swaps (remove 1, add 2)
    # This is expensive, so we do limited iterations
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        current = best_set[:]
        set_current = set(current)
        # To speed up, we might just try random swaps
        # But we want deterministic improvement.
        # Given the size, we can iterate all u in S and pairs (v,w) in complement
        # But that's O(|S|^2 * N^2).
        # |S| ~ 50. N ~ 250. 50 * 60000 = 3M checks.
        # Each check is O(|S|^2) = 2500. Total 7.5B. Too slow.
        
        # Skip complex local search if too slow. 
        # The greedy with primes first is likely strong.
        
    return best_set
