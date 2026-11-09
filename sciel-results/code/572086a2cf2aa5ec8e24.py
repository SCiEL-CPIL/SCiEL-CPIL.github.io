"""Improved program: multi-start greedy with bitmask coprimality + hill-climbing."""
import time
from math import gcd
from itertools import combinations

# Deadline guard
_DEADLINE = 150.0
_START_TIME = None

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > _DEADLINE - 5.0  # 5s safety margin

def _score(chosen):
    """Compute sum 1/(512 - a) for chosen list."""
    return sum(1.0 / (512 - a) for a in chosen)

def _build_coprime_masks():
    """
    For each number a in 1..511, compute a bitmask of which numbers b in 1..511
    are coprime to a. We'll use a list of 511 integers (bit i set means number i+1 is coprime).
    Actually, for efficiency in greedy, we just need to check gcd(a, b)==1 for candidates.
    Let's precompute gcd relationships differently.
    """
    pass

def _greedy_construct(start_order):
    """
    Greedy: iterate through start_order, add a if coprime to all chosen so far.
    start_order is a list of numbers 1..511 in some order.
    """
    chosen = []
    chosen_set = set()
    for a in start_order:
        if a in chosen_set:
            continue
        # Check coprime with all chosen
        ok = True
        for b in chosen:
            if gcd(a, b) != 1:
                ok = False
                break
        if ok:
            chosen.append(a)
            chosen_set.add(a)
    return chosen

def _hill_climb(chosen, max_iters=200000):
    """
    Hill-climbing: try to replace one element with another to improve score,
    or add a new element if possible.
    """
    chosen = list(chosen)
    chosen_set = set(chosen)
    current_score = _score(chosen)
    n = len(chosen)
    
    # Precompute gcd with all numbers for speed
    # Actually, let's just do it on the fly but efficiently
    
    for _ in range(max_iters):
        if _deadline_exceeded():
            break
        
        improved = False
        
        # Try adding a new number
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            ok = True
            for b in chosen:
                if gcd(a, b) != 1:
                    ok = False
                    break
            if ok:
                new_score = current_score + 1.0 / (512 - a)
                if new_score > current_score + 1e-15:
                    chosen.append(a)
                    chosen_set.add(a)
                    current_score = new_score
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try replacing one element
        for i in range(n):
            old = chosen[i]
            # Try replacing with a larger number (higher contribution)
            for a in range(511, 0, -1):
                if a == old or a in chosen_set:
                    continue
                # Check if a is coprime to all other chosen elements
                ok = True
                for j, b in enumerate(chosen):
                    if j == i:
                        continue
                    if gcd(a, b) != 1:
                        ok = False
                        break
                if ok:
                    delta = 1.0 / (512 - a) - 1.0 / (512 - old)
                    if delta > 1e-15:
                        chosen[i] = a
                        chosen_set.discard(old)
                        chosen_set.add(a)
                        current_score += delta
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return chosen

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Strategy: multi-start greedy with different orderings, then hill-climb.
    # Best found so far
    best_chosen = []
    best_score = -1.0
    
    # Ordering 1: descending (prefer large a, i.e., small 512-a)
    order1 = list(range(511, 0, -1))
    # Ordering 2: ascending
    order2 = list(range(1, 512))
    # Ordering 3: primes first (they're coprime to many things)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [p for p in range(2, 512) if is_prime(p)]
    non_primes = [x for x in range(511, 0, -1) if x not in set(primes)]
    order3 = primes + non_primes
    
    # Ordering 4: numbers with small 512-a that are odd (coprime to 2)
    order4 = list(range(511, 0, -1, 2))  # odd numbers descending
    
    # Ordering 5: numbers coprime to 6 (i.e., not divisible by 2 or 3), descending
    order5 = [x for x in range(511, 0, -1) if x % 2 != 0 and x % 3 != 0]
    
    orderings = [order1, order2, order3, order4, order5]
    
    for order in orderings:
        if _deadline_exceeded():
            break
        
        chosen = _greedy_construct(order)
        chosen = _hill_climb(chosen)
        s = _score(chosen)
        
        if s > best_score:
            best_score = s
            best_chosen = list(chosen)
    
    # Additional: try hill-climbing from a random-ish start
    import random
    random.seed(42)
    for _ in range(10):
        if _deadline_exceeded():
            break
        random_order = list(range(1, 512))
        random.shuffle(random_order)
        chosen = _greedy_construct(random_order)
        chosen = _hill_climb(chosen)
        s = _score(chosen)
        if s > best_score:
            best_score = s
            best_chosen = list(chosen)
    
    return best_chosen
