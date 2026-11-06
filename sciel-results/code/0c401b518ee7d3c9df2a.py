"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import gcd

def _is_bad_triple(a, b, c):
    """Check if three elements have all pairwise gcds equal."""
    return gcd(a, b) == gcd(a, c) == gcd(b, c)

def _can_add(selected, x):
    """Check if x can be added to selected without creating a bad triple."""
    for a, b in combinations(selected, 2):
        if _is_bad_triple(a, b, x):
            return False
    return True

def _greedy(order):
    """Greedy construction with a given order."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_add(selected):
    """Try to add any element that fits. Repeat until no more can be added."""
    selected = list(selected)
    changed = True
    while changed:
        changed = False
        for x in range(1, 129):
            if x not in selected and _can_add(selected, x):
                selected.append(x)
                changed = True
    return selected

def _num_divisors(n):
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 2
            if i * i == n:
                count -= 1
    return count

def construct():
    deadline = time.time() + 80.0
    best = []
    
    # Precompute divisor counts
    div_counts = {n: _num_divisors(n) for n in range(1, 129)}
    
    # Multiple greedy orderings
    orders = [
        list(range(1, 129)),                          # natural
        list(range(128, 0, -1)),                      # reverse
        sorted(range(1, 129), key=lambda x: div_counts[x]),           # few divisors first
        sorted(range(1, 129), key=lambda x: div_counts[x], reverse=True),  # many divisors first
        sorted(range(1, 129), key=lambda x: (div_counts[x], x)),      # divisors then value
        sorted(range(1, 129), key=lambda x: x),                       # same as natural
    ]
    
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        s = _greedy(order)
        if len(s) > len(best):
            best = s
    
    # Local search: try to add more elements
    if time.time() < deadline - 5.0:
        best = _local_add(best)
    
    # Try a few more greedy + local if time permits
    if time.time() < deadline - 10.0:
        # Random-ish orderings
        import random
        rng = random.Random(42)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, 129))
            rng.shuffle(order)
            s = _greedy(order)
            s = _local_add(s)
            if len(s) > len(best):
                best = s
    
    return best
# EVOLVE-BLOCK-END
