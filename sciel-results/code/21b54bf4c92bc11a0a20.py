"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _has_bad_triple(selected):
    """Check if selected has three distinct elements with equal pairwise lcms."""
    n = len(selected)
    if n < 3:
        return False
    # For each pair (i,j), compute lcm, then check if any k forms the triple
    # More efficient: for each triple check
    # But we can be smarter: group by lcm values
    # For a triple (a,b,c) to be bad: lcm(a,b)=lcm(a,c)=lcm(b,c)
    # This means all three lcms are equal to some L
    # So a,b,c all divide L, and their combined lcm is L
    # Key: if we pick elements with distinct "prime power signatures" we avoid this
    
    # Brute force for small sets is fine
    for i in range(n):
        for j in range(i+1, n):
            ab = _lcm(selected[i], selected[j])
            for k in range(j+1, n):
                ac = _lcm(selected[i], selected[k])
                if ac != ab:
                    continue
                bc = _lcm(selected[j], selected[k])
                if bc == ab:
                    return True
    return False

def _can_add(selected, x):
    """Check if adding x to selected creates a bad triple."""
    # A bad triple involving x: (x, a, b) where a,b in selected
    # lcm(x,a) == lcm(x,b) == lcm(a,b)
    n = len(selected)
    for i in range(n):
        xa = _lcm(x, selected[i])
        for j in range(i+1, n):
            xb = _lcm(x, selected[j])
            if xa != xb:
                continue
            ab = _lcm(selected[i], selected[j])
            if ab == xa:
                return False
    return True

def _greedy_build(order):
    """Build set greedily following given order."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _try_improve(selected):
    """Try to improve by removing one element and adding others."""
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        # Try removing each element and see if we can add more
        for i in range(len(selected)):
            test = selected[:i] + selected[i+1:]
            # Try to add elements not in selected
            added = False
            for x in range(1, 129):
                if x in test:
                    continue
                if _can_add(test, x):
                    test.append(x)
                    added = True
            if len(test) > len(selected):
                selected = test
                improved = True
                break
    return selected

def _local_search(selected):
    """Local search: try swaps and additions."""
    selected = list(selected)
    n = len(selected)
    
    # First, try to add more elements
    changed = True
    while changed:
        changed = False
        for x in range(1, 129):
            if x not in selected:
                if _can_add(selected, x):
                    selected.append(x)
                    changed = True
    
    # Try removing pairs and adding different elements
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            test = selected[:i] + selected[i+1:j] + selected[j+1:]
            # Try to add elements
            added_count = 0
            for x in range(1, 129):
                if x not in test:
                    if _can_add(test, x):
                        test.append(x)
                        added_count += 1
            if len(test) > len(selected):
                selected = test
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Strategy: elements with unique prime power structure are safer
    # Powers of 2: 1,2,4,8,16,32,64,128 - these have distinct 2-adic valuations
    # But lcm(2,4)=4, lcm(2,8)=8, lcm(4,8)=8 - so (2,4,8): lcm(2,4)=4, lcm(2,8)=8 != 4
    # Actually let me think more carefully
    
    # Key observation: if all elements are powers of distinct primes or have
    # very different prime factorizations, we're safer.
    
    # Let's try: include all primes, all prime powers, and carefully selected composites
    
    # Greedy approach with multiple orderings
    base_elements = list(range(1, 129))
    
    # Try different orderings
    orderings = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        # Primes first
        [x for x in range(1, 129) if all(x % p != 0 for p in range(2, int(x**0.5)+1))] + 
        [x for x in range(1, 129) if not all(x % p != 0 for p in range(2, int(x**0.5)+1))],
        # Even then odd
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        # Odd then even
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        # By number of distinct prime factors (fewer first)
        sorted(range(1, 129), key=lambda x: sum(1 for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113] if x % p == 0)),
    ]
    
    best_set = []
    for order in orderings:
        if time.time() > deadline:
            break
        s = _greedy_build(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search improvement
    if time.time() < deadline:
        best_set = _local_search(best_set)
    
    if time.time() < deadline:
        best_set = _try_improve(best_set)
    
    # Verify no bad triples
    if _has_bad_triple(best_set):
        # Fallback: just use powers of 2 and primes
        fallback = [1, 2, 4, 8, 16, 32, 64, 128]
        for x in range(3, 129):
            if all(x % p != 0 for p in range(2, int(x**0.5)+1)):
                fallback.append(x)
        best_set = fallback
    
    if not best_set:
        best_set = [1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
