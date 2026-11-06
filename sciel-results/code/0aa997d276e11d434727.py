"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 192
MAX_K = 48
DEADLINE = 80.0  # safety margin before 90s kill

def _valid_set(S):
    """Check if set S is valid: no element divides sum of any 3 distinct other elements."""
    n = len(S)
    for i in range(n):
        a = S[i]
        # Check all triples from S excluding a
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                for l in range(k + 1, n):
                    if l == i:
                        continue
                    if (S[j] + S[k] + S[l]) % a == 0:
                        return False
    return True

def _can_add_fast(S, x):
    """Check if x can be added to S without violating constraints.
    Only need to check:
    1. For each a in S: does any triple involving x (and two others from S) sum to multiple of a?
       Also: does any triple of three elements from S (not involving x) sum to multiple of x?
    2. For x itself: does any triple from S sum to multiple of x?
    
    Actually, the constraint is: for every element a in S∪{x}, no triple of 3 DISTINCT OTHER elements sums to multiple of a.
    
    When adding x to S:
    - For existing a in S: new triples are those that include x. So triples {x, b, c} where b,c in S, b!=c, b!=a, c!=a.
      We need: (x + b + c) % a != 0 for all such b,c.
      Also existing triples (not involving x) were already valid.
    - For x: triples of 3 elements from S (all distinct from x, which they are) must not sum to multiple of x.
      So: for all triples {b,c,d} from S, (b+c+d) % x != 0.
    """
    n = len(S)
    # Check for x: all triples from S must not sum to multiple of x
    if n >= 3:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (S[i] + S[j] + S[k]) % x == 0:
                        return False
    # Check for each a in S: triples involving x and two others from S
    for i in range(n):
        a = S[i]
        # Triples {x, S[j], S[k]} where j != k, j != i, k != i
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                if (x + S[j] + S[k]) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Greedy: try to add elements in given order."""
    S = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(S, x):
            S.append(x)
    return S

def _local_search(S, deadline):
    """Try to improve S by swapping elements or adding new ones."""
    S = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any missing element
        for x in range(1, N + 1):
            if x not in S and _can_add_fast(S, x):
                S.append(x)
                improved = True
                break
        # Try swaps: remove one, add one
        if not improved:
            for i in range(len(S)):
                if time.time() > deadline:
                    return S
                old = S[i]
                for x in range(1, N + 1):
                    if x == old or x in S:
                        continue
                    S_new = S[:i] + S[i+1:]
                    if _can_add_fast(S_new, x):
                        S_new.append(x)
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multiple random restarts with different orderings
    best = []
    
    # First: greedy from high to low (original approach)
    S = _greedy(range(N, 0, -1), deadline)
    if len(S) > len(best):
        best = S
    
    # Greedy from low to high
    if time.time() < deadline:
        S = _greedy(range(1, N + 1), deadline)
        if len(S) > len(best):
            best = S
    
    # Random restarts
    rng = random.Random(42)
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = _greedy(order, deadline)
        if len(S) > len(best):
            best = S
            # Try local search on the best
            if time.time() < deadline - 10:
                S_local = _local_search(best, deadline)
                if len(S_local) > len(best):
                    best = S_local
    
    # Final local search on best
    if time.time() < deadline - 2:
        S_final = _local_search(best, deadline)
        if len(S_final) > len(best):
            best = S_final
    
    return best
# EVOLVE-BLOCK-END
