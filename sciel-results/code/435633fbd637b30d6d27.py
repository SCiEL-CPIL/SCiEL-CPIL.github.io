"""
4-term progression-free set construction for [1, 512].
Uses greedy construction followed by local search (add-back and swap moves).
"""
import time
import random
from itertools import combinations

N = 512
K = 4  # length of arithmetic progression

def has_4ap_in_set(S):
    """Check if set S contains a non-constant 4-term arithmetic progression."""
    S = set(S)
    for a in S:
        for d in range(1, (N - a) // 3 + 1):
            if a + d in S and a + 2*d in S and a + 3*d in S:
                return True
    return False

def is_valid_addition(S, x):
    """Check if adding x to S keeps it 4-AP-free."""
    # A 4-AP containing x can have x at position 0, 1, 2, or 3
    # x, x+d, x+2d, x+3d  -> x at pos 0
    # x-d, x, x+d, x+2d  -> x at pos 1
    # x-2d, x-d, x, x+d  -> x at pos 2
    # x-3d, x-2d, x-d, x -> x at pos 3
    for d in range(1, N):
        # x at position 0: need x+d, x+2d, x+3d all in S
        if x + 3*d <= N and (x+d) in S and (x+2*d) in S and (x+3*d) in S:
            return False
        # x at position 1: need x-d, x+d, x+2d all in S
        if x - d >= 1 and (x-d) in S and (x+d) in S and (x+2*d) in S:
            return False
        # x at position 2: need x-2d, x-d, x+d all in S
        if x - 2*d >= 1 and (x-2*d) in S and (x-d) in S and (x+d) in S:
            return False
        # x at position 3: need x-3d, x-2d, x-d all in S
        if x - 3*d >= 1 and (x-3*d) in S and (x-2*d) in S and (x-d) in S:
            return False
    return True

def greedy_construct(order=None):
    """Greedy: add elements in given order if valid."""
    if order is None:
        order = list(range(1, N+1))
    S = set()
    for x in order:
        if is_valid_addition(S, x):
            S.add(x)
    return S

def local_search(S, time_limit=30.0):
    """Local search: try adding removed elements, then swap moves."""
    S = set(S)
    start_time = time.time()
    improved = True
    removed = []
    
    # First pass: try to add back elements not in S
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for x in range(1, N+1):
            if x not in S and is_valid_addition(S, x):
                S.add(x)
                improved = True
    
    # Second pass: try swap moves (remove one, add two)
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for a in list(S):
            S.remove(a)
            added = 0
            for x in range(1, N+1):
                if x not in S and is_valid_addition(S, x):
                    S.add(x)
                    added += 1
                    break  # add at most one per swap attempt
            if added > 0:
                improved = True
            else:
                S.add(a)  # put back
    return S

def random_greedy(seed, order=None):
    """Greedy with random tie-breaking."""
    rng = random.Random(seed)
    if order is None:
        order = list(range(1, N+1))
        rng.shuffle(order)
    S = set()
    for x in order:
        if is_valid_addition(S, x):
            S.add(x)
    return S

def construct():
    """Construct a large 4-AP-free subset of [1, 512]."""
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Start with greedy in natural order
    best_S = greedy_construct()
    best_size = len(best_S)
    
    # Try greedy in reverse order
    S_rev = greedy_construct(list(range(N, 0, -1)))
    if len(S_rev) > best_size:
        best_S = S_rev
        best_size = len(S_rev)
    
    # Try several random orderings
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        S = random_greedy(trial, order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Local search on the best found so far
    if time.time() < deadline:
        S_ls = local_search(best_S, time_limit=min(20.0, deadline - time.time()))
        if len(S_ls) > best_size:
            best_S = S_ls
            best_size = len(S_ls)
    
    # Try local search on a few other candidates
    if time.time() < deadline:
        for trial in range(10):
            if time.time() > deadline:
                break
            order = list(range(1, N+1))
            rng.shuffle(order)
            S = random_greedy(1000 + trial, order)
            S_ls = local_search(S, time_limit=min(5.0, deadline - time.time()))
            if len(S_ls) > best_size:
                best_S = S_ls
                best_size = len(S_ls)
    
    return sorted(best_S)
