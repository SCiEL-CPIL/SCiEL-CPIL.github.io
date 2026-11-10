"""Improved program: 3-AP-free greedy with maximization and full-constraint verification."""
import random

def _is_3ap_free_add(x, S):
    """Return True if adding x to set S keeps it free of 3-term arithmetic progressions."""
    two_x = 2 * x
    # Check if x is the mean of any pair in S
    for b in S:
        c = two_x - b
        if c in S and c != b:
            return False
    # Check if x is an endpoint of any 3-AP with two elements in S
    # Case 1: x is the smaller endpoint (x, b, c) -> 2b = x + c -> c = 2b - x
    # Case 2: x is the larger endpoint (a, b, x) -> 2b = a + x -> a = 2b - x
    # In both cases, we check if there exists b in S such that 2b - x is also in S.
    for b in S:
        c = 2 * b - x
        if c in S and c != b: # c != b is guaranteed if x != b, which is true
            return False
    return True

def _element_is_mean(a, S):
    """Return True if a is the exact arithmetic mean of some subset of S\{a} of size >= 2."""
    others = [x for x in S if x != a]
    n = len(others)
    if n < 2:
        return False
    max_sum = sum(others)
    mask = (1 << (max_sum + 1)) - 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in others:
        for c in range(n - 1, -1, -1):
            if dp[c]:
                dp[c + 1] |= (dp[c] << e) & mask
    for c in range(2, n + 1):
        target = c * a
        if target <= max_sum and ((dp[c] >> target) & 1):
            return True
    return False

def _is_fully_valid(S):
    for a in S:
        if _element_is_mean(a, S):
            return False
    return True

def _maximize_3ap_free(S, universe):
    """Try to add any elements from universe that are not in S and keep it 3-AP-free."""
    S = set(S)
    remaining = [x for x in universe if x not in S]
    for x in remaining:
        if _is_3ap_free_add(x, S):
            S.add(x)
    return S

def construct():
    rng = random.Random(20240601)
    universe = list(range(1, 513))
    best_valid = set()
    best_len = 0
    
    n_restarts = 2000
    for _ in range(n_restarts):
        order = universe[:]
        rng.shuffle(order)
        
        S = set()
        # Greedy phase
        for x in order:
            if _is_3ap_free_add(x, S):
                S.add(x)
                
        # Maximization phase
        S = _maximize_3ap_free(S, universe)
        
        # Verification phase
        if len(S) <= best_len:
            continue
            
        if _is_fully_valid(S):
            best_len = len(S)
            best_valid = S
            
        # Early termination if we hit the theoretical max or a very high number
        if best_len >= 48:
            break

    # Fallback
    if not best_valid:
        best_valid = {1, 2, 4, 8, 16, 32, 64, 128, 256}

    result = sorted(x for x in best_valid if 1 <= x <= 512)[:48]
    return result
