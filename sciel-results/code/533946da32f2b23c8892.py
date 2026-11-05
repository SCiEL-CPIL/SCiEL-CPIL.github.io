"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random

# EVOLVE-BLOCK-START
def _check_valid(seq):
    """Check if all consecutive block sums in seq are distinct."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_forward():
    """Greedy: try values 1..256, append if valid."""
    values = []
    sums = set()
    for x in range(1, 257):
        # New consecutive block sums when appending x
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_reverse():
    """Greedy: try values 256..1, add to set if valid when sorted."""
    S = []
    for x in range(256, 0, -1):
        candidate = sorted(S + [x])
        if _check_valid(candidate):
            S.append(x)
    return sorted(S)

def _greedy_random(order_seed=None):
    """Greedy with random ordering of candidates."""
    if order_seed is None:
        order_seed = random.randint(0, 2**31 - 1)
    rng = random.Random(order_seed)
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    S = []
    for x in candidates:
        candidate = sorted(S + [x])
        if _check_valid(candidate):
            S.append(x)
    return sorted(S)

def construct():
    best = []
    
    # Strategy 1: Forward greedy (original)
    seq1 = _greedy_forward()
    if len(seq1) > len(best):
        best = seq1
    
    # Strategy 2: Reverse greedy
    seq2 = _greedy_reverse()
    if len(seq2) > len(best):
        best = seq2
    
    # Strategy 3: Multiple random restarts
    for seed in range(50):
        seq3 = _greedy_random(order_seed=seed)
        if len(seq3) > len(best):
            best = seq3
    
    # Strategy 4: Try a few more random seeds with different rng
    for seed in range(50, 150):
        seq4 = _greedy_random(order_seed=seed * 7919 + 13)
        if len(seq4) > len(best):
            best = seq4
    
    return best
# EVOLVE-BLOCK-END
