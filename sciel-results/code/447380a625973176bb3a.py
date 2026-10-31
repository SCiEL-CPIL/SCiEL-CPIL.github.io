import time
import math

_start_time = None
_DEADLINE = 310.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Prioritize vectors with 3 distinct symbols.
    2. Among those, use Shannon entropy as a smooth balance measure.
    3. Penalize cyclic structural correlation (proxy for AP-proneness).
    4. Use a fast linear hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Fast count of symbols
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Distinct count
    distinct_count = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Shannon entropy as smooth balance surrogate (replaces chi-squared)
    if n > 0:
        entropy = 0.0
        for c in (c0, c1, c2):
            if c > 0:
                p = c / n
                entropy -= p * math.log(p)
        balance_score = entropy / math.log(3)  # normalized to [0, 1]
    else:
        balance_score = 0.0
    
    # Cyclic correlation: structural proxy for 3-term AP involvement
    if n > 1:
        corr = 0
        for i in range(n):
            corr += vector[i] * vector[(i + 1) % n]
        corr_score = corr / (4.0 * n)  # normalize to ~[0,1]
    else:
        corr_score = 0.0
    
    # Fast linear hash for tie-breaking
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    # Combined score:
    # - distinct_count dominates (3 > 2 > 1)
    # - entropy rewards balanced symbol distribution (smooth surrogate)
    # - cyclic correlation penalizes structured patterns
    # - hash breaks ties deterministically
    distinct_score = distinct_count * 10000.0
    balance_scaled = balance_score * 100.0
    corr_scaled = -corr_score * 50.0
    
    return float(distinct_score + balance_scaled + corr_scaled + hash_score)
