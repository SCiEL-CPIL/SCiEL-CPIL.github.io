import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s limit

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def _compute_structural_score(vector, n):
    """
    Compute a structural score that captures cap-set-relevant features:
    1. Zero count (penalty - zeros increase triple risk with dense vectors)
    2. Coordinate balance (reward balanced 1/2 distribution)
    3. Adjacent coordinate patterns (penalize patterns prone to forming triples)
    """
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
    
    # Balance score: reward when c1 and c2 are close
    balance = abs(c1 - c2)
    
    # Adjacent pattern penalty: count consecutive same non-zero values
    # These patterns are more likely to participate in arithmetic progressions
    adj_penalty = 0
    prev = -1
    for v in vector:
        if v != 0:
            if v == prev:
                adj_penalty += 1
            prev = v
        else:
            prev = -1
    
    return c0, balance, adj_penalty

def priority(vector, n):
    # Check deadline - if we're running out of time, return a deterministic
    # but simple score to avoid computation overhead
    if time.time() > _DEADLINE:
        h = _hash_vec(vector)
        return h - 30.0
    
    c0, balance, adj_penalty = _compute_structural_score(vector, n)
    
    h = _hash_vec(vector)
    
    # Composite score:
    # - Base hash for deterministic ordering
    # - Strong penalty for zeros (they create more triple opportunities)
    # - Moderate penalty for imbalance between 1s and 2s
    # - Small penalty for adjacent same-value patterns
    score = h - 45.0 * c0 - 2.0 * balance - 1.0 * adj_penalty
    
    return score
