import time

_start_time = None
_DEADLINE = 285.0  # Safety margin before 300s kill


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score. Strongly prefer vectors
       using all three symbols to ensure full algebraic structure.
    2. Secondary: L2 balance penalty. Smooth surrogate for uniform symbol
       distribution (n/3, n/3, n/3).
    3. Tertiary: Variance-based danger proxy. Penalize high coordinate variance
       (extreme 0s and 2s) which are more likely to form affine lines.
    4. Quaternary: Lightweight spectral refinement. Use autocorrelation to
       measure local structural regularity; penalize high autocorrelation
       (repetitive patterns) which may correlate with AP formation.
    5. Symmetry breaking: Favor vectors with more 2s to prevent clustering.
    6. Deterministic hash tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast single-pass symbol counting and moment computation
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # For autocorrelation: sum of x[i]*x[i+1] for adjacent pairs
    autocorr = 0
    prev_val = None
    
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4
        
        if prev_val is not None:
            autocorr += prev_val * v
        prev_val = v

    # Factor 1: Quadratic distinct-symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Variance-based danger proxy
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Lightweight spectral refinement via autocorrelation
    # Normalize autocorrelation: divide by number of pairs and by variance
    # High autocorrelation indicates repetitive local structure
    if n > 1 and variance > 1e-9:
        # Normalize by variance to get correlation-like measure
        autocorr_norm = autocorr / ((n - 1) * variance)
        # Penalize high positive autocorrelation (smooth/repetitive patterns)
        spectral_score = -autocorr_norm * 50.0
    else:
        spectral_score = 0.0

    # Factor 6: Fast polynomial rolling hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + weight_score + balance_score + danger_score + spectral_score + hash_score
    return float(total)
