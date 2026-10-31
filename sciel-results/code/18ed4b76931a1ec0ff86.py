"""Cap set greedy builder with projection-based noise shaping."""
import time
import numpy as np

# Deadline tracking
_start_time = None
_DEADLINE = 330.0 * 0.95  # 313.5s safety margin

# Fixed high-entropy pseudo-random vectors for projection in F_3^n
# These are generated once and reused across all priority calls
_PROJ_SEED = 12345
_PROJ_VECTORS = []
_PROJ_MAX_N = 20  # Precompute for n up to 20

def _init_projections():
    """Initialize fixed projection vectors for noise shaping."""
    global _PROJ_VECTORS
    if _PROJ_VECTORS:
        return
    rng = np.random.RandomState(_PROJ_SEED)
    for n in range(1, _PROJ_MAX_N + 1):
        # Generate 3 independent random vectors in F_3^n
        # Map to {0, 1, 2} for F_3 arithmetic
        vecs = []
        for _ in range(3):
            v = rng.randint(0, 3, size=n)
            vecs.append(v)
        _PROJ_VECTORS.append(vecs)

def _smooth_balance(c0, c1, c2, n):
    """
    Smooth surrogate for balance using soft-min via log-sum-exp.
    Instead of hard L1 deviation, use a smooth measure that approximates
    how close the counts are to uniform (n/3 each).
    
    Uses: -log(sum(exp((ci - ideal)^2 / T)) for i in {0,1,2})
    which is a soft-max of the squared deviations, negated to reward balance.
    """
    ideal = n / 3.0
    T = max(1.0, n / 10.0)  # Temperature parameter
    sq_devs = [(c0 - ideal) ** 2 / T, (c1 - ideal) ** 2 / T, (c2 - ideal) ** 2 / T]
    # Soft-max via log-sum-exp for numerical stability
    m = max(sq_devs)
    log_sum = m + np.log(np.exp(sq_devs[0] - m) + np.exp(sq_devs[1] - m) + np.exp(sq_devs[2] - m))
    return -log_sum * 5.0  # Scale to be comparable to other terms

def priority(vector, n):
    """
    Static priority function for cap set construction in {0,1,2}^n.
    
    Strategy: Projection-based noise shaping with smooth balance surrogate.
    1. Primary: Prioritize vectors with all 3 distinct symbols (most "generic").
    2. Secondary: Projection magnitude onto fixed high-entropy pseudo-random vectors
       in F_3^n, treating selection as noise-shaping to decorrelate from lattice symmetries.
    3. Tertiary: Smooth balance measure using soft-min (log-sum-exp) of squared deviations.
    4. Quaternary: Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        _init_projections()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0
    
    # Fast symbol counting
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
    
    # Distinct count (3 is best, 2 is middle, 1 is worst)
    # Scale to make it dominant: 3 distinct gets +30000, 2 gets +20000, 1 gets +10000
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0 or c0 > 0 and c2 > 0 or c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Projection-based noise shaping:
    # Compute projection magnitude onto fixed pseudo-random vectors in F_3^n
    # This treats the greedy selection as a noise-shaping process that
    # decorrelates the chosen set from structured lattice symmetries
    proj_score = 0.0
    if n <= _PROJ_MAX_N and n > 0:
        proj_vecs = _PROJ_VECTORS[n - 1]
        for pv in proj_vecs:
            # Compute inner product in F_3: sum(vector[i] * pv[i]) mod 3
            # But for projection magnitude, we want the actual magnitude, not mod 3
            # Use the integer sum of products as a proxy for projection magnitude
            dot = 0
            for i, val in enumerate(vector):
                dot += val * pv[i]
            # Use absolute value and normalize by sqrt(n) for scale invariance
            proj_score += abs(dot)
        proj_score /= (3 * np.sqrt(n))  # Normalize by number of vectors and dimension
    
    # Smooth balance measure using soft-min via log-sum-exp
    balance_score = _smooth_balance(c0, c1, c2, n)
    
    # Fast hash: simple polynomial rolling hash with bit masking for speed
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    # Combine: distinct dominates, then projection, then balance, then hash
    return float(distinct_score + proj_score * 100.0 + balance_score + hash_score)
