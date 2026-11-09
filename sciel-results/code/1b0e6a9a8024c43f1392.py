"""Cap set greedy construction with zero-identity bias and time-varying hash."""
import time
import math

# Deadline guard
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 315.0  # safety margin before 330s kill

def _elapsed():
    return time.monotonic() - _START_TIME

def _remaining():
    return _DEADLINE_SECONDS - _elapsed()

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    Structural insight: 0 is the additive identity in F_3. Vectors with more zeros
    have fewer 'active' coordinates, making them structurally less likely to form
    the symmetric triples required for 3-term APs. We heavily favor vectors with
    many zeros, with a time-varying hash to break symmetry and explore different
    regions of vector space over time.
    """
    # Time guard: if we're close to deadline, return a simple priority
    if _remaining() < 5.0:
        c0 = sum(1 for x in vector if x == 0)
        return float(c0)
    
    v = vector
    n_local = n
    
    # Count digits
    c0 = 0
    c1 = 0
    c2 = 0
    for x in v:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Primary: strongly favor vectors with many zeros (identity bias)
    # The more zeros, the fewer coordinates that can participate in AP formation
    zero_score = c0 * 1000.0
    
    # Secondary: balance among non-zero digits
    # Among the non-zero positions, prefer balance between 1s and 2s
    non_zero = c1 + c2
    if non_zero > 0:
        balance_penalty = abs(c1 - c2) * 10.0
    else:
        balance_penalty = 0.0
    
    # Tertiary: time-varying hash to break symmetry
    # This causes the greedy algorithm to explore different regions over time,
    # escaping static-heuristic local patterns
    elapsed = _elapsed()
    phase = int(elapsed / 20.0)  # change every 20 seconds
    
    # Compute a hash-like value from the vector, modulated by phase
    hash_val = 0.0
    for i, x in enumerate(v):
        # Position-dependent weighting with phase shift
        weight = math.sin(i * 0.7 + phase * 1.3) * 0.5 + 0.5
        hash_val += x * weight
    
    # Normalize hash to [0, 1]
    hash_norm = hash_val / (n_local * 2.0) if n_local > 0 else 0.0
    
    # Quaternary: lexicographic tie-breaker
    lex_val = 0.0
    for x in v:
        lex_val = lex_val * 3.0 + x
    
    # Combine: zero bias dominates, then balance, then time-varying hash, then lex
    score = zero_score - balance_penalty + hash_norm * 500.0 + lex_val * 0.001
    
    return score
