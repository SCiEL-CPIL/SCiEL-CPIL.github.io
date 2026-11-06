import time
import hashlib

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s kill

def _get_priority(vector, n):
    """
    Compute priority for a vector in {0,1,2}^n.
    
    Strategy:
    1. Primary: Hamming Weight (number of non-zero elements).
       Hypothesis: Vectors with higher Hamming weight are 'more generic' 
       and less likely to form lines with 'structured' vectors early on.
       We prioritize high Hamming weight.
    2. Secondary: Sum of coordinates.
    3. Tertiary: Deterministic Hash (SHA256) for tie-breaking and fine-grained ordering.
    """
    # Feature 1: Hamming Weight (count of non-zeros)
    # High weight = fewer zeros.
    h_weight = 0
    s_sum = 0
    for v in vector:
        if v != 0:
            h_weight += 1
        s_sum += v
        
    # Feature 2: Hash
    # Use SHA256 for a strong, uniform distribution
    h = hashlib.sha256(str(vector).encode()).hexdigest()
    # Convert first 16 hex chars to int for a 64-bit value
    hash_val = int(h[:16], 16)
    
    # Combine:
    # Scale Hamming Weight to dominate the order.
    # Scale Hash to break ties.
    # We want higher priority for higher Hamming Weight.
    
    # Normalization:
    # h_weight is in [0, n]
    # hash_val is in [0, 2^64)
    
    # To ensure Hamming Weight is the primary sort key:
    # priority = h_weight * LARGE_CONSTANT + (hash_val % LARGE_CONSTANT)
    
    # Let LARGE_CONSTANT be 2^32 to be safe, or just use a large multiplier.
    # Since n is likely small (e.g., 10-20), h_weight is small.
    # We can map h_weight to the top bits.
    
    # Let's use a simpler combination:
    # Sort by h_weight descending, then by hash_val.
    # priority = h_weight * 1e20 + hash_val / 1e10
    
    # To avoid floating point issues, let's return a float that preserves order.
    # h_weight * 10^10 + (hash_val % 10^10)
    
    # However, the function returns a float.
    # If n is up to 20, h_weight is up to 20.
    # 20 * 10^10 is 2e11.
    # hash_val % 10^10 is up to 1e10.
    # This fits in a float (53 bits of precision ~ 9e15).
    
    priority = h_weight * 1e10 + (hash_val % 10000000000)
    
    return float(priority)

def priority(vector, n):
    """
    Priority function for cap set greedy construction.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return 0.0
    
    try:
        return _get_priority(vector, n)
    except Exception:
        return 0.0
