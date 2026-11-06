import time

# Deadline guard: track start time to ensure we finish well before 330s
_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin of 10s before 330s kill

def _is_time_up():
    return (time.time() - _START_TIME) > _DEADLINE_SECONDS

def priority(vector, n):
    """
    Prioritizes vectors based on the number of zero entries (Hamming weight complement)
    with a secondary bonus for uniform non-zero entries.
    Hypothesis: Vectors with more zeros are 'sparser' and less likely to form
    3-term arithmetic progressions. Additionally, vectors whose non-zero entries
    are all the same value have less internal structure and may interact less
    with other vectors, further reducing AP formation risk.
    """
    # 1. Count the number of zeros in the vector
    zero_count = 0
    nonzero_values = set()
    for v in vector:
        if v == 0:
            zero_count += 1
        else:
            nonzero_values.add(v)
    
    # 2. Base score: Higher zero count is better.
    base_score = float(zero_count)
    
    # 3. Secondary bonus: if non-zero entries are all the same value,
    #    the vector has less structural complexity.
    #    This is a smooth refinement: uniform non-zeros get a small boost.
    if nonzero_values and len(nonzero_values) == 1:
        base_score += 0.5
    
    # 4. Generate a pseudo-random tie-breaker
    # Encode vector as base-3 integer
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Apply SplitMix64 mixing for good distribution
    val = (val + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 31)
    
    # Convert to fraction [0, 1)
    random_frac = (val >> 11) / (1 << 53)
    
    # 5. Combine
    return base_score + random_frac
