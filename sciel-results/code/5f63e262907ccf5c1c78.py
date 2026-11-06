def priority(vector, n):
    """
    Prioritizes vectors based on the number of zero entries (Hamming weight complement).
    Hypothesis: Vectors with more zeros are 'sparser' and less likely to form
    3-term arithmetic progressions with other sparse vectors, allowing for a larger
    cap set. This focuses on the structural role of the zero element in F_3.
    """
    # 1. Count the number of zeros in the vector
    # This is a structural feature specific to the value 0, unlike distinct symbol count.
    zero_count = 0
    for v in vector:
        if v == 0:
            zero_count += 1
            
    # 2. Base score: Higher zero count is better.
    # We use a float to allow tie-breaking.
    base_score = float(zero_count)
    
    # 3. Generate a pseudo-random tie-breaker
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
    
    # 4. Combine
    return base_score + random_frac
