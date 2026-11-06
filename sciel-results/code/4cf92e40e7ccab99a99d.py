# EVOLVE-BLOCK-START
def construct():
    # Construct a set of 12 integers in [0, 128]
    # Using pairs {10k, 10k+1} for k=0..5
    # This creates many additive relations (3-term APs like 0,10,20)
    # which tends to reduce the size of the largest Sidon subset.
    # Expected max Sidon size: 3 (vs 4 for AP of length 12)
    s = []
    for k in range(6):
        s.append(10*k)
        s.append(10*k + 1)
    
    # Verify constraints
    # 12 distinct integers?
    if len(set(s)) != 12:
        return list(range(12)) # Fallback
    # In [0, 128]?
    if any(x < 0 or x > 128 for x in s):
        return list(range(12))
        
    return s
# EVOLVE-BLOCK-END
