import hashlib

# No global cache needed if we assume vectors are unique in the sort, 
# but caching helps if the evaluator queries the same vector multiple times 
# (unlikely in a single sort, but safe to keep if fast).
# However, the prompt says "vectors ... are added greedily". 
# Usually this implies a single pass over a sorted list.
# We will keep the function stateless or lightly cached.

def priority(vector, n):
    """
    Returns a priority for the vector. Higher priority means the vector is 
    added earlier in the greedy process.
    
    Strategy: 
    1. Use a 64-bit slice of SHA256 for a high-quality random base order.
    2. Add a strong bias towards vectors with HIGHER weight (fewer zeros).
       Hypothesis: Low-weight vectors (like all-zeros) are 'hubs' in the 
       conflict graph (they form lines with many pairs). By deprioritizing 
       them, we allow the greedy algorithm to select a larger set of 
       'peripheral' high-weight vectors that conflict less with each other.
    """
    # Convert vector to bytes for hashing
    # vector is a tuple of ints 0, 1, 2
    v_bytes = bytes(vector)
    
    # Compute SHA256
    h = hashlib.sha256(v_bytes).digest()
    
    # Take the first 8 bytes for a 64-bit integer
    hash_val = int.from_bytes(h[:8], byteorder='big')
    
    # Normalize to [0, 1)
    random_part = hash_val / (2**64)
    
    # Compute weight: number of non-zero elements
    # This is the "distance" from the zero vector
    weight = sum(1 for x in vector if x != 0)
    
    # Normalize weight to [0, 1]
    # If n is 0, avoid division by zero, though n should be >= 1
    if n == 0:
        norm_weight = 0.0
    else:
        norm_weight = weight / n
        
    # Bias: We want to prioritize vectors with HIGHER weight.
    # The random part is in [0, 1).
    # We add a bias up to 0.5.
    # This means a vector with weight n will have a priority boost of 0.5.
    # A vector with weight 0 will have a boost of 0.
    # This effectively pushes high-weight vectors to the top.
    bias = 0.5 * norm_weight
    
    # Final priority
    p = random_part + bias
    
    return p
