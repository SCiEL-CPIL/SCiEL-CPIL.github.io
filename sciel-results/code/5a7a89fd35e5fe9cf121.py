"""Improved program for the suite task; uses weighted greedy with vectorized updates."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a greedy algorithm with a candidate order sorted by weight centrality
    to maximize the number of packed codewords.
    """
    N = 18
    MIN_DIST = 6
    size = 1 << N
    
    # 1. Precompute masks for Hamming distance < MIN_DIST
    # We need all vectors with weight < 6.
    # Generating 2^18 integers and filtering is fast enough.
    masks = np.array([i for i in range(size) if i.bit_count() < MIN_DIST], dtype=np.uint32)
    
    # 2. Prepare candidate order
    # Heuristic: Process candidates with weight close to N/2 first.
    # This often leads to better packing in Hamming spaces.
    # We create an array of all candidates and sort by |weight - N/2|.
    # To avoid storing a huge list of tuples, we can use numpy argsort.
    
    # Calculate weights for all candidates
    # This is a bit slow in pure Python, but we can do it efficiently.
    # Or, we can just iterate through weight layers.
    
    # Let's generate candidates in order of increasing distance from weight 9.
    # Weights: 9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 17, 0, 18
    
    candidates = []
    for w in range(9, -1, -1):
        # Add weight w and weight 18-w?
        # Actually, we need to cover all weights.
        pass

    # Simpler approach: 
    # Generate all candidates, compute weights, sort by weight distance from 9.
    # 262k elements. Sorting is O(N log N). ~4M ops. Fast.
    
    all_candidates = np.arange(size, dtype=np.uint32)
    
    # Compute weights vectorized?
    # numpy doesn't have popcount for uint32 in all versions.
    # We can use a lookup table for bytes.
    # Or just use the list comprehension from the seed for weights too?
    # [i.bit_count() for i in range(size)] takes ~0.1s.
    
    weights = np.array([i.bit_count() for i in range(size)], dtype=np.uint8)
    target_weight = N // 2  # 9
    weight_diff = np.abs(weights.astype(np.int32) - target_weight)
    
    # Sort indices by weight_diff, then by value (to be deterministic)
    # We use a stable sort or just argsort on a combined key?
    # Just argsort on weight_diff. Ties will be in original order (0..size-1).
    # This means for a given weight, we process 0, 1, 2...
    # That's fine.
    
    sorted_indices = np.argsort(weight_diff, kind='mergesort')
    
    # 3. Greedy Selection
    forbidden = np.zeros(size, dtype=bool)
    code = []
    
    # Function to mark forbidden neighbors
    def mark_forbidden(c):
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True
        
    # Start with 0? 
    # 0 has weight 0. It will be processed last in our sorted order.
    # But 0 is a good seed.
    # Let's force 0 to be the first codeword.
    code.append(0)
    mark_forbidden(0)
    
    # Iterate through sorted candidates
    # We can skip 0 since it's already added.
    for idx in sorted_indices:
        c = idx
        if c == 0:
            continue
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)
            
    return code

# EVOLVE-BLOCK-END
