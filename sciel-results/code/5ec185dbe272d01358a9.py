"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # safety margin
    
    N = 18
    MIN_DIST = 6
    
    # Build the extended quadratic residue code of length 18
    # The quadratic residues mod 19: {1,4,9,16,6,17,11,7,5}
    # Extended QR code: add a parity bit
    # Generator matrix approach: use the known [18,6,6] code
    
    # Actually, let's use a simpler approach: 
    # Start with a good seed code and greedily add words
    
    # Initial code: use the first-order Reed-Muller RM(1,4) extended to 18 bits
    # This gives 32 words with distance 8 (very safe)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Convert to numpy array for fast operations
    # We'll use a greedy approach: try to add words that are far from all existing
    
    # For efficiency, we'll maintain the code as a list and use vectorized distance checks
    code = np.array(words, dtype=np.uint32)
    
    def min_dist_to_code(candidate, code_arr):
        """Compute minimum Hamming distance from candidate to all words in code_arr."""
        # XOR and popcount
        xored = code_arr ^ candidate
        # Count bits using lookup or numpy
        # Use bit_count via numpy - we need a fast method
        # For small arrays, use a lookup table
        return None  # placeholder
    
    # Better approach: use a precomputed popcount for 18-bit values
    # 2^18 = 262144, manageable
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Greedy addition
    # We'll try words in a specific order to maximize additions
    # Order: start with high-weight words, then vary
    
    # Actually, let's use a smarter greedy: 
    # 1. Start with the RM code
    # 2. For each candidate word (in some order), check if min distance >= 6
    # 3. If yes, add it
    
    # To make this fast, we'll use the popcount table
    # And we'll process candidates in batches
    
    # Generate candidates in Gray code order or random order
    # Let's try: all words from 0 to 2^18-1, but skip if too close
    
    # This is 262144 candidates, each checked against up to ~1000 code words
    # That's 262144 * 1000 = 262 million distance checks - too slow in Python
    
    # Better: use numpy vectorization
    # code is shape (k,), candidate is scalar
    # xored = code ^ candidate -> shape (k,)
    # dists = popcount[xored] -> shape (k,)
    # min_dist = dists.min()
    
    # This should be fast with numpy!
    
    added = 0
    # Try all 2^18 words
    # But that's 262144 iterations, each with a numpy operation on array of size k
    # For k=1000, that's 262144 * (1000 ops) = manageable with numpy
    
    # Let's try it
    for candidate in range(1 << N):
        if time.time() > DEADLINE:
            break
        
        # Check if candidate is already in code
        # (skip for speed, we'll handle duplicates later)
        
        xored = code ^ candidate
        dists = popcount[xored]
        min_d = dists.min()
        
        if min_d >= MIN_DIST:
            code = np.append(code, candidate)
            added += 1
    
    # Convert back to list of integers
    result = code.tolist()
    
    # Ensure all are distinct and valid
    result = list(set(result))
    
    # Verify minimum distance
    # (skip for speed, trust the greedy)
    
    return result
# EVOLVE-BLOCK-END
