"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary code of length 18 with minimum Hamming distance at least 6.
    Uses a batched randomized greedy approach to find a large nonlinear code.
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = time.time() + 55.0  # Safety margin for 60s limit
    
    # Start with a small seed set, e.g., the Reed-Muller RM(1, 4) extended to 18 bits
    # RM(1, 4) has 32 words. We can use these as a starting point.
    # Or just start with 0 and 1...0?
    # Let's start with the all-zero vector and a few random ones to ensure diversity.
    
    code = [0]  # Start with 0
    
    # Add the all-ones vector? Distance 18.
    code.append((1 << N) - 1)
    
    # Add some random vectors to bootstrap
    for _ in range(10):
        v = random.getrandbits(N)
        # Check if it's valid against current code
        valid = True
        for w in code:
            if (v ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            code.append(v)
            
    # Convert to NumPy array for fast distance calculation
    # Shape: (num_words, N)
    # We will store as int array, and use XOR + sum for distance.
    
    while time.time() < DEADLINE:
        if len(code) > 500: # We aim for 512, but 500+ is a good stop if we can't get more
            break
            
        # Generate a batch of candidates
        batch_size = 5000
        candidates = np.array([random.getrandbits(N) for _ in range(batch_size)], dtype=np.uint32)
        
        # Current code as NumPy array
        code_array = np.array(code, dtype=np.uint32)
        
        # Compute distances between candidates and all code words
        # candidates shape: (batch_size, 1)
        # code_array shape: (1, num_words)
        # XOR shape: (batch_size, num_words)
        
        # To avoid memory blowup if code is large (512 * 5000 * 4 bytes ~ 10MB, OK)
        # But we need to count bits.
        
        # Efficient bit counting:
        # 1. XOR
        # 2. Use a lookup table or built-in bit_count if available in numpy? 
        # NumPy doesn't have a direct bit_count for uint32 arrays that is very fast in pure numpy without C extensions.
        # However, we can use a trick:
        # Split into bytes and use a lookup table?
        # Or simply: since N=18, we can just use Python's bit_count in a list comprehension?
        # 5000 * 512 = 2.5M operations. Python loop is ~1us per op -> 2.5s. Too slow for many iterations.
        
        # Let's use a NumPy-friendly bit count method.
        # Or, we can check distance < 6 by checking against spheres.
        # Sphere of radius 5 around 0 has size 12616.
        # If we pre-compute the set of "bad" offsets (distance < 6), we can check if (candidate - w) is in the set?
        # XOR is not subtraction.
        
        # Let's stick to the NumPy XOR and then use a vectorized bit count.
        # We can create a lookup table for 8-bit chunks.
        
        # Actually, for N=18, we can just do:
        # xors = candidates[:, None] ^ code_array[None, :]
        # Then we need popcount.
        
        # A fast way to popcount in NumPy:
        # xors = xors.astype(np.uint8) # No, uint32
        
        # Let's try the Python loop but optimized with early termination?
        # No, let's use the "bad offsets" idea.
        # Pre-compute all vectors v such that wt(v) < 6.
        # There are 12616 such vectors.
        # For a candidate c, it is invalid if there exists w in code such that c ^ w is in BadSet.
        # i.e., w is in { c ^ b for b in BadSet }.
        # So for each candidate, we generate the set of "forbidden" code words.
        # If any of these forbidden words are in the current code, the candidate is invalid.
        
        # This is:
        # ForbiddenSet = { c ^ b for b in BadSet }
        # Check intersection with CodeSet.
        
        # CodeSet is a set of integers.
        # BadSet is a list of 12616 integers.
        # For each candidate, we do 12616 XORs and set lookups.
        # 5000 candidates * 12616 ops = 63M ops. Too slow in Python.
        
        # Let's go back to NumPy XOR + Popcount.
        # We can use a lookup table for 16-bit chunks.
        
        # Create a 65536 entry lookup table for bit counts of 16-bit values.
        # This is memory efficient and fast.
        
        if not hasattr(construct, 'bit_count_table'):
            construct.bit_count_table = np.array([i.bit_count() for i in range(65536)], dtype=np.uint8)
        
        xors = candidates[:, None] ^ code_array[None, :]
        
        # Split into high and low 16 bits? N=18, so we need 2 chunks of 9 bits?
        # Or just 1 chunk of 18 bits? Table size 2^18 = 262144.
        # Let's use a 2^18 table.
        
        if not hasattr(construct, 'bit_count_table_18'):
            construct.bit_count_table_18 = np.array([i.bit_count() for i in range(1 << N)], dtype=np.uint8)
        
        # Map xors to indices
        # xors are uint32, but values < 2^18
        dists = construct.bit_count_table_18[xors]
        
        min_dists = np.min(dists, axis=1)
        
        valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        if len(valid_indices) > 0:
            # Add these candidates to the code
            new_words = candidates[valid_indices].tolist()
            code.extend(new_words)
            
    return code

# EVOLVE-BLOCK-END
