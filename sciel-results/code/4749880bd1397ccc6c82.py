import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    # Construct the extended QR(17) code deterministically
    p = 17
    
    # Quadratic residues modulo 17: {1, 2, 4, 8, 9, 13, 15, 16}
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Create the base vector v of length 17
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    mask_p = (1 << p) - 1
    for _ in range((p - 1) // 2):  # 8 shifts
        basis.append(current)
        current = ((current << 1) | (current >> (p - 1))) & mask_p
    
    # Add the all-ones vector (9th basis vector)
    basis.append((1 << p) - 1)
    
    num_basis = len(basis)  # 9
    
    # Vectorized generation of all 512 codewords using NumPy
    # Create all 512 coefficient vectors as uint16 array
    coeffs = np.arange(1 << num_basis, dtype=np.uint16)
    
    # For each coefficient vector, compute XOR combination of basis vectors
    # Use bit-level operations: for each bit position in the coefficient,
    # XOR the corresponding basis vector
    code_words = np.zeros(1 << num_basis, dtype=np.uint32)
    
    for i in range(num_basis):
        # Select codewords where bit i is set
        mask = (coeffs & (1 << i)) != 0
        code_words[mask] ^= basis[i]
    
    # Compute parity bits and form 18-bit words
    # Weight of each 17-bit word
    weights = np.array([w.bit_count() for w in code_words], dtype=np.uint8)
    parities = weights % 2
    
    # Form 18-bit integers: (word << 1) | parity
    full_words = (code_words.astype(np.uint32) << 1) | parities.astype(np.uint32)
    
    code = full_words.tolist()
    
    # Verify no duplicates (should be 512 distinct words)
    if len(set(code)) != len(code):
        code = list(set(code))
    
    # Local search: try to add more codewords
    best_code = list(code)
    best_size = len(best_code)
    
    # Pre-filter: valid weights are 6 to 12 (distance from 0 and all-ones)
    # 0 is in the code, all-ones (2^18-1) is in the code
    # dist(w, 0) = wt(w) >= 6, dist(w, all_ones) = 18 - wt(w) >= 6
    
    random.seed(42)
    attempts = 0
    max_attempts = 200000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate random 18-bit word
        w = random.randint(0, (1 << N) - 1)
        
        wt = w.bit_count()
        if wt < 6 or wt > 12:
            continue
        
        # Check against all codewords
        valid = True
        for c in best_code:
            if (w ^ c).bit_count() < MIN_DIST:
                valid = False
                break
        
        if valid:
            best_code.append(w)
            best_size = len(best_code)
    
    return best_code
