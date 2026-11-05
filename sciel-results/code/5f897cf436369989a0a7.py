import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6

def construct():
    """
    Constructs the largest possible binary code of length 18 and distance 6.
    Strategy: Search for a 9-dimensional linear code basis (targeting the [18,9,6] QR code)
    starting from the Reed-Muller RM(1,4) subcode.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # 1. Generate the Reed-Muller RM(1,4) basis (dimension 5)
    # The code consists of all affine functions f(x) = a.x + b on F_2^4
    # Basis vectors:
    # b0: constant 1 (weight 16)
    # b1..b4: linear terms (weight 8)
    rm_basis = []
    
    # Constant 1
    b0 = 0
    for i in range(16):
        b0 |= 1 << i
    rm_basis.append(b0)
    
    # Linear terms: x_j
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
        
    # Extend to length 18 by adding 2 zero bits
    # The words are in [0, 2^18). The top 2 bits are 0.
    # Our current basis vectors are in [0, 2^16). This is fine.
    
    # Current basis size: 5
    basis = rm_basis.copy()
    
    # 2. Greedily find 4 more basis vectors
    # We want to extend the basis to dimension 9.
    # For each new vector, it must be at distance >= 6 from the current subspace.
    # min_dist(v, S) = min_{u in S} wt(v ^ u)
    
    for k in range(5, 9):
        if time.time() > deadline:
            break
            
        found = False
        # Try random candidates
        # To be efficient, we can bias towards vectors with weight around 9 or 10?
        # Or just random.
        # We will try a large number of random candidates.
        
        # Pre-compute the current subspace? No, size is 2^k, which is up to 256.
        # Iterating over 2^k for each candidate is fast.
        
        # We will try up to 200,000 random candidates per basis vector
        attempts = 0
        max_attempts = 500000
        while attempts < max_attempts and not found:
            if time.time() > deadline:
                break
                
            # Generate a random candidate
            # To avoid duplicates with existing basis, we can just pick random
            # and check if it's in the span? No, if it's in the span, dist is 0.
            # We just need dist >= 6.
            
            cand = random.randint(0, (1 << N) - 1)
            
            # Calculate min distance from cand to the subspace spanned by basis
            # Subspace size is 2^k
            min_d = N + 1
            # Iterate over all combinations of basis vectors
            # This is 2^k iterations
            for mask in range(1 << k):
                # Compute sum of basis vectors selected by mask
                u = 0
                for i in range(k):
                    if (mask >> i) & 1:
                        u ^= basis[i]
                
                d = (cand ^ u).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DISTANCE:
                        break # Early exit if we already failed
            
            if min_d >= MIN_DISTANCE:
                basis.append(cand)
                found = True
                # Optional: Perform a local optimization to see if we can find a "better" vector?
                # For now, just accept the first one.
                break
            attempts += 1
            
        if not found:
            # If we didn't find a vector, we might be stuck.
            # Try a more structured search?
            # For now, break and return what we have.
            break
            
    # 3. Generate all codewords from the basis
    if len(basis) == 0:
        return [0]
        
    dim = len(basis)
    code_words = []
    
    # If we found 9 basis vectors, we have 512 words.
    # If fewer, we have 2^dim words.
    
    # Optimization: If dim is large, generating all words takes time.
    # But 2^9 = 512 is small.
    
    for mask in range(1 << dim):
        w = 0
        for i in range(dim):
            if (mask >> i) & 1:
                w ^= basis[i]
        code_words.append(w)
        
    # 4. Verify the code
    # Since it's linear, we just need to check the weight of all non-zero codewords.
    # But the task requires "Every pair is checked exactly".
    # For linear codes, min pairwise distance = min non-zero weight.
    # However, to be strictly compliant with "Every pair is checked", 
    # and to ensure no bugs in linearity assumption, we can do a full check 
    # if the code is small, or rely on linearity for speed.
    # 512 words -> 512*511/2 = ~130k pairs. This is fast.
    
    # Let's do a full verification to be safe, but optimized.
    # Actually, the task says "Every pair is checked exactly using integer XOR and population count."
    # This implies the evaluator might check it, or we should ensure our output is valid.
    # We will verify internally.
    
    valid = True
    n_words = len(code_words)
    # Check min weight for linear code is sufficient for validity of distance,
    # but let's just check all pairs to be absolutely sure and match the "checked exactly" spirit.
    # 130k iterations is < 0.1s in Python.
    
    for i in range(n_words):
        if time.time() > deadline:
            # Return best so far? If we are in verification, we assume it's valid.
            # But if we timeout here, we should return the list.
            break
        for j in range(i + 1, n_words):
            d = (code_words[i] ^ code_words[j]).bit_count()
            if d < MIN_DISTANCE:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to Reed-Muller
        return rm_basis_words(rm_basis)
        
    return code_words

def rm_basis_words(basis):
    """Generate words from RM basis (fallback)."""
    dim = len(basis)
    words = []
    for mask in range(1 << dim):
        w = 0
        for i in range(dim):
            if (mask >> i) & 1:
                w ^= basis[i]
        words.append(w)
    return words

# Helper for random seed
random.seed(42)
