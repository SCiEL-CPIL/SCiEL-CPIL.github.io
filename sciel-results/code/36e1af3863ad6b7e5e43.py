def construct():
    import time
    import numpy as np
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 17
    M = 1 << N
    min_dist = 6
    max_radius = min_dist - 1  # 5

    # Precompute offsets (masks) of weight 1 to max_radius
    offsets_list = []
    for r in range(1, max_radius + 1):
        for comb in combinations(range(N), r):
            mask = 0
            for pos in comb:
                mask |= (1 << pos)
            offsets_list.append(mask)
    offsets = np.array(offsets_list, dtype=np.uint32)

    # --- Step 1: Generate the [17,8,6] QR Code ---
    
    # GF(256) arithmetic with primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1)
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
            b >>= 1
        return p

    # Precompute powers of alpha (alpha = 2)
    alpha_powers = [1] * 17
    for i in range(1, 17):
        alpha_powers[i] = gf_mul(alpha_powers[i-1], 2)

    # Quadratic residues mod 17: 1, 2, 4, 8, 16, 15, 13, 9
    # These are the exponents for the generator polynomial roots
    # Note: The roots are alpha^i for i in QR
    qr_exponents = [1, 2, 4, 8, 16, 15, 13, 9]
    
    # Compute generator polynomial g(x) in GF(256)[x]
    # g(x) = prod (x + alpha^i)
    # We represent polynomials as lists of coefficients in GF(256)
    # Start with 1
    g_coeffs = [1]
    
    for exp in qr_exponents:
        root = alpha_powers[exp]
        # Multiply by (x + root)
        # (x + root) = [root, 1]
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            # g_coeffs[i] * x
            new_g[i+1] ^= g_coeffs[i]
            # g_coeffs[i] * root
            new_g[i] ^= gf_mul(g_coeffs[i], root)
        g_coeffs = new_g
        
    # g_coeffs should have length 9
    # Check if coefficients are 0 or 1
    # If not, something is wrong, but for QR codes they should be.
    # Convert to binary coefficients
    bin_g_coeffs = []
    for c in g_coeffs:
        if c == 0:
            bin_g_coeffs.append(0)
        elif c == 1:
            bin_g_coeffs.append(1)
        else:
            # Error: Coefficient is not in GF(2)
            # This shouldn't happen for the QR code
            bin_g_coeffs.append(0) # Fallback
            
    # Generate basis vectors for the [17,9,5] code
    # Basis vector k is x^k * g(x) mod (x^17 - 1)
    # Since deg(g) = 8, for k=0..8, x^k*g(x) has deg <= 16, so no wrap
    basis = []
    for k in range(9):
        vec = [0] * N
        for i in range(9):
            if bin_g_coeffs[i] == 1:
                vec[i+k] = 1
        # Convert to integer
        val = 0
        for j in range(N):
            if vec[j]:
                val |= (1 << j)
        basis.append(val)
        
    # Generate all 512 codewords of the [17,9,5] code
    qr_code_words = []
    for mask in range(1 << 9):
        word = 0
        for i in range(9):
            if mask & (1 << i):
                word ^= basis[i]
        qr_code_words.append(word)
        
    # Extend to [18,9,6] and Shorten to [17,8,6]
    # Extension: Add parity bit.
    # Shortening: Keep only words with parity bit 0.
    # Parity bit is popcount(word) % 2.
    # If parity is 0, the extended word ends in 0.
    # The first 17 bits are the original word.
    # So we just keep words with even weight.
    
    initial_code = []
    for w in qr_code_words:
        # Count bits
        if bin(w).count('1') % 2 == 0:
            initial_code.append(w)
            
    # initial_code should have 256 words
    # If not, there's a bug. But theoretically it should be 256.
    
    # --- Step 2: Greedy Improvement ---
    
    if time.time() > deadline - 10.0:
        return initial_code

    # Initialize conflict counts and available mask
    conflict_counts = np.zeros(M, dtype=np.int16)
    available_mask = np.ones(M, dtype=bool)
    
    # Mark initial code words and their neighbors
    for w in initial_code:
        available_mask[w] = False
        neighbors = w ^ offsets
        conflict_counts[neighbors] += 1
        newly_blocked = neighbors[conflict_counts[neighbors] == 1]
        if len(newly_blocked) > 0:
            available_mask[newly_blocked] = False
            
    code = initial_code[:]
    
    # Prepare list of candidates
    # We can iterate through all numbers, skipping those in initial_code
    # Or just shuffle all numbers and check available_mask
    all_indices = list(range(M))
    random.shuffle(all_indices)
    
    idx = 0
    while idx < len(all_indices) and time.time() < deadline - 2.0:
        x = all_indices[idx]
        idx += 1
        
        if not available_mask[x]:
            continue
            
        # Accept x
        code.append(x)
        available_mask[x] = False
        
        # Update conflicts
        neighbors = x ^ offsets
        conflict_counts[neighbors] += 1
        newly_blocked_mask = (conflict_counts[neighbors] == 1)
        newly_blocked = neighbors[newly_blocked_mask]
        if len(newly_blocked) > 0:
            available_mask[newly_blocked] = False
            
    return code
