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
    
    # We need the generator polynomial for the binary QR code of length 17.
    # This is the minimal polynomial of a primitive 17th root of unity over GF(2).
    # It must be an irreducible polynomial of degree 8 dividing x^17 + 1 in GF(2)[x].
    
    def poly_divides(p, q):
        # Check if p divides q in GF(2)[x]
        # p, q are integers representing polynomials (bit i is coeff of x^i)
        # deg(p) <= deg(q)
        deg_p = p.bit_length() - 1
        deg_q = q.bit_length() - 1
        if deg_p > deg_q:
            return False
        if p == 0:
            return False
        r = q
        while r.bit_length() - 1 >= deg_p and r != 0:
            shift = (r.bit_length() - 1) - deg_p
            r ^= (p << shift)
        return r == 0

    # Find an irreducible polynomial of degree 8 dividing x^17 + 1
    # x^17 + 1 in GF(2) is represented by (1 << 17) | 1
    target_poly = (1 << 17) | 1
    gen_poly = None
    
    # Iterate through all monic polynomials of degree 8 with constant term 1
    # There are 2^6 = 64 such polynomials.
    for mid_bits in range(1 << 6):
        # Construct polynomial: x^8 + ... + 1
        # Bits 7..1 are from mid_bits (shifted by 1), Bit 0 is 1, Bit 8 is 1
        p = (1 << 8) | (mid_bits << 1) | 1
        
        # Check if p is irreducible? 
        # Actually, if it divides x^17+1 and is not (x+1) or (x^2+x+1) etc, 
        # and has degree 8, it is likely the minimal polynomial we want.
        # The cyclotomic coset size is 8, so the minimal polynomial has degree 8.
        # Any irreducible factor of x^17+1 of degree 8 is a valid generator for the QR code.
        
        if poly_divides(p, target_poly):
            # Ensure it's not x+1 (degree 1) or other small factors.
            # Degree is 8 by construction.
            gen_poly = p
            break
            
    if gen_poly is None:
        # Fallback: Use a known valid generator if search fails (should not happen)
        # One known generator for [17,9,5] QR is 0b100110111 (x^8+x^5+x^3+x^2+x+1? No, let's trust the search)
        # If search fails, we return a small random code to avoid crash
        initial_code = [0]
        return initial_code

    # Generate basis vectors for the [17,9,5] code
    # Basis vector k is x^k * g(x) mod (x^17 - 1)
    # Since deg(g) = 8, for k=0..8, x^k*g(x) has deg <= 16, so no wrap needed for basis generation
    # We just need the 9 basis words.
    basis = []
    for k in range(9):
        # x^k * g(x)
        val = gen_poly << k
        # Mask to 17 bits (though deg is <= 16, so it fits)
        val &= (1 << N) - 1
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
    # The [17,9,5] code is shortened from the [18,9,6] extended code.
    # The extension bit is the parity of the 17 bits.
    # Shortening means we require the extension bit to be 0.
    # So we keep words with even parity (even number of 1s in the 17 bits).
    
    initial_code = []
    for w in qr_code_words:
        if bin(w).count('1') % 2 == 0:
            initial_code.append(w)
            
    # initial_code should have 256 words
    
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
