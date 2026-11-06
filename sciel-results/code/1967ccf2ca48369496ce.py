import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    
    # --- Step 1: Construct the [17, 8, 6] QR Code ---
    # GF(2^8) arithmetic with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    PRIM_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find a primitive 17th root of unity alpha in GF(2^8)
    # alpha^17 = 1, alpha != 1
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            # Check if it's primitive (order 17)
            if gf_pow(a, 1) != 1:
                alpha = a
                break
                
    if alpha is None:
        # Fallback if no alpha found (should not happen)
        alpha = 3 # Try a known one or just fail gracefully later

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Construct generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    # We work with polynomial coefficients as lists [c0, c1, ..., ck]
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        # Multiply current g(x) by (x + root)
        # (x + root) is represented as [root, 1]
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            # c_i * root
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            # c_i * 1 * x
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert g_coeffs to integer representation
    # g_coeffs[0] is constant term, g_coeffs[8] is x^8 term
    g_int = 0
    for i, c in enumerate(g_coeffs):
        if c == 1: # In GF(2^8), coefficients are in GF(2) for the binary code? 
                   # Wait. The code is binary. The generator polynomial is in GF(2)[x].
                   # The roots are in GF(2^8). The coefficients of g(x) must be in GF(2).
                   # Let's verify if g_coeffs are 0 or 1.
                   # If not, the construction is flawed for a binary code.
                   # The QR code is binary, so g(x) is in GF(2)[x].
                   # The product of minimal polynomials of conjugates is in GF(2)[x].
                   # Q contains all conjugates?
                   # Conjugates of alpha^1 are alpha^(1*2^k).
                   # 1, 2, 4, 8, 16, 15, 13, 9.
                   # Yes, Q is the cyclotomic coset of 1 mod 17.
                   # So the product is the minimal polynomial of alpha^1, which is in GF(2)[x].
                   # So coefficients should be 0 or 1.
            g_int |= (1 << i)
            
    # Generate all 256 codewords
    # c(x) = m(x) * g(x) mod (x^17 - 1)
    # Since deg(m) <= 8 and deg(g) = 8, deg(c) <= 16. No reduction needed if we just multiply.
    # Actually, c(x) is a multiple of g(x).
    # We can just compute m(x) * g(x).
    
    base_code = []
    for m in range(256):
        c = 0
        # m(x) * g(x)
        # This is a convolution
        # But since we are in GF(2), it's XOR of shifted g_int
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        base_code.append(c)
        
    # Ensure uniqueness (should be 256 distinct words)
    base_code = list(set(base_code))
    
    # --- Step 2: Vectorized Local Search ---
    
    # Convert to numpy array for fast operations
    code_array = np.array(base_code, dtype=np.uint32)
    
    # Helper to add words to the code array
    def add_words(new_words):
        nonlocal code_array
        if len(new_words) == 0:
            return
        new_arr = np.array(new_words, dtype=np.uint32)
        code_array = np.concatenate([code_array, new_arr])
        
    # Precompute unpackbits for speed? 
    # np.unpackbits is slow. 
    # Alternative: Use a lookup table for 8-bit popcount?
    # Or use np.sum((x >> i) & 1) which is also slow.
    # Let's try a lookup table approach for 16-bit or 8-bit.
    
    # Create a lookup table for popcount of 0..255
    pop_lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def fast_popcount(arr):
        # arr is uint32
        # Split into 4 bytes
        b0 = (arr >> 0) & 0xFF
        b1 = (arr >> 8) & 0xFF
        b2 = (arr >> 16) & 0xFF
        b3 = (arr >> 24) & 0xFF
        return pop_lut[b0] + pop_lut[b1] + pop_lut[b2] + pop_lut[b3]

    # We want to find words w such that dist(w, c) >= 6 for all c in code.
    # dist(w, c) = popcount(w ^ c).
    
    rng = random.Random(42)
    
    # Strategy: Generate batches of candidates.
    # Candidates can be random, or structured.
    # Structured: Weight 8 or 9 vectors?
    # The linear code has min weight 6.
    # Nonlinear extensions often involve adding words of intermediate weight.
    
    max_batch_size = 10000
    iterations = 0
    
    while time.time() < deadline - 5.0:
        iterations += 1
        
        # Generate candidates
        # Mix of random and specific weights
        # Let's try random first, then specific weights if stuck.
        
        # Heuristic: If we haven't added many words recently, try higher weights?
        # For now, just random 17-bit integers.
        
        # To improve efficiency, we can filter candidates by weight.
        # Words with weight < 6 cannot be in the code (dist 0 < 6 to 0? No, 0 is in code. 
        # If w has weight < 6, dist(w, 0) < 6. So w cannot be added.
        # So candidates must have weight >= 6.
        # Also, if w has weight > 11, dist(w, 0) > 11, which is fine, 
        # but dist(w, c) might be small for some c.
        
        # Let's generate random candidates with weight between 6 and 11.
        # This is a good heuristic because the average weight in the code is around 8.5.
        
        candidates = []
        for _ in range(max_batch_size):
            w = rng.randint(6, 11)
            pos = rng.sample(range(N), w)
            val = 0
            for p in pos:
                val |= (1 << p)
            candidates.append(val)
            
        cand_arr = np.array(candidates, dtype=np.uint32)
        
        # Check conflicts
        # We need to check dist(cand, c) >= 6 for all c in code_array.
        # Vectorized:
        # dist_matrix[i, j] = popcount(cand_arr[i] ^ code_array[j])
        
        # This creates a matrix of size (10000, len(code)).
        # If len(code) is 256, matrix is 10000x256 = 2.5M elements.
        # Popcount on 2.5M uint32s.
        
        # To avoid huge memory spikes, we can process in chunks if code gets large.
        # But 256 is small.
        
        # Compute XOR
        # cand_arr[:, None] ^ code_array[None, :]
        
        # Optimization: 
        # We can early terminate if we find a conflict.
        # But NumPy is eager.
        
        # Let's do it in chunks of candidates to keep memory bounded and allow time checks.
        chunk_size = 1000
        valid_cands = []
        
        for i in range(0, len(cand_arr), chunk_size):
            if time.time() > deadline - 2.0:
                break
                
            chunk = cand_arr[i:i+chunk_size]
            
            # Compute distances
            # xor_chunk = chunk[:, None] ^ code_array[None, :]
            # dists = fast_popcount(xor_chunk)
            # valid_mask = np.all(dists >= MIN_DIST, axis=1)
            
            # To speed up, we can check against a subset of the code first?
            # No, must check all.
            
            # Let's use the fast_popcount
            # Note: fast_popcount expects uint32.
            
            # We can vectorize the XOR and then popcount.
            # However, creating the 2D array is the bottleneck.
            
            # Alternative: Loop over code words?
            # No, that's slow.
            
            # Let's try the vectorized approach.
            
            # Memory: 1000 * 256 * 4 bytes = 1MB. Fine.
            
            xor_mat = chunk[:, None] ^ code_array[None, :]
            
            # Apply popcount lookup
            b0 = (xor_mat >> 0) & 0xFF
            b1 = (xor_mat >> 8) & 0xFF
            b2 = (xor_mat >> 16) & 0xFF
            b3 = (xor_mat >> 24) & 0xFF
            
            dists = pop_lut[b0] + pop_lut[b1] + pop_lut[b2] + pop_lut[b3]
            
            # Check if all distances are >= 6
            valid_mask = np.all(dists >= MIN_DIST, axis=1)
            
            if np.any(valid_mask):
                valid_cands.extend(chunk[valid_mask].tolist())
                
        if valid_cands:
            add_words(valid_cands)
            
        # If we are stuck (no new words added in last few iterations), 
        # we might need a different strategy.
        # For now, we just keep trying randoms.
        
    return code_array.tolist()

# Run once to check for errors in local env? No, just provide the code.
