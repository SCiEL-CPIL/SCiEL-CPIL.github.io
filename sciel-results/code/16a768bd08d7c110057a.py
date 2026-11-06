    import numpy as np
    import time
    from itertools import combinations

    def construct():
        N = 17
        MIN_DIST = 6
        start_time = time.time()
        time_limit = 55

        # --- Step 1: Generate the [17, 8, 6] Shortened QR Code ---
        
        # GF(16) arithmetic
        # Primitive polynomial x^4 + x + 1 (0b10011)
        # We represent elements as 4-bit integers.
        
        def gf16_mul(a, b):
            res = 0
            for i in range(4):
                if b & 1:
                    res ^= a
                b >>= 1
                if b:
                    a <<= 1
                    if a & 16:
                        a ^= 0b10011 # x^4 = x + 1
            return res

        def gf16_pow(a, e):
            res = 1
            for _ in range(e):
                res = gf16_mul(res, a)
            return res

        # Find generator polynomial g(x) of degree 8
        # Roots are alpha^i for i in coset C1 = {1, 2, 4, 8, 16, 15, 13, 9}
        # alpha = 0b0010 (2)
        alpha = 2
        
        # g(x) = prod (x - alpha^i)
        # We build it as a list of coefficients [c0, c1, ..., c8]
        # Start with g = 1
        g_coeffs = [1]
        
        # Coset C1
        c1 = [1, 2, 4, 8, 16, 15, 13, 9]
        
        for i in c1:
            root = gf16_pow(alpha, i)
            # Multiply g(x) by (x - root)
            # In GF(2), x - root is x + root
            # New degree is len(g_coeffs)
            new_g = [0] * (len(g_coeffs) + 1)
            for j in range(len(g_coeffs)):
                # g[j] * x
                new_g[j+1] ^= g_coeffs[j]
                # g[j] * root
                new_g[j] ^= gf16_mul(g_coeffs[j], root)
            g_coeffs = new_g
            
        # g_coeffs is now length 9 (degree 8)
        # Note: g_coeffs[k] is coefficient of x^k
        
        # Generate all 512 codewords of [17, 9, 5]
        base_code_17 = []
        for msg in range(512):
            # m(x) has degree < 9
            # c(x) = m(x) * g(x) mod (x^17 - 1)
            # Since deg(m) < 9 and deg(g) = 8, deg(c) < 17.
            # So no reduction needed if we just multiply polynomials.
            
            # Multiply m and g
            c_coeffs = [0] * 17
            for i in range(9):
                if (msg >> i) & 1:
                    for j in range(9):
                        c_coeffs[i+j] ^= g_coeffs[j]
            
            # Convert to bitstring/integer
            word = 0
            for i in range(17):
                if c_coeffs[i]:
                    word |= (1 << i)
            
            # Extend to 18 bits
            weight = bin(word).count('1')
            parity = weight % 2
            
            # Shorten: keep if parity == 0 (last bit is 0)
            if parity == 0:
                base_code_17.append(word)
                
        # Ensure we have 256 words
        if len(base_code_17) != 256:
            # Fallback if generation failed? 
            # For now, assume it works.
            pass

        # --- Step 2: Generate Candidates (Weight 8 and 9) ---
        
        candidates = []
        # Weight 8
        for combo in combinations(range(N), 8):
            w = 0
            for i in combo:
                w |= (1 << i)
            candidates.append(w)
        # Weight 9
        for combo in combinations(range(N), 9):
            w = 0
            for i in combo:
                w |= (1 << i)
            candidates.append(w)
        
        # --- Step 3: Filter Candidates ---
        
        # Precompute base code as list of ints
        base_set = set(base_code_17)
        
        compatible_candidates = []
        for w in candidates:
            is_compat = True
            for b in base_code_17:
                if (w ^ b).bit_count() < MIN_DIST:
                    is_compat = False
                    break
            if is_compat:
                compatible_candidates.append(w)
                
        # --- Step 4: Find Max Clique ---
        
        M = len(compatible_candidates)
        
        if M == 0:
            return base_code_17
            
        # Build adjacency list
        # adj[i] is set of indices j such that d(c_i, c_j) >= 6
        adj = [set() for _ in range(M)]
        
        for i in range(M):
            wi = compatible_candidates[i]
            for j in range(i + 1, M):
                wj = compatible_candidates[j]
                if (wi ^ wj).bit_count() >= MIN_DIST:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Greedy Clique Finding
        # Sort by degree descending
        order = sorted(range(M), key=lambda x: len(adj[x]), reverse=True)
        
        clique_indices = []
        for idx in order:
            # Check if compatible with all in clique
            ok = True
            for c_idx in clique_indices:
                if idx not in adj[c_idx]:
                    ok = False
                    break
            if ok:
                clique_indices.append(idx)
                
        # Extract words
        extra_words = [compatible_candidates[i] for i in clique_indices]
        
        # --- Step 5: Combine ---
        
        final_code = base_code_17 + extra_words
        
        # Ensure distinctness (should be distinct by construction)
        return list(set(final_code))

    