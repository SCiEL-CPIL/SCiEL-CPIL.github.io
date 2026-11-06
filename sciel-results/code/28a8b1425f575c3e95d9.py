import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # GF(2^8) arithmetic
    PRIM_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    # Find alpha (primitive element)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
            
    if alpha is None:
        return greedy_construct(deadline)
    
    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Calculate generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert to binary polynomial coefficients
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)
        
    # Convert bin_g to integer for fast shifting
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    # Generate all 512 codewords of the [17,9,6] code
    base_code = []
    
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        
        # Check even weight
        if c.bit_count() % 2 == 0:
            base_code.append(c)
            
    code = base_code[:]
    code_set = set(code)
    
    rng = random.Random(42)
    
    # Build fast lookup: for each bit position, list of codeword indices having 1 at that position
    # This allows quick "conflict" checking: a candidate word is incompatible if any codeword
    # differs in fewer than 6 positions, i.e., agrees in more than 11 positions.
    
    # More efficient: precompute pairwise incompatibility as a bitset
    # A word w is "blocked" by codeword c if Hamming(w,c) < 6, i.e., they share >= 12 ones in the XOR=0 sense
    # Actually: dist(w,c) = popcount(w^c). We need dist >= 6 for all c in code.
    # So w is blocked if exists c with popcount(w^c) <= 5.
    
    # Precompute: for each possible 17-bit word w, we can't store all 2^17.
    # Instead, maintain a "conflict count" array indexed by... hmm.
    
    # Alternative: use a blocked set approach with bit positions.
    # For each codeword c in code, all words within distance 5 of c are blocked.
    # But enumerating all such words is expensive.
    
    # Better approach: maintain for each bit pattern of length up to 5, count how many codewords
    # would conflict. Actually, let's use a different strategy.
    
    # Strategy: maintain a "distance-5 neighborhood" count using inclusion.
    # For a candidate w, w conflicts with c iff popcount(w ^ c) <= 5.
    # This means w is in the union of radius-5 balls around each codeword.
    
    # Efficient check: for candidate w, compute popcount(w ^ c) for each c in code.
    # With 256+ codewords, this is ~256 popcounts per candidate. For 1M candidates, that's 256M ops.
    # In Python, too slow. In numpy with vectorized XOR and popcount, much faster.
    
    # Let's use numpy for the code matrix and vectorized distance computation.
    
    # Build the code matrix as uint8 array, shape (len(code), N)
    code_matrix = np.array([[(c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
    
    def can_add_fast(cand):
        """Check if cand has min distance >= MIN_DIST from all codewords using numpy."""
        if len(code_matrix) == 0:
            return True
        cand_bits = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
        diffs = code_matrix ^ cand_bits
        dists = diffs.sum(axis=1)
        return bool(np.min(dists) >= MIN_DIST)
    
    # Phase 1: Greedy addition from candidate pool
    # Generate candidates by weight, starting from most likely to fit
    candidate_pool = []
    for w in range(6, 13):
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            if c not in code_set:
                candidate_pool.append(c)
    
    rng.shuffle(candidate_pool)
    
    added_count = 0
    for idx, cand in enumerate(candidate_pool):
        if time_left() < 8.0:
            break
        if cand in code_set:
            continue
        if can_add_fast(cand):
            code.append(cand)
            code_set.add(cand)
            new_row = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_matrix = np.vstack([code_matrix, new_row])
            added_count += 1
    
    # Phase 2: Iterative improvement with multiple restarts of random search
    best_size = len(code)
    best_code = code[:]
    best_code_set = code_set.copy()
    best_matrix = code_matrix.copy()
    
    for restart in range(8):
        if time_left() < 10.0:
            break
        
        # Start from best found so far, try random additions
        current_code = best_code[:]
        current_set = best_code_set.copy()
        current_matrix = best_matrix.copy()
        
        improved_this_restart = True
        while improved_this_restart and time_left() > 8.0:
            improved_this_restart = False
            # Try 500 random candidates
            for _ in range(500):
                if time_left() < 8.0:
                    break
                w = rng.randint(6, 12)
                positions = rng.sample(range(N), w)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
                
                if candidate in current_set:
                    continue
                
                cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
                diffs = current_matrix ^ cand_bits
                dists = diffs.sum(axis=1)
                
                if len(dists) == 0 or np.min(dists) >= MIN_DIST:
                    current_code.append(candidate)
                    current_set.add(candidate)
                    current_matrix = np.vstack([current_matrix, cand_bits])
                    improved_this_restart = True
                    if len(current_code) > best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        best_code_set = current_set.copy()
                        best_matrix = current_matrix.copy()
        
        # Also try swap-based improvement: remove one, add one or two
        if time_left() > 15.0:
            for _ in range(100):
                if time_left() < 8.0:
                    break
                
                rem_idx = rng.randint(0, len(current_code) - 1)
                removed = current_code[rem_idx]
                
                # Try to find replacement
                for _ in range(200):
                    if time_left() < 8.0:
                        break
                    
                    w = rng.randint(6, 12)
                    positions = rng.sample(range(N), w)
                    candidate = 0
                    for p in positions:
                        candidate |= (1 << p)
                    
                    if candidate in current_set or candidate == removed:
                        continue
                    
                    # Temporarily remove
                    current_set.remove(removed)
                    mask = np.ones(len(current_code), dtype=bool)
                    mask[rem_idx] = False
                    temp_matrix = current_matrix[mask]
                    
                    cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
                    diffs = temp_matrix ^ cand_bits
                    dists = diffs.sum(axis=1)
                    
                    if len(dists) == 0 or np.min(dists) >= MIN_DIST:
                        # Can add candidate after removing
                        # Check if we can also add another word
                        temp_matrix2 = np.vstack([temp_matrix, cand_bits])
                        current_set.add(candidate)
                        
                        for _ in range(30):
                            if time_left() < 8.0:
                                break
                            w2 = rng.randint(6, 12)
                            pos2 = rng.sample(range(N), w2)
                            cand2 = 0
                            for p in pos2:
                                cand2 |= (1 << p)
                            if cand2 in current_set or cand2 == removed:
                                continue
                            cb2 = np.array([(cand2 >> i) & 1 for i in range(N)], dtype=np.uint8)
                            d2 = temp_matrix2 ^ cb2
                            dist2 = d2.sum(axis=1)
                            if len(dist2) == 0 or np.min(dist2) >= MIN_DIST:
                                # Net gain of 1
                                new_code = current_code[:]
                                new_code[rem_idx] = candidate
                                new_code.append(cand2)
                                new_set = current_set.copy()
                                new_set.add(cand2)
                                new_matrix = np.vstack([temp_matrix2, cb2])
                                
                                if len(new_code) > best_size:
                                    best_size = len(new_code)
                                    best_code = new_code
                                    best_code_set = new_set
                                    best_matrix = new_matrix
                                # Also update current for further searches
                                current_code = new_code
                                current_set = new_set
                                current_matrix = new_matrix
                                break
                        else:
                            # No second word found, revert candidate addition
                            current_set.remove(candidate)
                        continue
                    
                    # Revert
                    current_set.add(removed)
                    break
    
    # Phase 3: Final greedy pass on best code
    if time_left() > 5.0:
        code = best_code[:]
        code_set = best_code_set.copy()
        code_matrix = best_matrix.copy()
        
        for _ in range(2000):
            if time_left() < 5.0:
                break
            w = rng.randint(6, 12)
            positions = rng.sample(range(N), w)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            
            if candidate in code_set:
                continue
            
            cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
            diffs = code_matrix ^ cand_bits
            dists = diffs.sum(axis=1)
            
            if len(dists) == 0 or np.min(dists) >= MIN_DIST:
                code.append(candidate)
                code_set.add(candidate)
                code_matrix = np.vstack([code_matrix, cand_bits])
    
    # Phase 4: Try to add words of weight 5 or 13-17 (unlikely but possible)
    if time_left() > 5.0:
        for w in [5, 13, 14, 15, 16, 17]:
            if time_left() < 5.0:
                break
            for combo in combinations(range(N), w):
                if time_left() < 5.0:
                    break
                c = 0
                for p in combo:
                    c |= (1 << p)
                if c in code_set:
                    continue
                cand_bits = np.array([(c >> i) & 1 for i in range(N)], dtype=np.uint8)
                diffs = code_matrix ^ cand_bits
                dists = diffs.sum(axis=1)
                if len(dists) == 0 or np.min(dists) >= MIN_DIST:
                    code.append(c)
                    code_set.add(c)
                    code_matrix = np.vstack([code_matrix, cand_bits])
    
    return code

def greedy_construct(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
