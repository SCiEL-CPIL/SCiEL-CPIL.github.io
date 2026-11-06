import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Construct the 256-word [17, 9, 6] code
    poly = 0x11D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    
    if beta is None:
        # Fallback to seed (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial G(x)
    poly_coeffs = [1]
    current_beta = beta
    for _ in range(8):
        new_coeffs = [0] * (len(poly_coeffs) + 1)
        for i in range(len(poly_coeffs)):
            new_coeffs[i+1] = poly_coeffs[i]
        for i in range(len(poly_coeffs)):
            new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
        poly_coeffs = new_coeffs
        current_beta = gf_mul(current_beta, current_beta)
        
    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        g_bits[i] = 1 if c == 1 else 0
        
    # Generate 256 codewords
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)
        
    # Extend and shorten
    shortened_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))
            
    code_set = set(shortened_codewords)
    if len(code_set) < 256:
        return list(code_set)
        
    code_list = list(code_set)
    
    # 2. Randomized Local Search with Simulated Annealing
    # Strategy: Try to add random words, and if they conflict, attempt to repair
    # by flipping bits in existing codewords to resolve conflicts.
    
    used = set(code_list)
    
    # Precompute popcount for fast Hamming distance
    pop17 = np.array([bin(i).count('1') for i in range(131072)], dtype=np.uint8)
    
    best_size = len(code_list)
    rng = np.random.RandomState(42)
    
    # Simulated annealing parameters
    T_initial = 1.0
    T_final = 0.001
    cooling_rate = 0.999
    max_iters = 50000
    
    # Track best solution found
    best_code = list(code_list)
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        T = T_initial * (cooling_rate ** iteration)
        
        # Try to add a random candidate word
        candidate = int(rng.randint(2**17))
        
        if candidate in used:
            continue
        
        # Check if candidate is compatible with all existing codewords
        conflicts = []
        for w in code_list:
            dist = (candidate ^ w).bit_count()
            if dist < 6:
                conflicts.append((w, dist))
        
        if not conflicts:
            # No conflicts - add the word
            code_list.append(candidate)
            used.add(candidate)
            if len(code_list) > best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        else:
            # Conflicts exist - try to repair by modifying existing codewords
            # For each conflicting word, try flipping 1-3 bits to resolve
            repaired = False
            for conflict_word, dist in conflicts:
                # Try to find a modification of conflict_word that resolves the conflict
                # while maintaining distance >= 6 from all other codewords
                for flip_mask in range(1, 1 << 17):
                    if flip_mask.bit_count() > 3:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    new_word = conflict_word ^ flip_mask
                    
                    if new_word == candidate:
                        continue
                    
                    # Check if new_word is compatible with candidate
                    if (new_word ^ candidate).bit_count() < 6:
                        continue
                    
                    # Check if new_word is compatible with all other codewords
                    compatible = True
                    for other in code_list:
                        if other == conflict_word:
                            continue
                        if (new_word ^ other).bit_count() < 6:
                            compatible = False
                            break
                    
                    if compatible and new_word not in used:
                        # Replace conflict_word with new_word
                        code_list[code_list.index(conflict_word)] = new_word
                        used.remove(conflict_word)
                        used.add(new_word)
                        repaired = True
                        break
                
                if repaired:
                    # Now try to add the candidate again
                    still_conflicts = False
                    for w in code_list:
                        if (candidate ^ w).bit_count() < 6:
                            still_conflicts = True
                            break
                    
                    if not still_conflicts:
                        code_list.append(candidate)
                        used.add(candidate)
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                    break
            
            if not repaired and T > T_final:
                # With probability exp(-1/T), accept a random perturbation to escape local optima
                if rng.random() < np.exp(-1.0 / max(T, 1e-10)):
                    # Randomly modify a random codeword by flipping 1-2 bits
                    idx = int(rng.randint(len(code_list)))
                    flip_count = int(rng.randint(1, 3))
                    bits = list(range(17))
                    rng.shuffle(bits)
                    flip_mask = 0
                    for b in bits[:flip_count]:
                        flip_mask |= (1 << b)
                    
                    old_word = code_list[idx]
                    new_word = old_word ^ flip_mask
                    
                    # Check if this modification is valid (maintains distance >= 6)
                    valid = True
                    for other in code_list:
                        if other == old_word:
                            continue
                        if (new_word ^ other).bit_count() < 6:
                            valid = False
                            break
                    
                    if valid and new_word not in used:
                        code_list[idx] = new_word
                        used.remove(old_word)
                        used.add(new_word)
    
    # Use the best solution found
    code_list = best_code
    
    # Final verification: ensure all pairs have distance >= 6
    n = len(code_list)
    valid = True
    for i in range(n):
        if time.time() > deadline - 1:
            break
        for j in range(i+1, n):
            if (code_list[i] ^ code_list[j]).bit_count() < 6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to the initial 256-word code if verification fails
        return list(code_set)
    
    return code_list
