import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    START_TIME = time.time()
    TIME_LIMIT = 55.0  # Safety margin

    # Fallback: Reed-Muller code (32 words)
    def reed_muller():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Try to find a [18, 9, 6] linear code (512 words)
    K = 9
    NUM_CW = 1 << K  # 512

    def try_find_linear_code():
        # Initialize random generator matrix G (K x N)
        G = [random.getrandbits(N) for _ in range(K)]
        
        # Precompute codewords and weights
        codewords = [0] * NUM_CW
        weights = [0] * NUM_CW
        
        # Generate codewords
        for u in range(1, NUM_CW):
            lsb = u & -u
            i = lsb.bit_length() - 1
            prev = u ^ lsb
            codewords[u] = codewords[prev] ^ G[i]
            weights[u] = codewords[u].bit_count()
        
        best_min_w = min(weights[1:])
        best_G = G.copy()
        best_codewords = codewords.copy()
        
        # Precompute affected indices for fast update? 
        # Actually, scanning all 512 for min is fast enough.
        
        while time.time() - START_TIME < TIME_LIMIT:
            # Pick random row r and col j
            r = random.randint(0, K - 1)
            j = random.randint(0, N - 1)
            
            # Flip G[r][j]
            old_bit = (G[r] >> j) & 1
            G[r] ^= (1 << j)
            
            # Update codewords and weights
            # We need to update all u where bit r is set
            # This is 256 codewords
            # Instead of precomputing lists, we can just loop u where (u>>r)&1
            # But doing this check 512 times is also fast.
            # Let's just update the specific ones.
            
            # Optimization: Only update weights for affected u
            # Then find min of all weights.
            
            # Identify affected u
            # u has bit r set
            # We can iterate through all u and check, or precompute.
            # Precomputing is better.
            
            # Let's precompute affected_u outside the loop
            # But we are inside the function.
            
            # Let's just do a full min scan, it's simpler and 512 is small.
            
            # Update codewords and weights for affected u
            # To avoid iterating all 512 to find affected, we can use the property:
            # Affected u are those where u & (1<<r) != 0.
            
            # We'll just update all affected and then scan all for min.
            
            mask = 1 << r
            for u in range(1, NUM_CW):
                if u & mask:
                    codewords[u] ^= (1 << j)
                    # Weight change:
                    # If old_bit was 1, it becomes 0. Bit in codeword flips 1->0. Weight -1.
                    # If old_bit was 0, it becomes 1. Bit in codeword flips 0->1. Weight +1.
                    delta = -1 if old_bit else 1
                    weights[u] += delta
            
            current_min_w = min(weights[1:])
            
            if current_min_w > best_min_w:
                best_min_w = current_min_w
                best_G = G.copy()
                best_codewords = codewords.copy()
                
                if best_min_w >= MIN_DIST:
                    return best_codewords
            elif current_min_w == best_min_w:
                # Accept to explore
                if random.random() < 0.5:
                    best_min_w = current_min_w
                    best_G = G.copy()
                    best_codewords = codewords.copy()
            else:
                # Revert
                G[r] ^= (1 << j)
                mask = 1 << r
                for u in range(1, NUM_CW):
                    if u & mask:
                        codewords[u] ^= (1 << j)
                        delta = -1 if old_bit else 1
                        weights[u] -= delta

        # If we didn't find a valid code, return None
        return None

    code = try_find_linear_code()
    if code and len(code) > 0:
        return code

    # Fallback to Reed-Muller
    return reed_muller()
