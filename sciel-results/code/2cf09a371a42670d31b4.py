import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 17
    D = 6

    def gf_pow(x, p):
        """Compute x^p in GF(2^4) where p is the Frobenius exponent."""
        result = 0
        for i in range(4):
            if (p >> i) & 1:
                # result = result * x^(2^i) but we're doing Frobenius
                # Actually for GF(2^4), x^(2^i) is computed by squaring i times
                # Let me just compute x^p directly
                pass
        # Simpler: just compute x^p by repeated multiplication
        # But for the quadratic residue code, we need specific structure
        # Let me use a different approach
        result = 1
        base = x
        exp = p
        while exp:
            if exp & 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            exp >>= 1
        return result

    def gf_mul(a, b):
        """Multiply in GF(2^4) with polynomial x^4 + x + 1."""
        result = 0
        for i in range(4):
            if (b >> i) & 1:
                result ^= a
            a = gf_shift(a)
        return result

    def gf_shift(a):
        """Multiply by x in GF(2^4) with polynomial x^4 + x + 1."""
        a <<= 1
        if a & 0x10:
            a ^= 0x13  # x^4 + x + 1 = 10011 in binary
        return a & 0x0F

    def construct_qr_code():
        """
        Construct the [17,9,6] quadratic residue code.
        The QR code of length 17 is based on the quadratic residues mod 17.
        Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
        That's 8 residues. The QR code has dimension (17-1)/2 = 8, so it's [17,8,d].
        But we want [18,9,6] shortened to [17,9,6].
        
        Actually, let me use a different approach. The [18,9,6] code is the
        extended QR code. Let me construct it directly.
        
        For the binary QR code of length 17:
        The generator polynomial is g(x) = product of (x - alpha^i) for i in QR
        where alpha is a primitive 17th root of unity.
        
        Actually, let me just use a known generator matrix or construct it
        using the parity check matrix approach.
        
        A simpler approach: The [17,9,6] code can be constructed as follows.
        The extended [18,9,6] code has a generator matrix. Let me use a
        known construction.
        
        Actually, the simplest approach for the QR code:
        The binary QR code of length p=17 has generator polynomial
        g(x) = prod_{i in QR} (x - alpha^i) where alpha is primitive 17th root.
        
        Let me just hardcode a valid [17,9,6] code or use a different method.
        
        Alternative: Use the fact that the [18,9,6] code is the even-weight
        subcode of the [18,10,4] code... no that's not right.
        
        Let me try a direct construction. The quadratic residues mod 17 are:
        QR = {1, 2, 4, 8, 9, 13, 15, 16}
        
        The generator matrix of the [17,8,d] QR code can be constructed,
        and the [17,9,6] code is obtained by adding the all-ones vector
        (since 17 is odd, the QR code doesn't contain the all-ones vector,
        and adding it gives a [17,9,d'] code).
        
        Actually, I think the cleanest approach is to just use a known
        generator matrix. Let me construct the code using the parity
        check matrix of the dual code.
        
        For now, let me use a simpler approach: construct a good seed
        using the RM code and extend it. The key insight from the inspirations
        is that we need to start from a large seed (like 256 words from the
        linear code) and then do local search to add more.
        
        Let me try to construct the [17,9,6] code properly.
        """
        # Quadratic residues mod 17
        QR = [1, 2, 4, 8, 9, 13, 15, 16]
        
        # The generator polynomial of the QR code
        # We need a primitive 17th root of unity in some extension field
        # This is getting complex. Let me use a different approach.
        
        # Actually, let me just use a known valid [17,9,6] code.
        # The code can be described by its parity check matrix.
        # For the [17,9,6] code, the parity check matrix H is 8x17.
        
        # Let me try yet another approach: use the fact that the [18,9,6]
        # extended QR code has a simple structure.
        
        # The extended QR code of length 18 has generator matrix:
        # [I_9 | P] where P is a 9x9 matrix
        # The parity check matrix is [P^T | I_9]
        
        # Actually, I'll use a computational approach. Let me construct
        # the code by finding all vectors in the null space of a parity
        # check matrix.
        
        # For the QR code, the parity check matrix can be constructed
        # from the quadratic residues. Let me use a known construction.
        
        # The [17,8,d] QR code has parity check matrix where row i
        # corresponds to alpha^(i*17/2) or something similar.
        
        # Let me just use a practical approach: construct a large seed
        # using a combination of the RM code and random extension, then
        # use simulated annealing to add more words.
        
        return None  # Placeholder

    def rm_code_16():
        """Generate 32-word first-order Reed-Muller code on 16 coords."""
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    def extended_rm_code():
        """
        Extend the 16-bit RM code to 17 bits by adding a parity bit.
        This gives a [17,5,8] code (or similar) with 32 words.
        Actually, the RM(1,4) code is [16,5,8] with 32 codewords.
        Adding a parity bit gives [17,5,8] if the original has even weight,
        or [17,6,6] or similar.
        """
        rm16 = rm_code_16()
        # Add parity bit to make even weight
        extended = []
        for w in rm16:
            parity = w.bit_count() % 2
            extended.append(w | (parity << 16))
        return extended

    def greedy_extend_fast(seed_words, rng, time_limit):
        """
        Fast greedy extension using numpy for vectorized distance computation.
        """
        current = np.array(seed_words, dtype=np.int32)
        n = len(current)
        
        while time.time() < time_limit:
            # Generate a batch of random candidates
            batch_size = 2048
            candidates = rng.integers(0, 1 << N, size=batch_size, dtype=np.int32)
            
            # Compute distances from each candidate to all current codewords
            # Using vectorized XOR and bit count
            best_cand = None
            best_score = 0
            
            # Process in chunks to avoid memory issues
            chunk_size = 256
            for i in range(0, batch_size, chunk_size):
                chunk = candidates[i:i+chunk_size]
                
                # Compute XOR with each current codeword
                # This is O(chunk_size * n) which can be large
                # Let's use a more efficient approach
                
                min_dists = np.full(len(chunk), N + 1, dtype=np.int32)
                
                for w in current:
                    xors = chunk ^ w
                    d = np.array([x.bit_count() for x in xors], dtype=np.int32)
                    min_dists = np.minimum(min_dists, d)
                
                # Find candidates with min_dist >= D
                valid = np.where(min_dists >= D)[0]
                
                if len(valid) > 0:
                    # Pick the one with maximum min_dist
                    best_idx = valid[np.argmax(min_dists[valid])]
                    if min_dists[best_idx] > best_score:
                        best_score = min_dists[best_idx]
                        best_cand = chunk[best_idx]
            
            if best_cand is not None:
                current = np.append(current, best_cand)
            else:
                # No improvement in this batch; try mutation-based candidates
                if len(current) > 50:
                    # Mutate existing codewords
                    for _ in range(512):
                        base_idx = rng.integers(0, len(current))
                        base = current[base_idx]
                        # Flip 3-5 random bits
                        num_flips = rng.integers(3, 6)
                        bits_to_flip = rng.choice(N, size=num_flips, replace=False)
                        mask = 0
                        for b in bits_to_flip:
                            mask |= (1 << b)
                        candidate = int(base) ^ mask
                        
                        # Check if valid
                        valid = True
                        for w in current:
                            if (candidate ^ int(w)).bit_count() < D:
                                valid = False
                                break
                        if valid:
                            current = np.append(current, candidate)
                            break
                else:
                    # Keep trying with new random candidates
                    pass
        
        return current.tolist()

    def simulated_annealing_extend(seed_words, rng, time_limit):
        """
        Simulated annealing approach to extend the code.
        Start with seed words, try to add new words, and if conflicts arise,
        try to repair by modifying existing codewords.
        """
        current = list(seed_words)
        n = len(current)
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.1
        cooling_rate = 0.999
        
        T = T_start
        no_improve_count = 0
        
        while time.time() < time_limit and T > T_end:
            T *= cooling_rate
            
            # Try to add a random word
            candidate = rng.randint(0, (1 << N) - 1)
            
            # Check if it's valid
            min_d = N + 1
            for w in current:
                d = (candidate ^ w).bit_count()
                if d < min_d:
                    min_d = d
            
            if min_d >= D:
                # Valid addition
                current.append(candidate)
                no_improve_count = 0
            else:
                # Try to repair: find the closest codeword and try to move it
                if no_improve_count < 100:
                    no_improve_count += 1
                    
                    # Find the codeword with minimum distance to candidate
                    min_d = N + 1
                    min_idx = -1
                    for i, w in enumerate(current):
                        d = (candidate ^ w).bit_count()
                        if d < min_d:
                            min_d = d
                            min_idx = i
                    
                    if min_idx >= 0 and min_d < D:
                        # Try to modify current[min_idx] to increase distance
                        # Flip 1-3 random bits in the conflicting codeword
                        for _ in range(20):
                            new_w = current[min_idx]
                            num_flips = rng.randint(1, 3)
                            bits_to_flip = rng.sample(range(N), num_flips)
                            mask = 0
                            for b in bits_to_flip:
                                mask |= (1 << b)
                            new_w ^= mask
                            
                            # Check if new_w is valid with respect to all other codewords
                            valid = True
                            for j, w in enumerate(current):
                                if j == min_idx:
                                    continue
                                if (new_w ^ w).bit_count() < D:
                                    valid = False
                                    break
                            
                            # Also check that new_w is far enough from candidate
                            if valid and (new_w ^ candidate).bit_count() >= D:
                                # Accept the modification
                                current[min_idx] = new_w
                                no_improve_count = 0
                                break
        
        return current

    # Generate initial seeds
    rng = random.Random(42)
    
    # Start with the extended RM code (32 words)
    base_seed = extended_rm_code()
    
    best_result = list(base_seed)
    best_size = len(base_seed)
    
    # Try multiple strategies
    strategies = [
        ("greedy", lambda seed, rng, tl: greedy_extend_fast(seed, rng, tl)),
        ("sa", lambda seed, rng, tl: simulated_annealing_extend(seed, rng, tl)),
    ]
    
    # Run each strategy with a time budget
    time_per_strategy = 15.0
    num_restarts = 2
    
    for strategy_name, strategy_fn in strategies:
        for restart in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Create a fresh seed for this restart
            if restart == 0:
                seed = list(base_seed)
            else:
                # Perturb the seed
                seed = list(base_seed)
                for _ in range(10):
                    idx = rng.randint(0, len(seed) - 1)
                    bits_to_flip = rng.sample(range(N), rng.randint(1, 3))
                    mask = 0
                    for b in bits_to_flip:
                        mask |= (1 << b)
                    seed[idx] ^= mask
            
            time_limit = time.time() + time_per_strategy
            result = strategy_fn(seed, rng, time_limit)
            
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
    
    # Final greedy pass to squeeze out a few more words
    if time.time() < deadline - 1.0:
        final_result = greedy_extend_fast(best_result, rng, deadline - 1.0)
        if len(final_result) > best_size:
            best_result = final_result
    
    # Ensure all words are distinct and valid
    # Remove duplicates
    seen = set()
    unique_result = []
    for w in best_result:
        if w not in seen:
            seen.add(w)
            unique_result.append(w)
    
    # Verify pairwise distances
    valid_result = []
    for w in unique_result:
        valid = True
        for v in valid_result:
            if (w ^ v).bit_count() < D:
                valid = False
                break
        if valid:
            valid_result.append(w)
    
    # Ensure non-empty
    if not valid_result:
        valid_result = [0]
    
    return valid_result
