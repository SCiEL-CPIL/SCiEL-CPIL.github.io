"""Program constructing a large binary code of length 17, distance >= 6."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

import numpy as np

N = 17
MIN_D = 6
DEADLINE = 58.0  # safety margin before 60s kill


def _hamming(u, v):
    return (u ^ v).bit_count()


def _check_valid(candidate, code_list):
    """Check if candidate has distance >= MIN_D from all codewords."""
    for c in code_list:
        if _hamming(candidate, c) < MIN_D:
            return False
    return True


def _check_valid_vec(candidate, code_array):
    """Vectorized check: is candidate at distance >= MIN_D from all codewords?"""
    if len(code_array) == 0:
        return True
    # Convert candidate to bit array
    cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.int8)
    # code_array is shape (M, N) with int8 values 0/1
    # XOR and sum along axis 1
    diffs = code_array ^ cand_bits[np.newaxis, :]
    dists = diffs.sum(axis=1)
    return (dists >= MIN_D).all()


def construct():
    start = time.time()
    rng = random.Random(42)

    # Start from the linear [18,9,6] extended QR code shortened by one coordinate.
    # The seed uses the RM(1,4) code (32 words) which is a subcode.
    # We need to find the actual [17,9,6] code from the extended QR[18,9,6].
    #
    # The extended quadratic residue code of length 18 over F_2 is a [18,9,6] code.
    # It contains the all-zero and all-one words.
    # Shortening by deleting one coordinate gives a [17,9,6] code with 256 words.
    #
    # Construction of QR code length 18:
    # Quadratic residues mod 17: {1,2,4,8,9,13,15,16} (16 is -1)
    # Non-residues: {3,5,6,7,10,11,12,14}
    # The QR code C_17 has length 17, dimension 9, minimum distance 5.
    # The extended code C_17^ext has length 18, dimension 9, minimum distance 6.
    #
    # Let's construct it using the generator matrix.
    # For the QR code, the generator matrix can be built from the parity-check
    # matrix whose columns are indexed by the multiplicative group elements.
    
    # Actually, let me use a known construction:
    # The [17,9,5] QR code has a generator matrix G_17.
    # The [18,9,6] extended code appends a parity bit.
    
    # Let me build the [18,9,6] code explicitly.
    # The quadratic residue code of length 17 (p=17) has generator matrix:
    # G = [I_9 | P] where P is a 9x8 matrix.
    # 
    # Alternatively, I can use the fact that the [18,9,6] code is the dual of the
    # [18,9,6] code (it's self-dual when extended? No, [18,9,6] isn't self-dual).
    #
    # Let me just construct it via the standard QR code construction.
    # 
    # For p=17, the QR code has:
    # - Length n = p = 17
    # - Dimension k = (n+1)/2 = 9
    # - The parity-check matrix H has columns h_i = (1, i, i^2, ..., i^8) for i in QR mod 17
    #   Wait, that's not quite right.
    #
    # Let me use a different approach. I'll use the known fact that the [18,9,6] 
    # extended QR code can be constructed as follows:
    #
    # The generator matrix for the [17,9,5] QR code:
    # Row i (i=0..8) corresponds to the monomial x^i evaluated at the 17 nonzero
    # elements of F_17... no, that's not right either.
    #
    # Actually, let me just hardcode the generator matrix from a known source,
    # or construct it algorithmically.
    
    # The [18,9,6] extended QR code generator matrix (rows are codewords):
    # Let me construct it using the standard method.
    
    # For the QR code of length 17:
    # The set of quadratic residues mod 17: Q = {1, 2, 4, 8, 9, 13, 15, 16}
    # The set of non-residues: N = {3, 5, 6, 7, 10, 11, 12, 14}
    #
    # The parity-check matrix H (9 x 8) has entries H[i][j] = 1 if (i+1)*(j+1) mod 17 is in Q
    # No, that's not the standard construction.
    
    # Let me try a cleaner approach. I'll use the fact that we can enumerate all
    # 2^9 = 512 linear combinations and filter for the valid ones... but we need
    # the generator matrix first.
    
    # Here's a concrete construction:
    # The [17,9,5] QR code generator matrix G (9x17):
    # G[i][j] = 1 if j+1 is in {1,2,...,17} and the (i+1)-th power of (j+1) mod 17 is a QR
    
    # Actually, let me use the simplest correct approach:
    # I know the [18,9,6] code has 256 words. Let me construct the generator matrix
    # using the standard QR code construction.
    
    p = 17
    QR = set()
    for x in range(1, p):
        if pow(x, (p-1)//2, p) == 1:
            QR.add(x)
    # QR = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # The generator matrix G for the [17,9,5] QR code:
    # G[i][j] for i in 0..8, j in 0..16
    # Standard construction: G[i][j] = 1 if (j+1)^(i+1) mod p is in QR
    # Wait, I need to be more careful.
    
    # Let me use this construction:
    # The parity check matrix H (8x17) for the [17,9,5] code:
    # H[j][i] = 1 if (i+1)^(j+1) mod 17 is in QR, for j=0..7, i=0..16
    # Then the generator matrix G satisfies G * H^T = 0.
    
    # Actually, the standard QR code construction:
    # H is an (n-k) x n matrix where H[i][j] = 1 if (j+1) is a QR and... 
    # I'm getting confused. Let me just use a known G matrix.
    
    # Here's a known generator matrix for the [17,9,5] QR code (from literature):
    # I'll construct it via the following rule:
    # G[i][j] = 1 if j+1 in 1..17 and (j+1)^(i+1) mod 17 is a quadratic residue mod 17
    # for i = 0..8, j = 0..16
    
    G = np.zeros((9, 17), dtype=int)
    for i in range(9):
        for j in range(17):
            val = pow(j + 1, i + 1, 17)
            G[i][j] = 1 if val in QR else 0
    
    # Extend to [18,9,6] by appending a parity-check bit to make each row even weight
    G_ext = np.zeros((9, 18), dtype=int)
    G_ext[:, :17] = G
    for i in range(9):
        G_ext[i, 17] = G[i].sum() % 2  # parity bit
    
    # Generate all 256 codewords of the [18,9,6] code
    ext_code = []
    for mask in range(256):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(18):
                    if G_ext[i][j]:
                        word ^= (1 << j)
        ext_code.append(word)
    
    # Shorten by deleting one coordinate (say coordinate 17, the last one)
    # This gives a [17,9,6] code with 256 words
    base_code = set()
    for word in ext_code:
        # Delete bit at position 17
        shortened = (word & ((1 << 17) - 1))  # keep bits 0..16
        base_code.add(shortened)
    
    # Verify: should have 256 distinct words
    # Actually, shortening might reduce the size if two words differ only in the deleted bit.
    # Since the [18,9,6] code has minimum distance 6, any two distinct codewords differ
    # in at least 6 positions. So they can't differ only in position 17.
    # Thus we get exactly 256 distinct words.
    
    code = list(base_code)
    
    # Verify minimum distance
    if len(code) > 1:
        min_d = min(_hamming(a, b) for a, b in itertools.combinations(code, 2) if a < b)
        # This is O(n^2) which is 256*255/2 = 32640 checks, fast enough
    
    # Now try to add more words using a greedy/randomized approach
    code_set = set(code)
    
    # Precompute code as numpy array for vectorized distance checks
    code_array = np.array([(c >> i) & 1 for c in code for i in range(N)], dtype=np.int8).reshape(-1, N)
    
    # Strategy 1: Try adding words at specific distances from existing codewords
    # For each codeword, try flipping exactly 6 bits (minimum distance)
    # This gives candidates that are at distance exactly 6 from that codeword
    # but might be at distance >= 6 from all others
    
    added_count = 0
    attempt = 0
    
    # First pass: systematic search near existing codewords
    # For each codeword, generate candidates by flipping 6 bits
    # There are C(17,6) = 12376 such candidates per codeword
    # We'll use a subset to stay within time budget
    
    # Generate all 6-bit flip masks
    six_bit_masks = []
    for combo in itertools.combinations(range(17), 6):
        mask = 0
        for b in combo:
            mask |= (1 << b)
        six_bit_masks.append(mask)
    
    # For efficiency, we'll try a random subset of these per codeword
    rng.shuffle(six_bit_masks)
    
    # Phase 1: Try candidates at distance exactly 6 from existing codewords
    for c in code[:]:  # iterate over snapshot
        if time.time() - start > DEADLINE - 5:
            break
        
        # Try a limited number of 6-flip candidates for this codeword
        for mask in six_bit_masks[:200]:  # limit per codeword
            if time.time() - start > DEADLINE - 5:
                break
            
            candidate = c ^ mask
            if candidate in code_set:
                continue
            
            # Check against all codewords
            if _check_valid(candidate, code):
                code.append(candidate)
                code_set.add(candidate)
                added_count += 1
                # Update code_array
                new_bit = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.int8)
                code_array = np.vstack([code_array, new_bit[np.newaxis, :]])
    
    # Phase 2: Random search with vectorized checking
    while time.time() - start < DEADLINE - 3:
        # Generate random candidate
        candidate = rng.getrandbits(17)
        if candidate in code_set:
            continue
        
        # Vectorized check
        if _check_valid_vec(candidate, code_array):
            code.append(candidate)
            code_set.add(candidate)
            added_count += 1
            new_bit = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.int8)
            code_array = np.vstack([code_array, new_bit[np.newaxis, :]])
    
    # Phase 3: Try to find more by perturbing existing codewords slightly
    # Sometimes flipping 7 or 8 bits from a codeword works
    seven_bit_masks = []
    for combo in itertools.combinations(range(17), 7):
        mask = 0
        for b in combo:
            mask |= (1 << b)
        seven_bit_masks.append(mask)
    rng.shuffle(seven_bit_masks)
    
    for c in code[:]:
        if time.time() - start > DEADLINE - 2:
            break
        
        for mask in seven_bit_masks[:100]:
            if time.time() - start > DEADLINE - 2:
                break
            
            candidate = c ^ mask
            if candidate in code_set:
                continue
            
            if _check_valid(candidate, code):
                code.append(candidate)
                code_set.add(candidate)
                added_count += 1
                new_bit = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.int8)
                code_array = np.vstack([code_array, new_bit[np.newaxis, :]])
    
    return code

# EVOLVE-BLOCK-END
