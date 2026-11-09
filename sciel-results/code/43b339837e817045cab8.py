import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 18
    MIN_DIST = 6
    
    # Construct the Extended Quadratic Residue Code [18, 9, 6]
    # The QR(17) code has generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # This gives a [17, 9, 5] code. Extending with parity gives [18, 9, 6].
    
    # Generator polynomial: bits from LSB to MSB: 1,0,0,1,1,1,0,0,1 = 0b100111001 = 313
    g_val = 313
    
    # Build the generator matrix rows: x^i * g(x) mod (x^17 - 1) for i = 0, ..., 8
    # Since deg(g) = 8, x^i * g(x) has degree at most i+8 <= 16 < 17, so no reduction needed
    basis = []
    for i in range(9):
        basis.append(g_val << i)
    
    # Generate all 512 codewords of the [17, 9, 5] QR code
    # Use fast XOR-based generation
    qr_words_17 = [0] * 512
    for m in range(1, 512):
        # Find the highest set bit to use Gray-code-like generation
        # Or simply: word = XOR of basis[j] for each set bit j in m
        word = 0
        mm = m
        j = 0
        while mm:
            if mm & 1:
                word ^= basis[j]
            mm >>= 1
            j += 1
        qr_words_17[m] = word
    
    # Extend to length 18 by adding a parity bit at position 17
    qr_words_18 = []
    for w in qr_words_17:
        parity = w.bit_count() & 1
        w18 = w | (parity << 17)
        qr_words_18.append(w18)
    
    # Validate: check minimum weight (distance from zero)
    valid_qr = True
    for w in qr_words_18:
        if w == 0:
            continue
        if w.bit_count() < MIN_DIST:
            valid_qr = False
            break
    
    # Full pairwise distance check if min weight passes
    if valid_qr:
        # Use vectorized check for speed
        words_arr = np.array(qr_words_18, dtype=np.uint32)
        # Check pairwise distances using broadcasting in chunks to avoid memory issues
        n = len(words_arr)
        for i in range(0, n, 64):
            chunk = words_arr[i:i+64]
            # Compute XOR with all other words
            # To save memory, do it in sub-chunks
            for j_start in range(i+1, n, 64):
                other_chunk = words_arr[j_start:j_start+64]
                # XOR: chunk[None, :] ^ other_chunk[:, None]
                xor_matrix = chunk[None, :] ^ other_chunk[:, None]
                # Popcount
                popcounts = np.array([x.bit_count() for x in xor_matrix.ravel()]).reshape(-1, 64)
                if popcounts.min() < MIN_DIST:
                    valid_qr = False
                    break
            if not valid_qr:
                break
    
    if valid_qr:
        return qr_words_18
    
    # Fallback: if QR code validation failed, use greedy search
    # Start with words that have weight >= 6
    candidates = [w for w in qr_words_18 if w.bit_count() >= MIN_DIST]
    code = []
    for w in candidates:
        ok = True
        for c in code:
            if (w ^ c).bit_count() < MIN_DIST:
                ok = False
                break
        if ok:
            code.append(w)
    
    # Try to add more codewords from the full space using random sampling
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        remaining_time = deadline - time.time() - 5
        attempts = 0
        max_attempts = int(remaining_time * 5000)  # ~5000 attempts per second
        for _ in range(max_attempts):
            if time.time() > deadline - 5:
                break
            w = random.getrandbits(18)
            if w.bit_count() < MIN_DIST:
                continue
            ok = True
            for c in code:
                if (w ^ c).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                code.append(w)
    
    return code
