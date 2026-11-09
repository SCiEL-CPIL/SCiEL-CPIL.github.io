import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended QR(17) code: [17, 8, 5] -> [18, 8, 6] with 256 words
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # 0-indexed positions: {0, 1, 3, 7, 8, 12, 14, 15}
    row0 = 0
    for pos in [0, 1, 3, 7, 8, 12, 14, 15]:
        row0 |= (1 << pos)
        
    def rotate_left(x, n, width):
        return ((x << n) | (x >> (width - n))) & ((1 << width) - 1)

    basis = []
    for i in range(8):
        basis.append(rotate_left(row0, i, 17))
        
    qr_words_17 = []
    for mask in range(256):
        word = 0
        for i in range(8):
            if (mask >> i) & 1:
                word ^= basis[i]
        qr_words_17.append(word)

    # Extend to length 18 with parity bit
    qr_words_18 = []
    for w in qr_words_17:
        parity = w.bit_count() % 2
        extended = w | (parity << 17)
        qr_words_18.append(extended)
        
    current_words = list(qr_words_18)
    current_set = set(current_words)
    
    # Quick verification
    min_d = N
    for i in range(len(current_words)):
        for j in range(i + 1, len(current_words)):
            d = (current_words[i] ^ current_words[j]).bit_count()
            if d < min_d:
                min_d = d
    if min_d < MIN_DIST:
        # Fallback: try a smaller valid code
        # Reed-Muller [16,5,8] padded to 18 bits
        rm_words = []
        for a in range(16):
            for b in range(2):
                for c in range(2):
                    word = 0
                    for x in range(4):
                        for y in range(4):
                            bit = (a * x * y + b * x + c * y + 1) % 2
                            if bit:
                                word |= (1 << (x * 4 + y))
                    rm_words.append(word << 2)
        return rm_words

    # Attempt greedy extension with NumPy vectorization
    if time.time() > deadline - 10.0:
        return current_words

    current_arr = np.array(current_words, dtype=np.uint32)
    
    # Try to add words by scanning candidates in batches
    # Focus on candidates that might be far from the QR code
    # Use a subset of candidates to keep it fast
    
    # Pre-compute popcount lookup for 8-bit values
    popcount8 = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    
    def fast_popcount_uint32(arr):
        """Compute popcount for an array of uint32 values using lookup."""
        # Split into 4 bytes
        b0 = (arr & 0xFF).astype(np.uint8)
        b1 = ((arr >> 8) & 0xFF).astype(np.uint8)
        b2 = ((arr >> 16) & 0xFF).astype(np.uint8)
        b3 = ((arr >> 24) & 0xFF).astype(np.uint8)
        return popcount8[b0] + popcount8[b1] + popcount8[b2] + popcount8[b3]
    
    # Try adding candidates from a strategic subset
    # First, try all-weight candidates and some random ones
    # But to be safe and fast, let's do a limited scan
    
    # Generate a set of candidate words to try
    # We'll try candidates with various weights
    candidates_to_try = []
    
    # Add some structured candidates
    # Words with weight 0, 1, 2, ... that might be far from QR code
    # Actually, let's just do a quick scan of a portion of the space
    
    # Given time constraints, do a limited greedy
    # Check candidates in order of increasing index, but stop early if no progress
    batch_size = 4096
    added_count = 0
    last_add_time = t_start
    
    for start in range(0, 1 << N, batch_size):
        if time.time() > deadline - 5.0:
            break
        if time.time() - last_add_time > 10.0 and added_count == 0:
            # No progress in 10 seconds, likely saturated
            break
            
        end = min(start + batch_size, 1 << N)
        batch = np.arange(start, end, dtype=np.uint32)
        
        # Remove existing words
        mask = np.ones(len(batch), dtype=bool)
        for w in current_set:
            if start <= w < end:
                mask[w - start] = False
        batch = batch[mask]
        
        if len(batch) == 0:
            continue
            
        # Compute distances to all current words
        # Process in sub-batches to manage memory
        sub_batch_size = 1024
        for sb_start in range(0, len(batch), sub_batch_size):
            if time.time() > deadline - 5.0:
                break
                
            sb_end = min(sb_start + sub_batch_size, len(batch))
            sub_batch = batch[sb_start:sb_end]
            
            # XOR with all current words
            xor_matrix = sub_batch[:, np.newaxis] ^ current_arr[np.newaxis, :]
            
            # Compute popcounts
            popcounts = fast_popcount_uint32(xor_matrix)
            min_pops = popcounts.min(axis=1)
            
            valid_indices = np.where(min_pops >= MIN_DIST)[0]
            
            for idx in valid_indices:
                candidate = int(sub_batch[idx])
                if candidate not in current_set:
                    current_words.append(candidate)
                    current_set.add(candidate)
                    current_arr = np.append(current_arr, candidate)
                    added_count += 1
                    last_add_time = time.time()
                    
    return current_words
