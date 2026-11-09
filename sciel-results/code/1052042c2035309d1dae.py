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
        
    # The QR code is cyclic. We can generate the 8 basis vectors by rotating row0.
    # Note: np.roll on an integer bitstring is not direct. We can use bit rotation.
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
    # Only verify a sample or all if fast enough. 256*255/2 ~ 32k pairs. Very fast.
    for i in range(len(current_words)):
        for j in range(i + 1, len(current_words)):
            d = (current_words[i] ^ current_words[j]).bit_count()
            if d < min_d:
                min_d = d
    if min_d < MIN_DIST:
        # Fallback if QR construction failed (should not happen)
        return current_words[:32] # Or just return what we have

    # Try to add more words using a fast greedy
    # We use a bitset of all possible words to quickly check availability? 
    # No, we need to check distance.
    
    # Efficient Greedy:
    # Maintain a list of current words.
    # For each candidate in a pre-defined order (e.g., by weight, or random),
    # check if it is at distance >= MIN_DIST from all current words.
    
    # To speed up, we can pre-compute the "forbidden" regions?
    # With 256 words, the forbidden regions are large.
    
    # Let's try a few random restarts or a specific ordering.
    # Given the high likelihood of saturation, we might not find any.
    # But we must try within the time budget.
    
    # Generate candidates in order of increasing weight (often denser packing)
    # Or just iterate through all 2^18? That's 262k.
    # Checking 262k candidates * 256 existing words * bit_count is 262k * 256 ~ 67M ops.
    # Python is slow. 67M ops might take > 10s.
    # We need to be smarter.
    
    # Optimization: Use NumPy for the distance check against the *current* set.
    # But the current set grows.
    
    # Let's stick to the previous NumPy approach but optimize the loop.
    # Actually, if we suspect 256 is max, we can just return.
    # But to be safe, we run a short greedy.
    
    if time.time() > deadline - 5.0:
        return current_words

    # Filter candidates: exclude those already in current_set
    # We can't filter all 2^18 easily without a huge array.
    # We will iterate through all candidates 0..2^18-1.
    
    # Vectorized check against current_words
    current_arr = np.array(current_words, dtype=np.uint32)
    
    # Process in batches
    batch_size = 4096
    for start in range(0, 1 << N, batch_size):
        if time.time() > deadline - 5.0:
            break
        
        end = min(start + batch_size, 1 << N)
        batch = np.arange(start, end, dtype=np.uint32)
        
        # Remove those already in current_set
        # This is tricky with numpy. 
        # We can just check them all, and skip if in set.
        
        # Compute XOR with all current words
        # Memory: 4096 * 256 * 4 bytes ~ 4MB. OK.
        xor_matrix = batch[:, np.newaxis] ^ current_arr[np.newaxis, :]
        
        # Popcount
        # Use a lookup table for 8-bit chunks?
        # Or use np.unpackbits?
        # xor_matrix is uint32.
        # Convert to uint8 array of bits?
        
        # Faster popcount for uint32 in numpy:
        # x = x - ((x >> 1) & 0x55555555)
        # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        # x = (x + (x >> 4)) & 0x0F0F0F0F
        # popcount = ((x * 0x01010101) >> 24) & 0xFF
        
        x = xor_matrix
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        popcounts = ((x * 0x01010101) >> 24) & 0xFF
        
        min_pops = popcounts.min(axis=1)
        valid_indices = np.where(min_pops >= MIN_DIST)[0]
        
        for idx in valid_indices:
            candidate = int(batch[idx])
            if candidate not in current_set:
                current_words.append(candidate)
                current_set.add(candidate)
                current_arr = np.append(current_arr, candidate)
                # Note: current_arr grows, so next batch will be slower.
                # If it grows too much, we might timeout.
                # But if we add many, we are improving score.
                
    return current_words
