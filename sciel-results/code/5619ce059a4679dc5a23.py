import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Generate the [18, 9, 6] extended QR code
    g_powers = [0, 3, 4, 5, 8]
    
    basis_17 = []
    for i in range(9):
        word = 0
        for p in g_powers:
            pos = (i + p) % 17
            word |= (1 << pos)
        basis_17.append(word)
        
    basis_18 = []
    for word in basis_17:
        parity = bin(word).count('1') % 2
        if parity:
            word |= (1 << 17)
        basis_18.append(word)
        
    current_code = []
    for msg in range(512):
        word = 0
        for i in range(9):
            if (msg >> i) & 1:
                word ^= basis_18[i]
        current_code.append(word)
        
    # 2. Check for additional words using NumPy
    # Convert current_code to numpy array
    codes_np = np.array(current_code, dtype=np.uint8)
    # Reshape to (512, 18)
    # We need to convert integer to 18-bit vector
    # Create a lookup or use bit operations
    
    # Efficient way to convert ints to bits:
    # codes_np is shape (512,)
    # We want (512, 18)
    
    # Let's create the bit matrix
    bits = np.zeros((512, 18), dtype=np.uint8)
    for i in range(18):
        mask = 1 << i
        bits[:, i] = (codes_np & mask) > 0
        
    # Generate all 2^18 words as integers
    # This might be memory heavy if done all at once, but 262144 is small.
    all_ints = np.arange(2**18, dtype=np.uint32)
    
    # Exclude the existing codes?
    # We can just compute distances and filter.
    
    # Process in chunks to manage memory
    chunk_size = 10000
    candidates = []
    
    for start in range(0, 2**18, chunk_size):
        end = min(start + chunk_size, 2**18)
        chunk_ints = all_ints[start:end]
        
        # Convert chunk_ints to bits (chunk_size, 18)
        chunk_bits = np.zeros((len(chunk_ints), 18), dtype=np.uint8)
        for i in range(18):
            mask = 1 << i
            chunk_bits[:, i] = (chunk_ints & mask) > 0
            
        # Compute XOR distances
        # chunk_bits: (C, 18)
        # bits: (512, 18)
        # We want (C, 512)
        
        # Broadcasting: chunk_bits[:, np.newaxis, :] ^ bits[np.newaxis, :, :]
        # This creates (C, 512, 18)
        # Sum over axis 2 -> (C, 512)
        
        # To save memory, we can do this in smaller sub-chunks if C is large,
        # but 10000 * 512 * 18 * 1 byte = 91 MB. This is okay.
        
        diff = chunk_bits[:, np.newaxis, :] ^ bits[np.newaxis, :, :]
        weights = diff.sum(axis=2)
        min_weights = weights.min(axis=1)
        
        # Find indices in chunk where min weight >= 6
        valid_idx = np.where(min_weights >= 6)[0]
        if len(valid_idx) > 0:
            candidates.extend(chunk_ints[valid_idx].tolist())
            
        if time.time() > deadline - 10:
            break

    if not candidates:
        return current_code

    # 3. Greedy addition
    # candidates is a list of integers
    # current_code is a list of integers
    # We will try to add candidates one by one
    
    # To make distance checks faster, we can keep the code in a numpy array
    # But the list is small (starts at 512).
    
    # Sort candidates? Maybe by weight or just arbitrary.
    # Let's just iterate.
    
    # We need to check distance against all existing codewords.
    # As we add more, the list grows.
    
    # Optimization: If we add a word, we can remove candidates that are too close to it.
    
    final_code = list(current_code)
    # Convert final_code to numpy for faster distance checks if it grows large
    # But 512 is small.
    
    # Let's use a set for fast lookup? No, we need distances.
    
    # We'll keep final_code as a list of ints.
    # And a numpy array for the bits of final_code to speed up checks?
    # If we add words, we update the array.
    
    # Let's stick to simple loops for the greedy part, as it's likely to be empty or very short.
    
    # However, if candidates is large, checking each against 512+ words is O(N * M).
    # If candidates is 1000, and M=512, that's 512k checks. Fast.
    
    # Let's try to add them.
    
    # We need to be careful with time.
    
    # Convert candidates to numpy array for potential vectorization?
    # No, greedy is sequential.
    
    # Let's just iterate through candidates.
    
    # To speed up distance calculation, we can pre-compute bits for final_code?
    # If final_code changes, we re-compute.
    
    # Let's just use popcount.
    
    # Python's bin(x).count('1') is slow.
    # Use int.bit_count() (Python 3.10+) or bin(x).count('1').
    # Assuming Python 3.8+?
    # The environment likely supports standard libraries.
    # Let's use a helper function or just bin.
    
    # Actually, we can vectorize the distance check for a single candidate against all final_code words.
    
    # Let's keep final_code as a list.
    
    # If candidates is not empty, we enter the loop.
    
    # To avoid timeout, we limit the number of additions or time spent.
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
            
        # Check distance to all words in final_code
        # If any distance < 6, skip
        valid = True
        for word in final_code:
            if (cand ^ word).bit_count() < 6:
                valid = False
                break
                
        if valid:
            final_code.append(cand)
            
    return final_code

