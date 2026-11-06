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
    N_WORDS = 2**17
    
    # 2. Greedy Add with Vectorization
    
    # Pre-compute a set for O(1) lookup of existing words
    used = np.zeros(N_WORDS, dtype=bool)
    for w in code_list:
        used[w] = True
        
    # Candidates are all unused words
    candidates = np.where(~used)[0]
    
    # Current code as numpy array for vectorized ops
    code_arr = np.array(code_list, dtype=np.uint32)
    
    # Helper to popcount a numpy array of uint32
    # np.bitwise_count is available in newer numpy? 
    # Let's check. If not, we can use a lookup table or bincount trick.
    # Actually, `np.bitwise_count` was added in numpy 2.0? 
    # Safe fallback: use a vectorized popcount via bincount or a simple loop if array is small.
    # For large arrays, a lookup table for bytes is fast.
    
    # Let's implement a fast vectorized popcount for uint32
    def popcount_array(arr):
        # arr is uint32
        # Split into bytes
        b0 = arr & 0xFF
        b1 = (arr >> 8) & 0xFF
        b2 = (arr >> 16) & 0xFF
        b3 = (arr >> 24) & 0xFF
        
        # Lookup table for 8-bit popcount
        # This is a standard trick
        # We can precompute a lookup table for 256 values
        # But creating it every time is slow. Let's assume it's small enough or use np.bincount?
        # Actually, for the batch sizes we use (e.g., 1000), a simple loop over the batch might be faster than complex numpy tricks if the batch is small.
        # However, we are doing M x N operations.
        # Let's try to use a precomputed table for 16-bit chunks.
        # Or just use `np.apply_along_axis`? No, that's slow.
        
        # Let's stick to a simple, efficient method:
        # If numpy version is old, we might not have bitwise_count.
        # Let's use a 16-bit lookup table approach.
        # Precompute popcount for 0..65535? That's 65k ints. 
        # Let's just use a Python loop for the popcount of the XOR results if the batch is small, 
        # or better, use `bin(x).count('1')` in a list comprehension? No.
        
        # Let's try `np.bitwise_count` if available, else fallback.
        try:
            return np.bitwise_count(arr)
        except AttributeError:
            # Fallback: lookup table for 8 bits
            # This is slow for large arrays but okay for batches of 1000.
            # Actually, let's just use a standard trick:
            # x = x - ((x >> 1) & 0x55555555)
            # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
            # x = (x + (x >> 4)) & 0x0F0F0F0F
            # return (x * 0x01010101) >> 24
            x = arr
            x = x - ((x >> 1) & 0x55555555)
            x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
            x = (x + (x >> 4)) & 0x0F0F0F0F
            return (x * 0x01010101) >> 24

    batch_size = 2000
    added_count = 0
    
    # We will iterate through the candidates list
    # Note: `candidates` is a numpy array. Removing elements is slow.
    # Better to maintain a pointer or a list of indices to keep.
    # Or just a boolean mask `is_candidate`.
    
    is_candidate = used.copy() # Initially False for used, True for unused? 
    # Wait, `used` is True for used. So `is_candidate` should be `~used`.
    is_candidate = ~used
    
    while time.time() < deadline - 2.0:
        if not np.any(is_candidate):
            break
            
        # Get a batch of candidates
        # We need to pick indices where is_candidate is True
        # To avoid shuffling, we can just take the first batch_size True values
        # But finding them is O(N). 
        # Let's maintain a list of available indices.
        
        # Actually, let's just iterate through the whole array of candidates once?
        # If we do that, we might miss opportunities if we add a word that blocks a later candidate.
        # But greedy is greedy.
        
        # Let's use a pointer approach on a pre-sorted list of all unused words.
        # We can't easily remove from the middle.
        # Let's just loop through all possible words 0..2^17-1?
        # That's 131k words.
        # For each word, if not used, check if it's compatible.
        # This is O(N * M * N_code) if done naively.
        
        # Let's go back to the batch approach but with a static list of candidates.
        # We will iterate through `candidates` (the initial list of unused words).
        # We keep a set of `added_in_this_pass` to avoid re-checking?
        # No, if we add a word, it might invalidate a previously checked candidate.
        # So we should re-check?
        # Standard greedy: 
        # 1. List of candidates.
        # 2. For each candidate in list:
        #    a. Check if compatible with current code.
        #    b. If yes, add to code, mark used.
        #    c. (Optional) Remove incompatible candidates from list.
        
        # To make this fast, we do the check in batches.
        # We process the `candidates` array in chunks.
        # For each chunk, we find valid ones.
        # We add them one by one.
        
        # Optimization: If we add a word, we don't need to re-check words already processed in the current chunk if we process sequentially?
        # Actually, if we process sequentially within a chunk, and add a word, subsequent words in the chunk must be checked against the NEW code.
        # So we can't just vectorize the whole chunk at once if we are adding words during the chunk processing.
        
        # Alternative: 
        # 1. Vectorize check of chunk against CURRENT code.
        # 2. Find all valid candidates in chunk.
        # 3. From these valid candidates, find a maximal set that are mutually compatible.
        # 4. Add all of them.
        # 5. Mark them used.
        # 6. Move to next chunk.
        
        # This is much faster.
        
        # Let's implement this.
        
        # We need to iterate through the candidates.
        # Since `candidates` is a numpy array, we can slice it.
        
        start_idx = 0
        num_candidates = len(candidates)
        
        while start_idx < num_candidates and time.time() < deadline - 2.0:
            end_idx = min(start_idx + batch_size, num_candidates)
            batch = candidates[start_idx:end_idx]
            
            # Filter out any that are already used (should be none if we managed the list well, 
            # but just in case if we skipped some)
            # Actually, if we added words from previous batches, they are marked in `used`.
            # We should ensure `batch` only contains unused words.
            # Since we process sequentially and mark used, and `candidates` is static,
            # we should skip any `w` in `batch` that is now `used`.
            
            valid_in_batch = []
            for w in batch:
                if not used[w]:
                    valid_in_batch.append(w)
                    
            if not valid_in_batch:
                start_idx = end_idx
                continue
                
            valid_in_batch_arr = np.array(valid_in_batch, dtype=np.uint32)
            
            # Compute distances to current code
            # xor = valid_in_batch_arr[:, np.newaxis] ^ code_arr[np.newaxis, :]
            # This creates a (M, N) array.
            
            # If M is 2000 and N is 256, that's 512k uint32s = 2MB. Fine.
            
            xor_matrix = valid_in_batch_arr[:, np.newaxis] ^ code_arr[np.newaxis, :]
            dist_matrix = popcount_array(xor_matrix)
            min_dists = np.min(dist_matrix, axis=1)
            
            # Find candidates with min_dist >= 6
            compatible_mask = min_dists >= 6
            
            if not np.any(compatible_mask):
                start_idx = end_idx
                continue
                
            compatible_indices = np.where(compatible_mask)[0]
            compatible_words = valid_in_batch_arr[compatible_indices]
            
            # Now we have a set of words that are all compatible with the CURRENT code.
            # We need to find a subset of these that are mutually compatible.
            # This is a Maximum Independent Set problem on a small graph.
            # Size of compatible_words is at most batch_size (2000).
            # But likely much smaller.
            
            # Greedy approach for the small set:
            # Sort by weight? Or just pick first?
            # Let's pick them one by one, checking against already picked in this step.
            
            newly_added = []
            # To speed up, we can just pick the first one, add it, then re-evaluate the rest?
            # Or just pick all that are compatible with each other greedily.
            
            # Let's do a simple greedy on the compatible_words list.
            # For each word in compatible_words:
            #   Check if it is compatible with all words in `newly_added` AND `code_arr`.
            #   (It is already compatible with code_arr by definition).
            #   So just check against `newly_added`.
            
            for w in compatible_words:
                if time.time() > deadline - 2.0:
                    break
                if used[w]:
                    continue
                    
                # Check against newly_added
                is_ok = True
                for nw in newly_added:
                    if (w ^ nw).bit_count() < 6:
                        is_ok = False
                        break
                        
                if is_ok:
                    newly_added.append(w)
                    used[w] = True
                    code_list.append(w)
                    code_arr = np.append(code_arr, w)
                    added_count += 1
                    
            # If we added any words, we should probably restart the check for the rest of the batch?
            # No, the remaining words in `compatible_words` were checked against the OLD code.
            # They are still compatible with the OLD code.
            # But they might NOT be compatible with the `newly_added` words.
            # The loop above handles this by checking against `newly_added`.
            
            start_idx = end_idx
            
    return code_list

# Test run
if __name__ == "__main__":
    words = construct()
    print(f"Number of words: {len(words)}")
    # Verify
    min_dist = 100
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
    print(f"Min distance: {min_dist}")
