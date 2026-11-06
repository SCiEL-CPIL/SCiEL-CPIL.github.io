"""Construct the largest known binary code of length 17 and distance 6."""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Generate generator vector g
    g = np.zeros(N, dtype=np.uint8)
    for q in Q:
        g[q] = 1
    
    # Generate basis: shifts of g (dimension 9)
    basis = []
    for i in range(9):
        v = np.roll(g, i)
        basis.append(v)
    basis = np.array(basis)
    
    # Generate all 2^9 = 512 linear combinations, keep even-weight ones (256 codewords)
    codewords = []
    for mask in range(512):
        w = np.zeros(N, dtype=np.uint8)
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        
        if np.sum(w) % 2 == 0:
            val = 0
            for j in range(N):
                if w[j]:
                    val |= (1 << j)
            codewords.append(val)
    
    # Start with the 256-word linear code
    selected = set(codewords)
    selected_arr = np.array(list(selected), dtype=np.uint32)
    
    # Precompute popcount for 17-bit integers using a lookup table approach
    # For speed, we'll use vectorized bit operations
    
    def batch_popcount(arr, nbits=17):
        """Vectorized popcount for uint32 array with nbits bits."""
        result = np.zeros(arr.shape, dtype=np.uint8)
        for i in range(nbits):
            result += (arr >> i) & 1
        return result
    
    # Greedy fill: try to add words from the full space
    # Process in batches for efficiency
    BATCH_SIZE = 2000
    
    for batch_start in range(0, 1 << N, BATCH_SIZE):
        if time.time() > deadline - 3.0:
            break
        
        batch_end = min(batch_start + BATCH_SIZE, 1 << N)
        candidates = np.arange(batch_start, batch_end, dtype=np.uint32)
        
        # Compute distances to all selected words
        # candidates: (B, 1), selected_arr: (1, K)
        xor = candidates[:, None] ^ selected_arr[None, :]
        dists = batch_popcount(xor)
        min_dists = np.min(dists, axis=1)
        
        valid_mask = min_dists >= MIN_DIST
        valid_indices = np.where(valid_mask)[0]
        
        # Add valid candidates, but verify against current selected set
        # (since selected may have grown within this batch)
        added_in_batch = []
        for idx in valid_indices:
            val = int(candidates[idx])
            if val not in selected:
                # Verify against all currently selected words
                valid = True
                for s in selected:
                    if (val ^ s).bit_count() < MIN_DIST:
                        valid = False
                        break
                if valid:
                    selected.add(val)
                    selected_arr = np.append(selected_arr, val)
                    added_in_batch.append(val)
        
        # If we added many words, we should re-check remaining candidates in this batch
        # But for simplicity and speed, we just continue to next batch
        # The next batch will check against the updated selected_arr
    
    # Simulated annealing phase: try to add more words by modifying existing ones
    # This allows escaping the linear subspace constraint
    if time.time() < deadline - 5.0:
        # Try random perturbations of existing codewords
        current_size = len(selected)
        improvement_found = True
        while improvement_found and time.time() < deadline - 2.0:
            improvement_found = False
            
            # Try a few random modifications
            for _ in range(500):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random existing word
                base = np.random.choice(list(selected))
                
                # Flip 1-3 random bits
                num_flips = np.random.randint(1, 4)
                flip_positions = np.random.choice(N, size=num_flips, replace=False)
                
                new_word = base
                for pos in flip_positions:
                    new_word ^= (1 << pos)
                
                new_word = int(new_word)
                
                if new_word not in selected:
                    # Check if it can be added
                    valid = True
                    for s in selected:
                        if (new_word ^ s).bit_count() < MIN_DIST:
                            valid = False
                            break
                    
                    if valid:
                        selected.add(new_word)
                        selected_arr = np.append(selected_arr, new_word)
                        improvement_found = True
    
    return list(selected)
