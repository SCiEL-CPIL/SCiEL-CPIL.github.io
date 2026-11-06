"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Start with the Reed-Muller construction (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    word_set = set(words)
    
    # Build numpy codebook: shape (M, N), uint8
    M0 = len(words)
    codebook = np.zeros((M0, N), dtype=np.uint8)
    for i, w in enumerate(words):
        for j in range(N):
            codebook[i, j] = (w >> j) & 1
    
    word_list = words[:]
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        batch_size = 20000
        candidates = rng.integers(0, 2**N, size=batch_size)
        
        # Convert candidates to binary vectors: (batch_size, N)
        cand_vecs = np.zeros((batch_size, N), dtype=np.uint8)
        for j in range(N):
            cand_vecs[:, j] = (candidates >> j) & 1
        
        # Check which candidates are already in the set
        # We'll handle duplicates by checking distance (dup gives dist 0 < 6)
        
        # Compute distances: XOR codebook (M, N) with each candidate
        # For efficiency, process in sub-batches to avoid memory issues
        sub_batch = 5000
        for start in range(0, batch_size, sub_batch):
            if time.time() > deadline:
                break
            
            end = min(start + sub_batch, batch_size)
            cands = candidates[start:end]
            cvecs = cand_vecs[start:end]
            
            # Filter out known words first
            mask_new = np.array([c not in word_set for c in cands], dtype=bool)
            if not np.any(mask_new):
                continue
            
            cvecs_new = cvecs[mask_new]
            cands_new = cands[mask_new]
            
            # Compute min distance from each candidate to all existing codewords
            # codebook shape: (M, N), cvecs_new shape: (K, N)
            # XOR: broadcast (M, 1, N) ^ (1, K, N) -> (M, K, N)
            # This could be large. Let's do it in chunks if needed.
            
            M = codebook.shape[0]
            K = cvecs_new.shape[0]
            
            # If M*K is manageable, do it all at once
            if M * K <= 50_000_000:  # ~50M uint8 values = 50MB
                xors = codebook[:, np.newaxis, :] ^ cvecs_new[np.newaxis, :, :]
                dists = np.sum(xors, axis=2)  # (M, K)
                min_dists = np.min(dists, axis=0)  # (K,)
            else:
                # Process in chunks over M
                min_dists = np.full(K, N, dtype=np.uint8)
                chunk = 2000
                for m_start in range(0, M, chunk):
                    m_end = min(m_start + chunk, M)
                    xors = codebook[m_start:m_end, np.newaxis, :] ^ cvecs_new[np.newaxis, :, :]
                    dists = np.sum(xors, axis=2)  # (chunk, K)
                    min_dists = np.minimum(min_dists, np.min(dists, axis=0))
            
            # Find valid candidates
            valid_mask = min_dists >= MIN_DIST
            valid_indices = np.where(valid_mask)[0]
            
            for idx in valid_indices:
                cand = int(cands_new[idx])
                word_list.append(cand)
                word_set.add(cand)
                
                # Add to codebook
                new_row = cvecs_new[idx]
                codebook = np.vstack([codebook, new_row[None, :]])
        
    return word_list
# EVOLVE-BLOCK-END
