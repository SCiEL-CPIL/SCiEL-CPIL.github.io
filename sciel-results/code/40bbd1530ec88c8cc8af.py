import time
import numpy as np

N = 18
MIN_DISTANCE = 6
DEADLINE_MARGIN = 5.0

# Precompute bit positions
BITS = np.arange(N)

def _reseeded_rng(seed):
    import random
    return random.Random(seed)

def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _compute_distances_numpy(code_words, all_indices):
    """
    Compute Hamming distances from a set of code words to a large set of candidate indices.
    
    Args:
        code_words: list of integers (codewords)
        all_indices: np.array of integers (candidate indices)
        
    Returns:
        np.array of shape (len(code_words), len(all_indices)) containing Hamming distances.
    """
    if not code_words:
        return np.zeros((0, len(all_indices)), dtype=np.uint8)
    
    # Convert code words to binary bits
    # code_words is a list of ints. We want shape (num_words, N) of 0s and 1s.
    cw = np.array(code_words, dtype=np.uint32)
    # Extract bits: (cw[:, None] >> BITS) & 1
    # This creates an array of shape (num_words, N)
    cw_bits = (cw[:, None] >> BITS) & 1
    
    # Convert candidate indices to binary bits
    # all_indices is an array of ints.
    ai = np.array(all_indices, dtype=np.uint32)
    ai_bits = (ai[:, None] >> BITS) & 1
    
    # Hamming distance = sum of XORs
    # XOR: cw_bits[:, :, None] ^ ai_bits[None, :, :]
    # Shape: (num_words, N, num_candidates)
    # Sum over axis 1 (bits) -> (num_words, num_candidates)
    
    # To save memory, we can do this in chunks if necessary, but 2^18 is 262k.
    # If code size is ~100, 100 * 262k * 18 * 1 byte ~ 470 MB. This might be too much for standard memory limits.
    # Let's chunk the candidates.
    
    num_candidates = len(all_indices)
    num_words = len(cw_bits)
    
    # We want the MINIMUM distance for each candidate.
    # dist_matrix shape: (num_words, chunk_size)
    # min_dists shape: (chunk_size,)
    
    min_dists = np.full(num_candidates, 255, dtype=np.uint8)
    
    chunk_size = 10000 # Adjust based on memory. 100 words * 10k candidates * 18 bits * 1 byte = 18 MB. Safe.
    
    for start in range(0, num_candidates, chunk_size):
        end = min(start + chunk_size, num_candidates)
        ai_chunk = ai[start:end]
        ai_chunk_bits = (ai_chunk[:, None] >> BITS) & 1
        
        # XOR
        # cw_bits: (num_words, N) -> (num_words, N, 1)
        # ai_chunk_bits: (chunk_size, N) -> (1, N, chunk_size)
        # Result: (num_words, N, chunk_size)
        xor = cw_bits[:, :, None] ^ ai_chunk_bits[None, :, :]
        
        # Sum over bits (axis 1)
        # Result: (num_words, chunk_size)
        dists = np.sum(xor, axis=1, dtype=np.uint8)
        
        # Min over words (axis 0)
        # Result: (chunk_size,)
        chunk_min = np.min(dists, axis=0)
        
        min_dists[start:end] = chunk_min
        
    return min_dists

def _local_search_improve(deadline, seed=123):
    """
    Local search: Start with seed code. Repeatedly find the candidate with the
    maximum minimum distance to the current code. Add it if it satisfies MIN_DISTANCE.
    """
    words = _seed_words()
    
    # Precompute all candidate indices
    all_indices = np.arange(2**N, dtype=np.uint32)
    
    # Exclude current words from candidates? 
    # Actually, if a word is in the code, its distance to itself is 0. 
    # We want to find candidates NOT in the code with dist >= MIN_DISTANCE.
    # If we include current words, they will have min_dist=0 and won't be selected 
    # unless no other valid candidates exist (which would mean we're stuck).
    
    rng = _reseeded_rng(seed)
    
    # Shuffle candidates to break symmetry? 
    # The "best" candidate is well-defined (max min_dist). If there are ties, 
    # the order of indices matters. Shuffling might help find diverse local optima.
    # However, the distance calculation is expensive. 
    # Let's just iterate through the standard range, but maybe we can randomize the 
    # tie-breaking by permuting the indices?
    
    # Actually, a better strategy for local search is to just pick the global best.
    # If there are multiple with the same max min_dist, picking the first one is fine.
    
    while time.time() < deadline:
        # Calculate min distance from every candidate to the current code
        min_dists = _compute_distances_numpy(words, all_indices)
        
        # Find the candidate with the maximum minimum distance
        # We only care about candidates with min_dist >= MIN_DISTANCE
        # If max min_dist < MIN_DISTANCE, we are stuck.
        
        max_min_dist = np.max(min_dists)
        
        if max_min_dist < MIN_DISTANCE:
            # No valid candidates left. Stuck.
            break
        
        # Find the index of the candidate with the maximum min_dist
        # np.argmax returns the first occurrence.
        best_idx = np.argmax(min_dists)
        
        # Get the word value
        best_word = int(all_indices[best_idx])
        
        # Check if it's already in the code (should be 0 dist, so unlikely to be max unless code is empty or something weird)
        # But wait, if best_word is in words, min_dist would be 0. 
        # If max_min_dist >= 6, best_word is definitely not in words.
        
        # Add the word
        words.append(best_word)
        
        # Optimization: We can remove the newly added word from the candidate pool?
        # No, the distance calculation uses the full array. 
        # We could mask it, but it's simpler to just let it be there. 
        # It will have dist 0 to itself, so it won't be picked again.
        
        # To save time, we could update the distance matrix incrementally, 
        # but recalculating is safer and vectorized. 
        # 2^18 * 18 * 1 byte * 100 words = ~500MB per iteration? 
        # Wait, 262k * 18 * 100 = 471,600,000 bytes = 471 MB. 
        # This is borderline. 
        # Let's check if we can reduce memory.
        # We can process in smaller chunks or use uint8.
        
    return words

def _greedy_fast(deadline, seed=123):
    """
    Fast greedy using bit-parallel distance check for batches.
    """
    words = _seed_words()
    
    all_indices = np.arange(2**N, dtype=np.uint32)
    
    rng = _reseeded_rng(seed)
    
    # We will scan through candidates in a random order, but in batches.
    # For each batch, compute min dist to current code.
    # Add all that are valid.
    
    # To ensure we don't miss good spots, maybe randomize the order of all_indices?
    # Permuting 2^18 elements takes time but is doable.
    perm = rng.permutation(all_indices)
    
    # Process in chunks
    chunk_size = 50000
    
    for start in range(0, len(perm), chunk_size):
        if time.time() > deadline:
            break
            
        end = min(start + chunk_size, len(perm))
        chunk = perm[start:end]
        
        # Compute min distances for this chunk
        min_dists = _compute_distances_numpy(words, chunk)
        
        # Find valid candidates (min_dist >= MIN_DISTANCE)
        valid_mask = min_dists >= MIN_DISTANCE
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            # Get the actual word values
            new_words = [int(chunk[i]) for i in valid_indices]
            
            # Add them all
            # Note: Adding multiple words at once is valid if they are mutually far apart?
            # NO! The min_dist calculation is against the OLD code.
            # Two new words might be close to each other.
            # We need to check mutual distances or add one by one.
            # Given the cost, let's just add the BEST one from the batch?
            # Or check mutual distances?
            
            # Let's just add the one with the highest min_dist to the old code,
            # and re-evaluate the rest? That's expensive.
            
            # Alternative: Add the first valid one found in the chunk?
            # That's basically greedy.
            
            # Let's try to add as many as possible from this chunk.
            # We can check mutual distances of the valid candidates.
            # If we have K valid candidates, check KxK distances.
            # K is likely small.
            
            if len(new_words) > 0:
                # Sort by min_dist descending?
                # Let's just iterate and check against current words + newly added
                added_in_chunk = []
                for w in new_words:
                    # Check against current words
                    # We already know dist to old words >= MIN_DISTANCE
                    # Check against added_in_chunk
                    ok = True
                    for aw in added_in_chunk:
                        if (w ^ aw).bit_count() < MIN_DISTANCE:
                            ok = False
                            break
                    if ok:
                        added_in_chunk.append(w)
                        
                words.extend(added_in_chunk)
                
    return words

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN
    
    best_words = None
    best_len = 0
    
    # Strategy 1: Local Search (Global best candidate each step)
    # This is slow per step but might find a better code.
    if time.time() < deadline:
        try:
            w = _local_search_improve(deadline, seed=1)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass
    
    # Strategy 2: Batched Greedy
    if time.time() < deadline:
        try:
            w = _greedy_fast(deadline, seed=42)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass
            
    # Strategy 3: More Local Search with different seeds
    for seed in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        try:
            w = _local_search_improve(deadline, seed=seed)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # Fallback
    if best_words is None:
        best_words = _seed_words()

    # Ensure distinctness and range
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < 2**N:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()

    return out
