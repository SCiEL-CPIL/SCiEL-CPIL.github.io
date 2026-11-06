import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Pre-compute all vectors as uint8 for fast XOR-based distance
    all_indices = np.arange(TOTAL, dtype=np.uint32)
    # all_vectors[i, j] = bit j of i
    all_vectors = (all_indices[:, None] >> np.arange(N)[None, :]) & 1
    all_vectors = all_vectors.astype(np.uint8)
    
    # Build the 256-word linear QR code seed.
    # The [18,9,6] extended quadratic-residue code shortened by one coordinate.
    # We construct it as a linear code: generate 9 basis vectors of length 17
    # with pairwise distance >= 6, then take all 2^9 = 512 combinations.
    # But we need only 256 distinct words, so we use 8 basis vectors (2^8=256).
    # Actually, let's use the RM(1,4) code extended to get a better seed.
    
    # Better approach: use the 256-word linear code from the QR construction.
    # The [18,9,6] QR code has generator matrix G of size 9x18.
    # Shortening by one coordinate gives a [17,8,6] or [17,9,6] code.
    # Let's construct a [17,8,6] linear code: 8 basis vectors, 256 codewords.
    
    # We'll find 8 linearly independent 17-bit vectors with all pairwise distances >= 6.
    # Use a greedy approach to find such basis vectors.
    
    def find_basis_vectors(num_vectors, deadline):
        """Find num_vectors linearly independent 17-bit words with pairwise dist >= 6."""
        basis = []
        basis_set = set()
        
        # Start with the all-zero vector (it's in any linear code)
        # Actually, for a linear code, we need the basis to span the code.
        # Let's find non-zero basis vectors.
        
        candidates = list(range(1, TOTAL))
        np.random.shuffle(candidates)
        
        for candidate in candidates:
            if time.time() > deadline:
                break
            if len(basis) >= num_vectors:
                break
            
            # Check if candidate is linearly independent from existing basis
            # And has distance >= 6 from all existing basis vectors AND all their combinations
            
            # First check: is it linearly independent?
            # Use Gaussian elimination style: check if candidate can be expressed as XOR of basis vectors
            # Simple check: candidate not in span of basis
            in_span = False
            # Enumerate all combinations of basis (up to 2^num_vectors)
            if len(basis) <= 12:
                for mask in range(1 << len(basis)):
                    combo = 0
                    for i in range(len(basis)):
                        if mask & (1 << i):
                            combo ^= basis[i]
                    if combo == candidate:
                        in_span = True
                        break
            
            if in_span:
                continue
            
            # Check distance to all existing codewords in the span
            # Generate all codewords in span
            span_words = [0]
            for b in basis:
                span_words = span_words + [w ^ b for w in span_words]
            
            # Check if candidate has distance >= 6 from all span words
            valid = True
            c_vec = all_vectors[candidate]
            for sw in span_words:
                s_vec = all_vectors[sw]
                dist = np.sum(c_vec != s_vec)
                if dist < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                basis.append(candidate)
                basis_set.add(candidate)
        
        return basis
    
    # Find 8 basis vectors for a [17,8,6] linear code
    basis = find_basis_vectors(8, deadline - 10)
    
    # If we couldn't find 8, try with fewer
    num_basis = len(basis)
    
    # Generate all 2^num_basis codewords from the basis
    if num_basis > 0:
        seed_words = [0]
        for b in basis:
            seed_words = seed_words + [w ^ b for w in seed_words]
        seed_words = list(set(seed_words))
    else:
        # Fallback: use RM(1,4) code
        seed_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                seed_words.append(word)
    
    seed_indices = np.array(seed_words, dtype=np.uint32)
    
    best_size = len(seed_indices)
    best_code = list(seed_indices)
    
    # Now use random greedy to extend the seed
    def greedy_pass(seed_idx_list, deadline):
        """Perform a random greedy pass starting from the given seed."""
        current_indices = list(seed_idx_list)
        current_set = set(current_indices)
        
        # Available candidates: all words not in current code
        available = np.setdiff1d(all_indices, current_indices)
        np.random.shuffle(available)
        
        # We'll process candidates in order, adding those that are valid.
        # To speed up, we'll use a vectorized distance check.
        # Maintain an array of current codeword vectors.
        current_vecs = all_vectors[np.array(current_indices)]
        
        added = 0
        for idx in available:
            if time.time() > deadline:
                break
            c_vec = all_vectors[idx]
            # Compute distances to all current codewords
            dists = np.sum(current_vecs != c_vec, axis=1)
            if np.min(dists) >= MIN_DIST:
                current_indices.append(idx)
                current_set.add(idx)
                current_vecs = np.vstack([current_vecs, c_vec])
                added += 1
        
        return current_indices
    
    # Run multiple greedy passes with different random seeds
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        
        # Shuffle the seed order to get different starting points
        shuffled_seed = list(seed_indices)
        np.random.shuffle(shuffled_seed)
        
        current_code = greedy_pass(shuffled_seed, deadline - 1)
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
    
    # Local search improvement: try to swap out poorly-placed codewords
    if best_size > 0 and time.time() < deadline - 5:
        current_code = best_code[:]
        current_set = set(current_code)
        current_vecs = all_vectors[np.array(current_code)]
        
        # Try to improve by removing a codeword and adding two new ones
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(current_code)):
                if time.time() > deadline - 2:
                    break
                # Try removing codeword i
                removed_word = current_code[i]
                temp_code = current_code[:i] + current_code[i+1:]
                temp_vecs = np.vstack([current_vecs[:i], current_vecs[i+1:]]) if len(temp_code) > 0 else np.array([], dtype=np.uint8).reshape(0, N)
                
                # Find candidates that are now available (were conflicting with removed_word)
                # and can be added
                # Sample some candidates
                sample_size = min(5000, TOTAL - len(temp_code))
                sample = np.random.choice(TOTAL, size=sample_size, replace=False)
                
                new_additions = []
                for idx in sample:
                    if idx in current_set and idx != removed_word:
                        continue
                    c_vec = all_vectors[idx]
                    if len(temp_vecs) > 0:
                        dists = np.sum(temp_vecs != c_vec, axis=1)
                        if np.min(dists) < MIN_DIST:
                            continue
                    # Check distance to all temp_code words
                    valid = True
                    for tw in temp_code:
                        if tw == idx:
                            continue
                        t_vec = all_vectors[tw]
                        dist = np.sum(c_vec != t_vec)
                        if dist < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        new_additions.append(idx)
                
                # If we can add at least 2 new words, do the swap
                if len(new_additions) >= 2:
                    # Add the best 2 (or more) new words
                    for new_word in new_additions[:2]:
                        if new_word not in current_set:
                            current_code.append(new_word)
                            current_set.add(new_word)
                            current_vecs = np.vstack([current_vecs, all_vectors[new_word]])
                    # Remove the old word
                    current_code.pop(i)
                    current_set.discard(removed_word)
                    current_vecs = np.vstack([current_vecs[:i], current_vecs[i+1:]]) if len(current_vecs) > 1 else np.array([], dtype=np.uint8).reshape(0, N)
                    improved = True
                    if len(current_code) > best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                    break
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
    
    return best_code
