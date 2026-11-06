import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Precompute bit masks
    bit_masks = np.array([1 << i for i in range(N)], dtype=np.uint32)
    
    # Helper: Convert list of ints to bit matrix (M x N) using vectorized ops
    def ints_to_mat(words):
        if len(words) == 0:
            return np.empty((0, N), dtype=np.uint8)
        arr = np.array(words, dtype=np.uint32)
        # Extract bits: (arr[:, None] & bit_masks[None, :]) > 0
        mat = ((arr[:, None] & bit_masks[None, :]) > 0).astype(np.uint8)
        return mat
    
    # Helper: Min distance from a single word (int) to a matrix
    def min_dist_to_mat(word_int, mat):
        if len(mat) == 0:
            return N
        w = ((word_int & bit_masks) > 0).astype(np.uint8)
        dists = (mat ^ w).sum(axis=1)
        return int(np.min(dists))
    
    # Helper: Check if word is valid (min dist >= MIN_DIST) against matrix
    def is_valid(word_int, mat):
        if len(mat) == 0:
            return True
        w = ((word_int & bit_masks) > 0).astype(np.uint8)
        dists = (mat ^ w).sum(axis=1)
        return bool(np.all(dists >= MIN_DIST))
    
    # Generate the 32-word Reed-Muller RM(1,4) seed
    def generate_rm_seed():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    if bit:
                        word |= (1 << point)
                # 17th bit = 0
                words.append(word)
        return words
    
    # Build initial code from seed using greedy + local search
    def build_code(seed_words, time_limit):
        words = seed_words[:]
        mat = ints_to_mat(words)
        best_words = words[:]
        best_size = len(words)
        
        attempts = 0
        stuck = 0
        max_stuck = 50
        
        while time.time() < time_limit:
            # Try to add a new word
            added = False
            
            # Strategy 1: Random word
            if random.random() < 0.3:
                candidate = random.getrandbits(N)
                if is_valid(candidate, mat):
                    words.append(candidate)
                    mat = np.vstack([mat, ((candidate & bit_masks) > 0).astype(np.uint8)])
                    added = True
                    if len(words) > best_size:
                        best_size = len(words)
                        best_words = words[:]
                    stuck = 0
                    continue
            
            # Strategy 2: Mutate existing word (flip 1-3 bits)
            if random.random() < 0.4:
                base = random.choice(words)
                num_flips = random.randint(1, 3)
                positions = random.sample(range(N), num_flips)
                candidate = base
                for p in positions:
                    candidate ^= (1 << p)
                if is_valid(candidate, mat):
                    words.append(candidate)
                    mat = np.vstack([mat, ((candidate & bit_masks) > 0).astype(np.uint8)])
                    added = True
                    if len(words) > best_size:
                        best_size = len(words)
                        best_words = words[:]
                    stuck = 0
                    continue
            
            # Strategy 3: XOR of 2-3 random words, then perturb
            if random.random() < 0.3:
                k = random.randint(2, 3)
                idxs = random.sample(range(len(words)), k)
                candidate = 0
                for i in idxs:
                    candidate ^= words[i]
                # Perturb with 1-2 bit flips
                if random.random() < 0.5:
                    pos = random.randint(0, N - 1)
                    candidate ^= (1 << pos)
                if is_valid(candidate, mat):
                    words.append(candidate)
                    mat = np.vstack([mat, ((candidate & bit_masks) > 0).astype(np.uint8)])
                    added = True
                    if len(words) > best_size:
                        best_size = len(words)
                        best_words = words[:]
                    stuck = 0
                    continue
            
            # Strategy 4: Swap - replace one word to allow better packing
            if not added and stuck > max_stuck:
                stuck = 0
                # Pick a word to remove
                idx = random.randint(0, len(words) - 1)
                old_word = words[idx]
                
                # Generate candidate to replace it
                if random.random() < 0.5:
                    base = random.choice(words)
                    num_flips = random.randint(3, 7)
                    positions = random.sample(range(N), num_flips)
                    candidate = base
                    for p in positions:
                        candidate ^= (1 << p)
                else:
                    candidate = random.getrandbits(N)
                
                # Check if candidate is valid against all words EXCEPT idx
                if len(words) > 1:
                    w_vec = ((candidate & bit_masks) > 0).astype(np.uint8)
                    dists = (mat ^ w_vec).sum(axis=1)
                    # Ignore distance to idx
                    valid = True
                    for j in range(len(dists)):
                        if j != idx and dists[j] < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        words[idx] = candidate
                        mat[idx] = w_vec
                        # Size unchanged, but packing may improve
                else:
                    words[idx] = candidate
                    mat[idx] = ((candidate & bit_masks) > 0).astype(np.uint8)
                
                # After swap, try to add
                # (will happen in next iteration)
            
            if not added:
                stuck += 1
            
            attempts += 1
            
            # Periodic restart of local search if stuck too long
            if stuck > max_stuck * 3:
                stuck = 0
                # Try a more aggressive restart: remove a random word and re-add
                if len(words) > 10:
                    idx = random.randint(0, len(words) - 1)
                    removed = words.pop(idx)
                    mat = np.delete(mat, idx, axis=0)
                    # Now try to add back a different word
                    for _ in range(20):
                        candidate = random.getrandbits(N)
                        if is_valid(candidate, mat):
                            words.append(candidate)
                            mat = np.vstack([mat, ((candidate & bit_masks) > 0).astype(np.uint8)])
                            break
    
    # Main: multiple restarts with different seeds
    best_overall = []
    best_size_overall = 0
    
    # First, build from the RM seed
    seed_words = generate_rm_seed()
    result = build_code(seed_words, deadline - 5.0)  # Leave 5s for other restarts
    if len(result) > best_size_overall:
        best_size_overall = len(result)
        best_overall = result
    
    # If we have time, do additional restarts with different initial codes
    while time.time() < deadline - 3.0:
        # Create a different seed by perturbing the RM code
        perturbed_seed = seed_words[:]
        for _ in range(3):
            idx = random.randint(0, len(perturbed_seed) - 1)
            pos = random.randint(0, N - 1)
            perturbed_seed[idx] ^= (1 << pos)
        
        # Verify the perturbed seed is still valid
        mat_check = ints_to_mat(perturbed_seed)
        valid = True
        for i in range(len(perturbed_seed)):
            for j in range(i + 1, len(perturbed_seed)):
                d = (mat_check[i] ^ mat_check[j]).sum()
                if d < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        time_left = deadline - time.time() - 2.0
        if time_left < 1.0:
            break
        
        result = build_code(perturbed_seed, time.time() + time_left)
        if len(result) > best_size_overall:
            best_size_overall = len(result)
            best_overall = result
    
    return best_overall

