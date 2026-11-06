import numpy as np
import time

N = 18
MIN_DIST = 6
MAX_WORDS = 2**N

# Precompute popcount for 9-bit chunks
PC9 = np.zeros(512, dtype=np.uint8)
for i in range(512):
    PC9[i] = i.bit_count()

def get_min_distances(candidates, code_arr):
    """
    candidates: np.array of shape (B,) with dtype uint32
    code_arr: np.array of shape (C,) with dtype uint32
    returns: np.array of shape (B,) with min Hamming distance to any word in code_arr
    """
    if len(code_arr) == 0:
        return np.ones(len(candidates), dtype=np.uint8)
    
    # XOR all candidates with all code words
    # Shape: (B, C)
    xors = candidates[:, None] ^ code_arr[None, :]
    
    # Split into low 9 bits and high 9 bits
    low = xors & 511
    high = xors >> 9
    
    # Popcount using lookup table
    dists = PC9[low] + PC9[high]
    
    # Min distance for each candidate
    return np.min(dists, axis=1)

def greedy_pass(seed, initial_code):
    """
    Runs a single greedy pass starting from initial_code.
    """
    np.random.seed(seed)
    code = list(initial_code)
    
    # Create a shuffled list of all words
    all_words = np.arange(MAX_WORDS, dtype=np.uint32)
    np.random.shuffle(all_words)
    
    batch_size = 2000 # Larger batch for fewer Python loop iterations
    for i in range(0, MAX_WORDS, batch_size):
        batch = all_words[i:i+batch_size]
        
        # Convert current code to array for vectorized check
        code_arr = np.array(code, dtype=np.uint32)
        
        if len(code_arr) == 0:
            valid_candidates = batch
        else:
            min_dists = get_min_distances(batch, code_arr)
            valid_indices = np.where(min_dists >= MIN_DIST)[0]
            valid_candidates = batch[valid_indices]
            
        # Filter valid_candidates to ensure they are mutually compatible
        # and compatible with the rest of the code (which was just checked, 
        # but we need to check against words added in THIS batch)
        for w in valid_candidates:
            # Check against all current code words
            # Optimization: If code is large, this is the bottleneck.
            # But valid_candidates is usually small.
            is_valid = True
            for cw in code:
                if (w ^ cw).bit_count() < MIN_DIST:
                    is_valid = False
                    break
            if is_valid:
                code.append(w)
                
    return code

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 5s safety margin
    
    # Initial Reed-Muller code (32 words)
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
            
    best_code = rm_words
    best_size = len(best_code)
    
    seed = 0
    # We can run multiple passes.
    # Each pass takes ~0.2-0.5s.
    # 55s / 0.5s = 110 passes.
    
    while time.time() < deadline:
        # Alternate strategies
        if seed % 3 == 0:
            init = rm_words
        elif seed % 3 == 1:
            init = []
        else:
            # Random start: pick 10 random words
            np.random.seed(seed)
            init = np.random.choice(MAX_WORDS, 10, replace=False).tolist()
            
        current_code = greedy_pass(seed, init)
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code
            
        seed += 1
        
    return best_code
