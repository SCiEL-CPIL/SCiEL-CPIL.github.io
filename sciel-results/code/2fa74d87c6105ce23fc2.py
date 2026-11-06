"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a large binary code of length 18 with minimum distance 6.
    Uses a greedy randomized algorithm with NumPy acceleration.
    """
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    N = 18
    MIN_DIST = 6
    
    # Precompute all possible words? 2^18 = 262144, which is manageable.
    # But storing pairwise distances for 262144^2 is too much.
    # Instead, we'll use a greedy approach.
    
    # Strategy:
    # 1. Start with a good seed (e.g., Reed-Muller code)
    # 2. Greedily add words that have distance >= 6 from all existing words
    # 3. Use random restarts with different orderings to find larger codes
    
    # Reed-Muller seed (32 words)
    def reed_muller_seed():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    seed = reed_muller_seed()
    
    # Bit matrix representation for fast distance computation
    # Each word is 18 bits. We'll use numpy arrays for vectorized operations.
    
    def to_bit_matrix(words):
        """Convert list of integers to numpy array of shape (len(words), N) with 0/1 bits."""
        n = len(words)
        mat = np.zeros((n, N), dtype=np.uint8)
        for i, w in enumerate(words):
            for j in range(N):
                mat[i, j] = (w >> j) & 1
        return mat
    
    def min_distance_to_set(candidate, bit_mat):
        """Compute minimum Hamming distance from candidate to all words in bit_mat."""
        # candidate is an integer
        cand_bits = np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)
        # XOR and count bits
        xor = bit_mat ^ cand_bits
        dists = xor.sum(axis=1)
        return dists.min()
    
    def try_greedy(seed_words, rng, time_limit):
        """Greedy addition of words from a candidate pool."""
        code = list(seed_words)
        bit_mat = to_bit_matrix(code)
        
        # Candidate pool: all 2^18 words, but we'll sample them in random order
        # To be efficient, we'll iterate through candidates and add if valid
        # But 2^18 = 262144 candidates, and each check is O(|code| * N)
        # As code grows, this becomes expensive.
        
        # Better: maintain a set of "forbidden" words (within distance < 6 of any codeword)
        # But that's also expensive to maintain.
        
        # Alternative: use a randomized greedy approach
        # 1. Shuffle all 2^18 candidates
        # 2. Iterate through them, adding if distance >= 6 from all current codewords
        
        all_candidates = np.arange(2**N, dtype=np.int64)
        rng.shuffle(all_candidates)
        
        added = 0
        for idx in range(len(all_candidates)):
            if time.time() > time_limit:
                break
            
            cand = int(all_candidates[idx])
            
            # Check if cand is already in code
            if cand in code:
                continue
            
            # Compute min distance to existing code
            if len(code) == 0:
                min_d = 100
            else:
                min_d = min_distance_to_set(cand, bit_mat)
            
            if min_d >= MIN_DIST:
                code.append(cand)
                # Update bit_mat
                new_row = np.array([(cand >> j) & 1 for j in range(N)], dtype=np.uint8)
                bit_mat = np.vstack([bit_mat, new_row])
                added += 1
        
        return code
    
    best_code = seed
    best_len = len(seed)
    
    # Try multiple random restarts with different shuffling
    rng = np.random.default_rng(42)
    
    # First, try with the seed
    code = try_greedy(seed, rng, deadline - 10)  # Leave 10s margin
    if len(code) > best_len:
        best_code = code
        best_len = len(code)
    
    # Try additional restarts if time permits
    while time.time() < deadline - 5:
        # Use a different seed for randomization
        rng2 = np.random.default_rng()
        code = try_greedy(seed, rng2, deadline - 5)
        if len(code) > best_len:
            best_code = code
            best_len = len(code)
    
    return best_code

# EVOLVE-BLOCK-END
