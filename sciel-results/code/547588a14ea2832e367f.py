"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reemuller_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Returns list of 32 integers, each with Hamming weight 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _verify(code):
    """Verify pairwise Hamming distance >= 6 for all pairs in code (list of ints)."""
    n = len(code)
    for i in range(n):
        for j in range(i + 1, n):
            if (code[i] ^ code[j]).bit_count() < 6:
                return False
    return True


def _greedy_grow(seed_words, deadline, max_candidates_per_round=500000):
    """Greedy set-packing: try to add more words to the seed set.
    
    Strategy: enumerate candidate words by increasing weight (since distance-6
    codes benefit from balanced weights), check each against the current code,
    and add if it maintains minimum distance 6.
    
    Uses numpy for batched distance checks against the current code.
    """
    code = list(seed_words)
    if not code:
        return code
    
    code_arr = np.array(code, dtype=np.uint32)
    min_dist = 6
    
    # Precompute bit positions for weight-based enumeration
    # We'll try weights from 4 to 14 (symmetric around 9)
    # Actually for n=18, d=6, typical weights are around 9.
    # Let's try all weights but prioritize middle weights.
    
    weights_to_try = list(range(4, 15))  # 4..14
    
    for weight in weights_to_try:
        if time.time() > deadline:
            break
        
        # Generate candidates with given weight
        # For efficiency, we'll sample rather than enumerate all C(18, weight)
        # C(18,9) = 48620, C(18,8)=43758, C(18,7)=31824, C(18,10)=43758, C(18,11)=31824
        # These are all manageable!
        
        from itertools import combinations
        candidates = []
        for positions in combinations(range(18), weight):
            word = 0
            for p in positions:
                word |= 1 << p
            candidates.append(word)
        
        # Shuffle for diversity (deterministic seed based on weight for reproducibility)
        rng = np.random.RandomState(42 + weight)
        rng.shuffle(candidates)
        
        added_this_weight = 0
        for word in candidates:
            if time.time() > deadline:
                break
            
            # Check distance against all existing code words
            # Use numpy for batched check
            xor = code_arr ^ word
            dists = np.array([v.bit_count() for v in xor])
            if dists.min() >= min_dist:
                code.append(word)
                code_arr = np.append(code_arr, word)
                added_this_weight += 1
        
        if added_this_weight > 0:
            pass  # Could log, but keep silent
    
    return code


def _greedy_random(seed_words, deadline, n_trials=200000):
    """Random greedy: repeatedly pick a random word, check if it fits, add if so."""
    code = list(seed_words)
    if not code:
        return code
    
    code_arr = np.array(code, dtype=np.uint32)
    min_dist = 6
    rng = np.random.RandomState(12345)
    
    added = 0
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Random word
        word = rng.randint(0, 2**18)
        
        # Quick check: if it's already in code, skip
        # (probabilistic, but fine)
        
        # Check distance
        xor = code_arr ^ word
        dists = np.array([v.bit_count() for v in xor])
        if dists.min() >= min_dist:
            code.append(word)
            code_arr = np.append(code_arr, word)
            added += 1
    
    return code


def _local_search_improve(code, deadline):
    """Try to improve the code via local search: 
    - Try swapping a codeword with a nearby one to allow more additions.
    - Or try adding words by relaxing one constraint temporarily.
    
    For now, a simpler approach: try to find words at distance exactly 6 from 
    multiple existing codewords (tight spots) and see if we can fit them.
    """
    # This is complex; for now, just return the code as-is.
    # A more sophisticated local search would try to optimize the packing.
    return code


def construct():
    """Construct the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin for 90s limit
    
    # Start with Reed-Muller seed
    code = _reemuller_words()
    
    # Phase 1: Structured greedy by weight
    code = _greedy_grow(code, deadline)
    
    # Phase 2: Random greedy to fill remaining space
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        n_trials = int(remaining_time * 20000)  # ~20k trials per second estimate
        n_trials = max(n_trials, 10000)
        code = _greedy_random(code, deadline, n_trials)
    
    # Phase 3: Verify
    if not _verify(code):
        # Fallback to seed
        code = _reemuller_words()
    
    return code

# EVOLVE-BLOCK-END
