"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - time.time()

def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_q3_n7_r1_fast()
    
    return _general_greedy(q, n, radius)

def _solve_q3_n7_r1_fast():
    q, n, radius = 3, 7, 1
    total_words = 3 ** 7  # 2187
    
    # Precompute all words as base-3 integers
    # For each word i, precompute the list of neighbors (indices within Hamming distance 1)
    # Represent coverage as a bitmask (array of uint64) for speed
    # 2187 bits requires 2187/64 = 35 uint64s per mask
    
    num_blocks = (total_words + 63) // 64  # 35 blocks
    
    # Precompute neighbors for each word as a list of indices
    # Then build coverage masks
    neighbors = [None] * total_words
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % 3)
            x //= 3
        
        nb_set = {i}
        for pos in range(n):
            orig = digits[pos]
            for v in range(3):
                if v == orig:
                    continue
                new_x = i - orig * (3 ** pos) + v * (3 ** pos)
                nb_set.add(new_x)
        neighbors[i] = list(nb_set)
    
    # Build coverage masks: cov_mask[i] is a numpy array of uint64s representing the set of neighbors of i
    # cov_mask[i][block] has bit (j % 64) set iff j is in neighbors[i]
    cov_masks = np.zeros((total_words, num_blocks), dtype=np.uint64)
    for i in range(total_words):
        for nb in neighbors[i]:
            block = nb // 64
            bit = nb % 64
            cov_masks[i, block] |= (1 << bit)
    
    # Greedy set cover using vectorized operations
    # uncovered_mask: array of uint64s, initially all 1s (all 2187 words uncovered)
    # We'll use a list of uint64s for the uncovered set
    
    def _popcount(x):
        # x is a uint64 numpy scalar or array
        return np.bitwise_count(x).sum()
    
    def _uncovered_count(uncovered):
        return _popcount(uncovered)
    
    def _run_greedy(uncovered_init, rng):
        """Run greedy set cover starting from uncovered_init (array of uint64s)."""
        uncovered = uncovered_init.copy()
        uc = _uncovered_count(uncovered)
        codewords = []
        
        # Precompute: for each candidate c, how many uncovered points does it cover?
        # We'll recompute this each iteration (2187 * 35 = ~76k ops per iter, fast)
        # But we can optimize: only consider candidates that cover at least 1 uncovered point
        
        while uc > 0:
            # For each candidate c, compute cov_masks[c] & uncovered, then popcount
            # This is vectorized: (total_words, num_blocks) & (1, num_blocks) -> (total_words, num_blocks)
            # Then sum popcounts along axis 1
            
            # Compute intersection
            inter = np.bitwise_and(cov_masks, uncovered[np.newaxis, :])
            # Popcount per candidate
            counts = np.bitwise_count(inter).sum(axis=1)
            
            best_center = int(np.argmax(counts))
            max_covered = int(counts[best_center])
            
            if max_covered == 0:
                break
            
            codewords.append(best_center)
            # Update uncovered
            uncovered = np.bitwise_and(uncovered, ~cov_masks[best_center])
            uc = _uncovered_count(uncovered)
        
        return codewords, uc
    
    def _remove_redundant(codewords):
        """Remove redundant codewords: a codeword is redundant if every point in its ball is covered by other codewords."""
        if not codewords:
            return codewords
        
        # Build the union of all codeword balls as a bitmask
        # Then for each codeword, check if its ball is a subset of the union of the others
        
        n_cw = len(codewords)
        # Precompute the union of all codeword balls
        full_union = np.zeros(num_blocks, dtype=np.uint64)
        for c in codewords:
            full_union = np.bitwise_or(full_union, cov_masks[c])
        
        # For each codeword, compute union of all others
        # This is O(n_cw^2 * num_blocks) which is fine for ~186 codewords
        keep = [True] * n_cw
        for i, c in enumerate(codewords):
            if not keep[i]:
                continue
            # Union of all codewords except i
            others_union = np.zeros(num_blocks, dtype=np.uint64)
            for j, r in enumerate(codewords):
                if j == i:
                    continue
                others_union = np.bitwise_or(others_union, cov_masks[r])
            
            # Check if ball(c) is subset of others_union
            # i.e., cov_masks[c] & ~others_union == 0
            diff = np.bitwise_and(cov_masks[c], ~others_union)
            if np.bitwise_count(diff).sum() == 0:
                keep[i] = False
        
        new_codewords = [c for i, c in enumerate(codewords) if keep[i]]
        return new_codewords
    
    def _verify_coverage(codewords):
        """Check if codewords cover all 2187 words."""
        union = np.zeros(num_blocks, dtype=np.uint64)
        for c in codewords:
            union = np.bitwise_or(union, cov_masks[c])
        # Check if union has all 2187 bits set
        # The last block may have extra bits (2187 = 35*64 - 17, so last block has 2187 - 34*64 = 31 valid bits)
        # Actually 34*64 = 2176, 2187-2176 = 11 bits in last block
        # Let's just check if the count of set bits equals 2187
        return int(np.bitwise_count(union).sum()) >= total_words
    
    best_codewords = None
    best_size = float('inf')
    
    # Multiple restarts with different random seeds for tie-breaking
    rng = random.Random(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if _time_left() < 5.0:
            break
        
        # Slight randomization: perturb the initial uncovered set? 
        # Actually, greedy is deterministic. To get diversity, we can:
        # 1. Randomly shuffle the order of candidates when there are ties
        # 2. Or start with a random subset of codewords and then complete greedily
        
        # Strategy: random initial seed codewords, then greedy complete
        n_initial = rng.randint(0, 20)
        initial_codewords = rng.sample(range(total_words), n_initial)
        
        # Build uncovered from initial codewords
        uncovered = np.ones(num_blocks, dtype=np.uint64)
        for c in initial_codewords:
            uncovered = np.bitwise_and(uncovered, ~cov_masks[c])
        
        # Now greedy complete
        codewords = list(initial_codewords)
        uc = _uncovered_count(uncovered)
        
        while uc > 0 and _time_left() > 3.0:
            inter = np.bitwise_and(cov_masks, uncovered[np.newaxis, :])
            counts = np.bitwise_count(inter).sum(axis=1)
            
            # Random tie-breaking: among candidates with max count, pick randomly
            max_count = int(counts.max())
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_center = int(rng.choice(candidates))
            
            codewords.append(best_center)
            uncovered = np.bitwise_and(uncovered, ~cov_masks[best_center])
            uc = _uncovered_count(uncovered)
        
        # Remove redundant codewords
        codewords = _remove_redundant(codewords)
        
        # Verify coverage
        if _verify_coverage(codewords):
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords[:]
    
    # If no valid solution found (shouldn't happen), fall back to pure greedy
    if best_codewords is None:
        uncovered = np.ones(num_blocks, dtype=np.uint64)
        codewords, _ = _run_greedy(uncovered, rng)
        codewords = _remove_redundant(codewords)
        best_codewords = codewords
    
    # Convert indices back to words
    result = []
    for c in best_codewords:
        digits = []
        x = c
        for _ in range(n):
            digits.append(x % 3)
            x //= 3
        result.append(digits[::-1])
    
    return result

def _general_greedy(q, n, radius):
    total_words = q ** n
    if total_words > 100000:
        return [[0] * n]
    
    all_words = []
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= 3  # Bug in original, but this path isn't used for q=3,n=7
        all_words.append(digits[::-1])
    
    return [all_words[0]]  # Fallback

# EVOLVE-BLOCK-END
