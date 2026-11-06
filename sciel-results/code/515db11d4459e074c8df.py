"""
Randomized Multi-Start Greedy Search for A_2(18, 6)
Strategy:
1. Use NumPy to vectorize distance checks.
2. Instead of scanning 0..2^N-1 sequentially (which is order-dependent and often suboptimal),
   perform multiple independent greedy passes with different random orderings of candidates.
3. For each pass, maintain the current code as a NumPy array.
4. Generate a pool of candidates (or the full space if memory/time allows).
5. In each iteration of a pass, compute distances from all remaining candidates to the current code.
   This is expensive if done naively. 
   
   Optimization: 
   Instead of checking all candidates against all codewords every time (O(M * K)), 
   we can use a "batch" approach. 
   However, M grows. 
   
   Better approach for 18 bits:
   Total space is 2^18 = 262,144. This is small enough to fit in memory.
   We can pre-compute a matrix of distances? No, 262k x 262k is too big (68 GB).
   
   Alternative:
   We keep a list of "available" candidates.
   We pick a random candidate from the available list.
   Check if it fits. If yes, add it and remove it (and potentially its neighbors? No, just remove it).
   This is O(M * K) per addition, which is slow in Python loops.
   
   Let's use a vectorized "filter" approach:
   1. Start with all 2^18 candidates in a NumPy array.
   2. Current code is empty (or seeded).
   3. Loop:
      a. If current code is empty, add a random candidate.
      b. Compute distances from ALL remaining candidates to the NEWLY added codeword.
      c. Any candidate with distance < 6 to the new codeword is "invalidated" and removed from the pool.
      d. Repeat until pool is empty or time runs out.
      
   This is the "Greedily remove neighbors" approach.
   Complexity: Each step removes at least 1 candidate. 
   In the worst case, we do 2^18 steps.
   Each step involves a vectorized XOR and popcount over the remaining pool.
   Pool size decreases.
   Sum of pool sizes is roughly N^2/2.
   N = 262k. N^2/2 is huge (34 billion). This is too slow in Python/NumPy if we do it one by one.
   
   We need to batch.
   
   Revised Batched Random Greedy:
   1. Shuffle the entire space of 2^18 candidates using a random seed.
   2. Iterate through the shuffled list.
   3. For each candidate, check against current code.
   4. If valid, add to code.
   
   This is the same as the current program but with random order.
   The current program does this sequentially. Random order might find a better packing.
   To make it faster, we can use NumPy to check a *batch* of candidates against the current code.
   
   Let's implement:
   - Pre-generate all 2^18 candidates.
   - Shuffle them.
   - Process in chunks of size C (e.g., 1000).
   - For each chunk, compute distances to all codewords in the current code.
   - Select valid ones from the chunk.
   - Add them to the code.
   
   Wait, if I add multiple candidates from a chunk, I must ensure they are mutually distant.
   The chunk candidates are checked against the *previous* code, not each other.
   So I need to filter the chunk candidates against each other too?
   Or just add them one by one within the chunk, updating the code?
   
   If I add them one by one, I lose the vectorization benefit of the chunk.
   But if the chunk is small, the overhead is low.
   
   Let's try a different angle: **Local Search / Hill Climbing**.
   Start with the Reed-Muller code (32 words).
   Try to add words. If stuck, try to swap a word for a better one?
   Or use a "repair" heuristic.
   
   Actually, the most robust strategy for this specific size (N=18) is:
   1. Use the full space 2^18.
   2. Use a randomized greedy pass.
   3. To speed up, use a bitset or efficient distance check.
   4. Since 2^18 is small, we can afford to be a bit slower if we find a better code.
   
   Let's optimize the Randomized Greedy Pass:
   - Candidates: np.arange(2**18)
   - Shuffle with seed.
   - Code: list of ints.
   - For each candidate in shuffled order:
     - Check distance to all codewords.
     - If min dist >= 6, add.
   
   To speed up the check, we can maintain the code as a NumPy array and use vectorized operations.
   However, adding to the array is O(M).
   
   Let's try to maximize the number of random restarts we can do in 80 seconds.
   If one pass takes T seconds, we can do 80/T passes.
   We keep the best result.
   
   Estimation:
   One pass: 262k candidates.
   Average M ~ 100-500.
   262k * 500 = 131M operations.
   In Python, this is slow (~10-20s).
   In NumPy, if we vectorize the *distance check* against the whole code for *each* candidate, it's still a loop over candidates.
   
   Can we vectorize over candidates?
   Yes.
   Code: shape (M,)
   Candidates: shape (K,)
   We want to check if min_{c in Code} dist(candidate, c) >= 6.
   
   We can do this in blocks of candidates.
   Let BlockSize = 1000.
   For each block of 1000 candidates:
     Compute distance matrix: (1000, M)
     This is 1000 * M XORs and popcounts.
     If M=500, 500k ops. Fast.
     Then find which candidates have min_dist >= 6.
     Add them to the code.
   
   But wait, if we add multiple candidates from the block, they might be too close to each other.
   So we must verify mutual distances among the accepted candidates in the block.
   Since the block is small (1000), we can do a pairwise check on the accepted ones.
   
   Algorithm:
   1. Shuffle all 2^18 candidates.
   2. Iterate in blocks of B (e.g., 5000).
   3. For the block, compute distances to current Code (size M).
      D = popcount(Code[None, :] ^ Block[:, None])
      min_d = D.min(axis=1)
      valid_mask = min_d >= 6
      valid_candidates = Block[valid_mask]
   4. Now, filter valid_candidates to ensure they are mutually distant >= 6.
      This is a greedy sub-problem on a small set.
      Iterate through valid_candidates:
         If dist to all already-accepted-in-this-block >= 6, accept.
   5. Add accepted candidates to Code.
   6. Repeat until end of list or time limit.
   
   This should be much faster than checking one by one because the heavy lifting (distance to existing code) is vectorized. The mutual check is on a small subset.
   
   We will run this for multiple seeds and keep the best code.
"""

import time
import numpy as np
import random

N = 18
MIN_DISTANCE = 6
DEADLINE_MARGIN = 5.0

def _seed_words():
    # Reed-Muller RM(1,4) extended to 18 bits (2 zero bits)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extend with 2 zeros (bits 16, 17 are 0)
            words.append(word)
    return words

def _run_random_greedy(seed, deadline):
    """
    Runs a single randomized greedy pass.
    Returns the list of codewords found.
    """
    rng = random.Random(seed)
    
    # Generate all candidates
    all_candidates = np.arange(1 << N, dtype=np.uint32)
    rng.shuffle(all_candidates.tolist()) # Shuffle list for randomness
    
    # We'll process in blocks to leverage NumPy
    # However, shuffling a numpy array directly is better for speed if we use rng.permutation
    # But rng.permutation creates a copy. 
    # Let's use numpy's random permutation for speed.
    # We need a reproducible seed for numpy random.
    np_rng = np.random.RandomState(seed)
    perm = np_rng.permutation(1 << N)
    
    code = _seed_words()
    if not code:
        code = [0]
        
    code_np = np.array(code, dtype=np.uint32)
    
    block_size = 2048 # Tunable parameter
    
    for i in range(0, len(perm), block_size):
        if time.time() > deadline:
            break
            
        block = perm[i : i + block_size]
        
        # Compute distances from block to current code
        # block shape: (B,)
        # code_np shape: (M,)
        # We want (B, M) matrix of distances
        
        # Optimization: If code is very large, this might be memory intensive.
        # M is likely < 1000. B is 2048. 2048*1000 = 2M entries. uint8 is fine.
        
        # XOR: (B, 1) ^ (1, M) -> (B, M)
        xor_matrix = block[:, None] ^ code_np[None, :]
        
        # Popcount
        # np.bit_count is available in NumPy 1.24+
        try:
            dist_matrix = np.bit_count(xor_matrix)
        except AttributeError:
            # Fallback: slow loop, but shouldn't happen in modern env
            dist_matrix = np.array([
                [(c ^ cand).bit_count() for c in code_np] 
                for cand in block
            ])
            
        # Find candidates in block that are far enough from ALL existing codewords
        min_dists = dist_matrix.min(axis=1)
        valid_mask = min_dists >= MIN_DISTANCE
        
        if not np.any(valid_mask):
            continue
            
        valid_candidates = block[valid_mask]
        
        # Now, we need to select a subset of valid_candidates that are mutually distant >= MIN_DISTANCE
        # This is a greedy set packing on a small set.
        # Sort them? Maybe by some heuristic? 
        # For now, just iterate in the order they appear in the block.
        
        newly_added = []
        # We can vectorize the mutual check too if newly_added is large, 
        # but it's usually small compared to block_size.
        
        for cand in valid_candidates:
            # Check against newly_added
            if newly_added:
                # Vectorized check against newly_added
                new_xor = np.array(newly_added, dtype=np.uint32) ^ cand
                # Use bit_count
                try:
                    new_dists = np.bit_count(new_xor)
                except AttributeError:
                    new_dists = np.array([(c ^ cand).bit_count() for c in newly_added])
                
                if new_dists.min() < MIN_DISTANCE:
                    continue
            
            # Check against original code_np (already done by valid_mask, 
            # but wait: valid_mask was computed against code_np BEFORE adding newly_added.
            # So we MUST check against newly_added. We did that above.
            # We do NOT need to re-check against code_np because valid_mask ensures it.
            
            newly_added.append(cand)
            
        if newly_added:
            # Update code
            code.extend(newly_added)
            code_np = np.array(code, dtype=np.uint32)
            
    return code

def construct():
    start_time = time.time()
    deadline = start_time + (90.0 - DEADLINE_MARGIN)
    
    # Try multiple seeds to find the best packing
    # Estimate time per run:
    # 2^18 / 2048 blocks = 128 blocks.
    # Each block: 2048 * M ops.
    # If M=500, 1M ops per block. 128M ops total.
    # NumPy can do ~100M-1G simple ops/sec.
    # So one run might take ~1-2 seconds? 
    # Let's try 10-20 runs.
    
    best_code = []
    
    # Always include the deterministic seed result as a baseline
    # But our random greedy starts with the seed, so it's included.
    
    num_runs = 50 # Max runs, limited by time
    
    for i in range(num_runs):
        if time.time() > deadline:
            break
            
        # Use different seeds
        seed = 42 + i
        
        current_code = _run_random_greedy(seed, deadline)
        
        if len(current_code) > len(best_code):
            best_code = current_code
            # Optional: early stop if we hit a theoretical upper bound? 
            # A_2(18, 6) is known to be around 512? 
            # If we hit 512, we can stop.
            # Reference: 512-word extended quadratic-residue construction.
            if len(best_code) >= 512:
                break
                
    # Fallback: if best_code is empty (shouldn't happen), return seed
    if not best_code:
        best_code = _seed_words()
        
    return best_code

# Keep the required interface
# The task asks to keep function names and signatures of the seed program.
# The seed program has:
# _seed_words()
# _min_distance_to_code(word, code)
# _try_add_word(word, code)
# _greedy_construct(deadline)
# construct()

# My implementation above defines _seed_words and construct.
# I should define the others to satisfy the "keep function names" constraint, 
# even if they are not used in the main strategy, or I can just ensure construct works.
# The prompt says: "Keep the function names and signatures of the seed program."
# This usually means the entry point `construct` must exist. 
# Internal helpers can be changed if the strategy is different, 
# but to be safe, I will stub the unused ones or adapt them.

def _min_distance_to_code(word, code):
    # Retained for interface compatibility, though not used in the vectorized path
    min_d = MIN_DISTANCE
    for c in code:
        d = (word ^ c).bit_count()
        if d < min_d:
            min_d = d
            if min_d < MIN_DISTANCE:
                return d
    return min_d

def _try_add_word(word, code):
    # Retained for interface compatibility
    if _min_distance_to_code(word, code) >= MIN_DISTANCE:
        code.append(word)
        return True
    return False

def _greedy_construct(deadline):
    # Retained for interface compatibility, delegates to the new strategy
    # This function signature expects a deadline and returns a list.
    # We can just call construct() logic here if needed, but construct() manages its own time.
    # To avoid double execution, we'll make this a simple wrapper that returns the seed 
    # or a quick greedy, but the main entry is construct().
    # Actually, the evaluator calls construct(). 
    # If _greedy_construct is called externally, we should provide a valid result.
    # Let's make it run one quick random pass.
    return _run_random_greedy(0, deadline)

# Final check: 
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program:
# def _seed_words():
# def _min_distance_to_code(word, code):
# def _try_add_word(word, code):
# def _greedy_construct(deadline):
# def construct():
#
# My code has all these.

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Found {len(res)} words in {end-start:.2f}s")
    # Verify a few
    if len(res) > 1:
        import itertools
        # Just check first 10 pairs to be fast
        checked = 0
        for i in range(min(10, len(res))):
            for j in range(i+1, min(10, len(res))):
                d = (res[i] ^ res[j]).bit_count()
                if d < MIN_DISTANCE:
                    print(f"ERROR: Pair {i},{j} has distance {d}")
                checked += 1
        print(f"Checked {checked} pairs")

