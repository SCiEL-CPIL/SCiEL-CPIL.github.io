"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _hamming_distance(a, b):
    return (a ^ b).bit_count()

def _rm_code_18():
    """Construct the extended Reed-Muller code RM(1,4) padded to 18 bits (32 words)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Pad with 2 zero bits at positions 16,17
            words.append(word)
    return words

def _vectorized_greedy(start_words, n_bits=18, min_dist=6, time_limit=50, seed=42):
    """Vectorized greedy: use NumPy to check many candidates at once."""
    start_time = time.time()
    deadline = start_time + time_limit - 2
    
    # Convert start words to numpy array
    words_arr = np.array(start_words, dtype=np.uint32)
    current_list = list(start_words)
    current_set = set(start_words)
    
    rng = np.random.RandomState(seed)
    
    # Try to add words greedily using vectorized checks
    # Process candidates in batches
    batch_size = 50000
    added_any = True
    total_added = 0
    
    while added_any and (time.time() < deadline):
        added_any = False
        
        # Generate a batch of random candidates
        candidates = rng.randint(0, 2**n_bits, size=batch_size).astype(np.uint32)
        
        # Check each candidate against all current words using vectorized XOR
        # We need: for each candidate, min distance to any current word >= min_dist
        # This is: for each candidate c, for all w in words: popcount(c ^ w) >= min_dist
        
        # Process in sub-batches to manage memory
        sub_batch = 5000
        for i in range(0, len(candidates), sub_batch):
            if time.time() > deadline:
                break
            
            cands = candidates[i:i+sub_batch]
            
            # Check membership
            mask_not_in = np.array([c not in current_set for c in cands], dtype=bool)
            if not mask_not_in.any():
                continue
            cands = cands[mask_not_in]
            
            # Vectorized distance check: compute popcount(cand ^ word) for all pairs
            # words_arr shape: (M,), cands shape: (K,)
            # We want: min over M of popcount(cand_j ^ word_i) >= min_dist for each j
            
            # Do it in chunks over words to manage memory
            m = len(words_arr)
            min_dists = np.full(len(cands), 999, dtype=np.int32)
            
            for wi in range(0, m, 100):
                w_chunk = words_arr[wi:wi+100]  # shape (chunk,)
                # XOR: cands[:, None] ^ w_chunk[None, :] -> shape (K, chunk)
                xor_vals = cands[:, None].astype(np.uint32) ^ w_chunk[None, :].astype(np.uint32)
                # Popcount using bit manipulation
                # Use numpy's bit_count if available (Python 3.10+ / numpy)
                # Fallback: use table lookup or manual
                # For uint32, we can use a lookup table
                # But let's use a simpler approach: process one word at a time for large sets
                # Actually for m <= ~200, let's just do it directly
                
                # Compute popcount for each pair
                # xor_vals shape: (K, chunk_size)
                # Use a vectorized popcount
                x = xor_vals
                x = (x & 0x55555555) + ((x >> 1) & 0x55555555)
                x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
                x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F)
                x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF)
                x = (x & 0x0000FFFF) + ((x >> 16) & 0x0000FFFF)
                x = (x & 0x000000FF) + ((x >> 8) & 0x000000FF)
                popcounts = (x & 0x000000FF).astype(np.int32)
                
                # min_dists[j] = min(min_dists[j], min over chunk of popcounts[j, :])
                chunk_min = popcounts.min(axis=1)
                min_dists = np.minimum(min_dists, chunk_min)
            
            # Find candidates where min_dist >= required
            good_mask = min_dists >= min_dist
            if good_mask.any():
                good_cands = cands[good_mask]
                for c in good_cands:
                    c_int = int(c)
                    if c_int not in current_set:
                        current_list.append(c_int)
                        current_set.add(c_int)
                        words_arr = np.append(words_arr, c_int)
                        added_any = True
                        total_added += 1
    
    return current_list

def _systematic_greedy(start_words, n_bits=18, min_dist=6, time_limit=50, seed=42):
    """More systematic greedy: try structured candidates by weight."""
    start_time = time.time()
    deadline = start_time + time_limit - 2
    
    current_list = list(start_words)
    current_set = set(start_words)
    words_arr = np.array(start_words, dtype=np.uint32)
    
    rng = np.random.RandomState(seed)
    
    # Try candidates organized by weight for better coverage
    for target_weight in range(0, n_bits + 1):
        if time.time() > deadline:
            break
        
        # Try many random candidates of approximately this weight
        num_tries = 3000
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            
            # Generate random word with approximately target_weight bits
            if target_weight == 0:
                candidate = 0
            elif target_weight == n_bits:
                candidate = (1 << n_bits) - 1
            else:
                # Pick target_weight random positions
                positions = rng.choice(n_bits, size=target_weight, replace=False)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
            
            if candidate in current_set:
                continue
            
            # Check distance to all current words
            ok = True
            for w in current_list:
                if (w ^ candidate).bit_count() < min_dist:
                    ok = False
                    break
            
            if ok:
                current_list.append(candidate)
                current_set.add(candidate)
                words_arr = np.append(words_arr, candidate)
    
    return current_list

def construct():
    start_time = time.time()
    total_time_limit = 80  # Safety margin from 90s
    
    # Build the Reed-Muller code base (32 words)
    rm_words = _rm_code_18()
    
    best_code = rm_words
    best_size = len(rm_words)
    
    # Strategy 1: Vectorized greedy from RM code
    if time.time() - start_time < total_time_limit - 10:
        t1 = time.time()
        code1 = _vectorized_greedy(rm_words, 18, 6, time_limit=min(35, total_time_limit - (time.time() - start_time) - 5), seed=42)
        if len(code1) > best_size:
            best_code = code1
            best_size = len(code1)
    
    # Strategy 2: Systematic greedy from RM code with different seed
    if time.time() - start_time < total_time_limit - 10:
        t2 = time.time()
        code2 = _systematic_greedy(rm_words, 18, 6, time_limit=min(30, total_time_limit - (time.time() - start_time) - 5), seed=123)
        if len(code2) > best_size:
            best_code = code2
            best_size = len(code2)
    
    # Strategy 3: Try from a different starting point if time allows
    if time.time() - start_time < total_time_limit - 15:
        # Try a different initial code: start with just the all-zero word and greedy
        start_words = [0]
        code3 = _vectorized_greedy(start_words, 18, 6, time_limit=min(25, total_time_limit - (time.time() - start_time) - 5), seed=999)
        if len(code3) > best_size:
            best_code = code3
            best_size = len(code3)
    
    return best_code
# EVOLVE-BLOCK-END
