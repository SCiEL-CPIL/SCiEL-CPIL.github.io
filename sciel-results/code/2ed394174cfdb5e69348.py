"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TIME_LIMIT = 80  # seconds, safety margin before 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _generate_reed_muller_seed():
    """Generate the 32-word first-order Reed-Muller code on 16 coords, extended by 0 bit."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit is 0
            words.append(word)
    return words

def _construct_linear_code_seed():
    """
    Construct the 256-word [17,9,6] linear code (shortened [18,9,6] QR code).
    We use a known generator matrix for the shortened QR code.
    The [18,9,6] QR code has generator matrix G18. We shorten by one coordinate.
    
    For the purpose of this task, we'll use a systematic form if possible,
    or a known set of 256 words. Since we need to be fast, let's use a 
    structured approach: the RM(1,4) code is [16,5,8]. That's not it.
    
    Actually, the [17,9,6] code is not Reed-Muller. 
    Let's use a simpler seed: the 32-word RM code is safe. 
    But to get a better seed, we can try to construct a [17, k, 6] code.
    
    However, constructing the specific QR code is complex. 
    Let's stick to the 32-word RM seed but use a smarter expansion.
    """
    return _generate_reed_muller_seed()

def _greedy_expand_structured(seed_words, rng, time_deadline):
    """
    Structured greedy expansion:
    1. Start with seed.
    2. Instead of random candidates, we generate candidates by flipping bits 
       of existing codewords in a structured way, or by using a priority queue 
       of 'promising' candidates.
       
    A better approach for this specific problem:
    Use the fact that we want distance >= 6.
    We can precompute the set of all words at distance >= 6 from the current code?
    No, that's too big.
    
    Alternative: 
    1. Compute the 'exclusion zone' around each existing codeword (all words within dist 5).
    2. The valid candidates are the complement of this zone.
    3. We can iterate through the universe [0, 2^17) and check if a word is in the exclusion zone.
    
    To make this fast:
    - Maintain a boolean array `forbidden` of size 2^17.
    - When we add a word `w`, we mark all `v` such that dist(w, v) <= 5 as forbidden.
    - How many such `v`? Sum_{i=0}^5 C(17, i) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402.
    - So each addition costs ~9402 operations to update the `forbidden` array.
    - Total candidates: 131072.
    - If we add ~100 words, that's 100 * 9402 = 940,200 operations to update.
    - Then we scan the 131072 candidates to find the next valid one.
    - This is O(N_words * 9402 + N_universe * N_words) roughly? No, the scan is O(N_universe) per addition if we just check the array.
    - Actually, we can just scan the `forbidden` array to find the first `False`? No, we need to pick one.
    - We can maintain a list of available candidates? That's memory heavy but 131k bools is fine.
    
    Let's implement the 'forbidden zone' approach.
    """
    words = list(seed_words)
    words_set = set(words)
    universe_size = 1 << N
    forbidden = np.zeros(universe_size, dtype=bool)
    
    # Precompute popcount for distance calculation
    popcount = np.array([i.bit_count() for i in range(universe_size)], dtype=np.uint8)
    
    # Helper to mark forbidden zone for a given word
    def mark_forbidden(w):
        # Find all v such that popcount[w ^ v] <= 5
        # This is equivalent to v = w ^ d where popcount(d) <= 5
        # We can precompute the list of all 'd' vectors with weight <= 5.
        pass
        
    # Precompute all deltas with weight <= 5
    deltas = []
    for d in range(universe_size):
        if popcount[d] <= 5:
            deltas.append(d)
    deltas = np.array(deltas, dtype=np.uint32)
    
    # Initialize forbidden for seed words
    for w in words:
        # Mark w and all neighbors within dist 5 as forbidden
        # Note: we only care about words NOT in the code being forbidden? 
        # Actually, if a word is in the code, it's already "taken". 
        # We want to find NEW words. 
        # So `forbidden` should mark all words that CANNOT be added.
        # Words in `words` are already in the code, so they are effectively forbidden for re-adding.
        # But more importantly, any word within dist 5 of an existing word is forbidden.
        
        # Vectorized marking:
        # neighbors = (w ^ deltas)
        # forbidden[neighbors] = True
        neighbors = (w ^ deltas).astype(np.uint32)
        forbidden[neighbors] = True
        
    # Now, greedily add words.
    # We need to pick candidates. 
    # Strategy: Pick the word with the highest "margin" or just the first available?
    # To be deterministic and fast, we can scan the universe in a fixed order or random order.
    # Random order might be better to avoid local minima? 
    # But scanning the whole universe each time is 131k checks. 
    # If we add 100 words, that's 13M checks. Fast enough.
    
    # Let's use a random permutation of the universe to pick candidates.
    # But we don't want to re-shuffle every time. 
    # We can just iterate through a shuffled list of all indices.
    
    indices = np.arange(universe_size)
    rng.shuffle(indices)
    
    # We'll iterate through indices. If index is not forbidden, add it.
    # But wait, if we add a word, we must update `forbidden` for its neighbors.
    # This means a previously "available" word might become forbidden.
    # So we must check `forbidden[idx]` at the time of consideration.
    
    for idx in indices:
        if time.time() > time_deadline - 5:
            break
            
        if forbidden[idx]:
            continue
            
        # Candidate is valid
        w = int(idx)
        words.append(w)
        words_set.add(w)
        
        # Update forbidden zone
        neighbors = (w ^ deltas).astype(np.uint32)
        forbidden[neighbors] = True
        
    return words

def _local_search_swap(words, rng, time_deadline):
    """
    Try to improve by removing one word and adding two.
    Uses the forbidden zone concept for efficiency.
    """
    words = list(words)
    universe_size = 1 << N
    popcount = np.array([i.bit_count() for i in range(universe_size)], dtype=np.uint8)
    deltas = np.array([d for d in range(universe_size) if popcount[d] <= 5], dtype=np.uint32)
    
    def rebuild_forbidden(word_list):
        forbidden = np.zeros(universe_size, dtype=bool)
        for w in word_list:
            neighbors = (w ^ deltas).astype(np.uint32)
            forbidden[neighbors] = True
        return forbidden
    
    best_words = words
    best_size = len(words)
    
    # Try a few random swaps
    for _ in range(10): # Limit iterations
        if time.time() > time_deadline - 5:
            break
            
        if len(best_words) < 2:
            break
            
        # Pick a random word to remove
        idx = rng.randint(0, len(best_words) - 1)
        removed = best_words[idx]
        
        # Temporarily remove
        temp_words = best_words[:idx] + best_words[idx+1:]
        
        # Rebuild forbidden for temp_words
        forbidden = rebuild_forbidden(temp_words)
        
        # Try to find 2 new words
        added = []
        # Scan a random subset of candidates to save time? Or scan all?
        # Scanning all 131k is fast (~1ms). 
        indices = np.arange(universe_size)
        rng.shuffle(indices)
        
        for idx_cand in indices:
            if len(added) == 2:
                break
            if forbidden[idx_cand]:
                continue
            # Candidate is valid
            w = int(idx_cand)
            # Add it to temp_words and update forbidden? 
            # For the purpose of finding the 2nd word, we must account for the 1st added word.
            temp_words.append(w)
            added.append(w)
            # Update forbidden for the new word
            neighbors = (w ^ deltas).astype(np.uint32)
            forbidden[neighbors] = True
            
        if len(added) == 2:
            new_words = temp_words
            if len(new_words) > len(best_words):
                best_words = new_words
                best_size = len(new_words)
        # Else revert (best_words unchanged)
        
    return best_words

def construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    rng = random.Random(42)
    
    # Start with the Reed-Muller seed (32 words)
    seed = _generate_reed_muller_seed()
    
    best_words = seed
    best_size = len(seed)
    
    # Strategy: Structured Greedy Expansion
    # This is much faster than the naive greedy because it uses vectorized forbidden zone updates.
    # We can run it multiple times with different random orderings.
    
    for trial in range(10): # More trials possible due to speed
        if time.time() > time_deadline - 10:
            break
        
        rng_trial = random.Random(1000 + trial)
        words_trial = _greedy_expand_structured(seed, rng_trial, time_deadline - 5)
        
        if len(words_trial) > best_size:
            best_words = words_trial
            best_size = len(words_trial)
            
        # If we already have a very large code, stop early?
        if best_size > 200: # Heuristic
            break

    # Strategy: Local Search Improvement
    if time.time() < time_deadline - 10 and len(best_words) > 50:
        rng_ls = random.Random(2000)
        improved_words = _local_search_swap(best_words, rng_ls, time_deadline - 5)
        if len(improved_words) > best_size:
            best_words = improved_words
            best_size = len(improved_words)
    
    # Final verification and cleanup
    seen = set()
    valid_words = []
    for w in best_words:
        if w not in seen:
            seen.add(w)
            valid_words.append(w)
    
    if len(valid_words) < 2:
        return valid_words if valid_words else [0]
    
    return valid_words
