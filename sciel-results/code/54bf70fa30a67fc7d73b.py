import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute weights to filter candidates early?
    # Candidates must have weight >= MIN_DIST to be compatible with 0?
    # Not necessarily, but if 0 is in the code, yes.
    # We will always keep 0 in the code for simplicity and symmetry.
    
    # Initial Greedy Construction
    def vectorized_greedy():
        words = [0]
        words_set = {0}
        # Candidates: all words with weight >= MIN_DIST (since 0 is in code)
        # Actually, we can start with all words, but 0 blocks weight < 6.
        candidates = [c for c in range(1, 1 << N) if popcount[c] >= MIN_DIST]
        
        while time.time() < deadline - 5.0: # Leave time for local search
            if not candidates:
                break
            
            # Batch check: are any candidates compatible with all words?
            # We can check in chunks to avoid huge memory usage if W is large
            # But W is likely < 500, Candidates < 2^18.
            # Matrix size: 262144 x 500 = 130M elements. 
            # 130M uint8 = 130MB. This is borderline but might fit.
            # Let's check in chunks of 10000 candidates.
            
            added_any = False
            chunk_size = 10000
            for i in range(0, len(candidates), chunk_size):
                if time.time() > deadline - 5.0:
                    break
                
                chunk = candidates[i:i+chunk_size]
                if not chunk:
                    continue
                
                c_arr = np.array(chunk, dtype=np.uint32)
                w_arr = np.array(words, dtype=np.uint32)
                
                # Compute XORs: (len(chunk), len(words))
                # This creates a large array. 
                # Alternative: Loop over words and mask candidates?
                # For each w, candidates must be at dist >= 6.
                # incompatible = OR over w of (popcount[c^w] < 6)
                
                # Let's do it word by word to save memory, but vectorize over candidates.
                valid_mask = np.ones(len(chunk), dtype=bool)
                
                for w in words:
                    # popcount[c^w] for all c in chunk
                    xors = c_arr ^ w
                    dists = popcount[xors]
                    # Update valid_mask: c is invalid if dist < MIN_DIST
                    valid_mask &= (dists >= MIN_DIST)
                    
                    # Early exit if no valid candidates left
                    if not np.any(valid_mask):
                        break
                
                if np.any(valid_mask):
                    # Pick a random valid candidate from this chunk
                    valid_indices = np.where(valid_mask)[0]
                    idx = random.choice(valid_indices)
                    new_word = chunk[idx]
                    
                    words.append(new_word)
                    words_set.add(new_word)
                    
                    # Remove new_word and its "conflicts" from candidates?
                    # We can just filter candidates list.
                    # But filtering is O(N). 
                    # Better to keep candidates list and let the check fail.
                    # But we should remove the added word from candidates.
                    candidates.remove(new_word)
                    
                    added_any = True
                    # If we added one, we should restart the chunk loop?
                    # Because the set of words changed, so other candidates might now be invalid.
                    # But checking is cheap.
                    # However, the `valid_mask` calculation was based on OLD words.
                    # So we must restart the outer while loop or at least re-check.
                    # Let's just break and restart the while loop.
                    break
            
            if not added_any:
                # No candidate added in this pass. 
                # We might be stuck. Break greedy phase.
                break
                
        return words, candidates

    # Local Search with Vectorized Checks
    def local_search(words, candidates):
        words = list(words)
        candidates = list(candidates)
        words_set = set(words)
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # 1. Try to Add
            # Check all candidates against words
            # We can do this in chunks
            for i in range(0, len(candidates), 10000):
                if time.time() > deadline - 1.0:
                    break
                
                chunk = candidates[i:i+10000]
                if not chunk:
                    continue
                
                c_arr = np.array(chunk, dtype=np.uint32)
                w_arr = np.array(words, dtype=np.uint32)
                
                valid_mask = np.ones(len(chunk), dtype=bool)
                
                for w in words:
                    xors = c_arr ^ w
                    dists = popcount[xors]
                    valid_mask &= (dists >= MIN_DIST)
                    if not np.any(valid_mask):
                        break
                
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    idx = random.choice(valid_indices)
                    new_word = chunk[idx]
                    
                    words.append(new_word)
                    words_set.add(new_word)
                    candidates.remove(new_word)
                    improved = True
                    break # Restart outer loop
            
            if improved:
                continue
                
            # 2. Try to Swap
            # For each word w in words, see if we can swap it with a candidate
            # To speed up, we can limit swaps to a subset of words or candidates
            # Or just iterate through words.
            
            # Shuffle words to randomize swap order
            # But we need to keep track.
            for idx_w, w in enumerate(words):
                if time.time() > deadline - 1.0:
                    break
                
                # Temporarily remove w
                # Check candidates against words without w
                # This is expensive if we do it for every w.
                # Optimization: Only try swaps for words that are "central" or random.
                # Let's just try a few random swaps per iteration.
                
            # Alternative Swap Strategy:
            # Pick a random word w to remove.
            # Find a candidate c compatible with rest.
            # If found, swap.
            
            if time.time() < deadline - 1.0:
                # Pick a random word to remove
                idx_w = random.randint(0, len(words) - 1)
                w_to_remove = words[idx_w]
                
                # Check candidates against words excluding w_to_remove
                # We can create a view or just skip in the loop
                valid_mask = np.ones(len(candidates), dtype=bool)
                # This is O(N_candidates * N_words). 
                # If N_words is 100, N_candidates is 10000, that's 1M ops. Fast.
                
                c_arr = np.array(candidates, dtype=np.uint32)
                
                for w in words:
                    if w == w_to_remove:
                        continue
                    xors = c_arr ^ w
                    dists = popcount[xors]
                    valid_mask &= (dists >= MIN_DIST)
                    if not np.any(valid_mask):
                        break
                
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    idx_c = random.choice(valid_indices)
                    new_word = candidates[idx_c]
                    
                    # Perform Swap
                    words[idx_w] = new_word
                    words_set.discard(w_to_remove)
                    words_set.add(new_word)
                    
                    candidates[idx_c] = w_to_remove
                    # Swap in candidates list is O(1) if we just swap values
                    # But we need to maintain distinctness? 
                    # w_to_remove is now in candidates. new_word is in words.
                    # This is fine.
                    
                    improved = True
                    break # Restart loop

        return words

    # Execution
    words, candidates = vectorized_greedy()
    words = local_search(words, candidates)
    
    return words
