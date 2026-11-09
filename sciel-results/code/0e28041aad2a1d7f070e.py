import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0  # Safety margin before 90s kill
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount for all 18-bit values
    # 2^18 = 262144
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Initial code
    words = [0]
    words_set = {0}
    
    # Candidates: all words with weight >= MIN_DIST (since 0 is in code)
    # This reduces the candidate pool significantly.
    # Weights 6 to 18.
    # Number of such words is sum(C(18, k) for k=6 to 18)
    # C(18,6) = 18564, C(18,7)=31824, ... C(18,18)=1
    # Total is roughly 2^18 / 2 = 131072. Still large, but manageable.
    
    # Instead of generating all candidates initially, we can generate them on the fly 
    # or just use the full range and filter. 
    # Let's use the full range 1..2^18-1 but filter by popcount >= 6 in the first pass.
    
    # To make greedy fast, we should pick candidates that are "far" from existing words.
    # A simple greedy:
    # 1. Pick a candidate.
    # 2. Check against all words.
    # 3. If valid, add.
    
    # Vectorized Greedy:
    # We maintain a list of candidates.
    # In each iteration, we take a chunk of candidates.
    # We check all of them against all current words.
    # We add all that are valid? Or just one?
    # Adding multiple at once is dangerous if they conflict with each other.
    # But if we check them against the *current* set, they might conflict with *each other*.
    # So we must add one by one, or use a greedy set packing approach.
    # Given the time limit, adding one by one with vectorized checks against the *current* set is safer.
    
    # Optimization: 
    # Instead of checking a chunk against all words, we can filter the chunk.
    # Let's create an array of all potential candidates.
    all_candidates = np.arange(1, 1 << N, dtype=np.uint32)
    # Filter by weight >= 6
    weights = popcount[all_candidates]
    all_candidates = all_candidates[weights >= MIN_DIST]
    
    # Shuffle to avoid bias
    np.random.shuffle(all_candidates)
    
    # We will process all_candidates in chunks.
    # For each chunk, we check compatibility with `words`.
    # If a candidate is compatible, we add it to `words` and remove it from `all_candidates` (conceptually).
    # Removing from a numpy array is slow. 
    # Better: Keep an index pointer or a boolean mask of "active" candidates.
    # Or simply: iterate through the shuffled list. 
    # If a candidate is compatible, add it. 
    # But checking compatibility with a growing `words` list is expensive if done in Python.
    
    # Let's use a different strategy:
    # 1. Greedy Phase:
    #    - Maintain `words` (list).
    #    - Maintain `candidates` (list of indices or values).
    #    - To speed up, we don't re-check all candidates every time.
    #    - We just iterate through the shuffled candidates.
    #    - For each candidate, check against `words`.
    #    - If valid, add to `words`.
    #    - This is O(N_candidates * N_words). 
    #    - With N_words growing, this becomes slow.
    
    # Improved Greedy:
    # Process candidates in batches.
    # For a batch of B candidates, compute a B x W matrix of distances.
    # Identify all candidates in the batch that are compatible with ALL current words.
    # From those, we can greedily pick a subset that are mutually compatible.
    # This is a Max Independent Set problem on a graph, which is hard.
    # However, if the code is sparse, we can just pick the first one, add it, and re-check the rest of the batch against the NEW set of words.
    
    # Let's stick to a simpler, robust method:
    # 1. Start with words = [0].
    # 2. Create a list of all valid candidates (weight >= 6).
    # 3. Shuffle.
    # 4. Iterate through the list.
    #    - For candidate c, check if min(dist(c, w) for w in words) >= 6.
    #    - If yes, add c to words.
    #    - This is O(N_candidates * N_words).
    #    - To make this fast, we can vectorize the check for a single candidate against all words.
    #    - `dists = popcount[c ^ np.array(words)]`
    #    - `if np.min(dists) >= 6: words.append(c)`
    #    - This is fast per candidate.
    #    - Total time: Sum_{w in words} (cost to check next candidate).
    #    - If we find 1000 words, the last check takes 1000 ops.
    #    - 262144 candidates * average 500 words = 131M ops.
    #    - NumPy can do 100M+ ops per second. So this should fit in 55s.
    
    # Let's implement this "Iterative Vectorized Check".
    
    words_arr = np.array([0], dtype=np.uint32)
    
    # Pre-filtered candidates
    # We can also filter candidates that are too close to 0? 
    # We already did weight >= 6.
    
    # To speed up further, we can randomize the order of candidates.
    # We will process them one by one.
    
    # Optimization: 
    # If `words` becomes large, the check `popcount[c ^ words_arr]` is a vector operation of size len(words).
    # This is very fast in NumPy.
    
    # Let's try to add as many as possible.
    
    # We need to be careful with the time limit.
    # If we run out of time, we return what we have.
    
    # Additional Idea: 
    # If the greedy phase is too slow, we can limit the number of candidates we check.
    # But 262k candidates is not too many.
    
    # Let's also add a Local Search phase to improve the code.
    # Local Search: 
    # 1. Try to add a random candidate from the rejected list.
    # 2. Try to swap a word with a candidate.
    
    # For now, let's just do the Greedy phase efficiently.
    
    # We need to track which candidates we've already rejected?
    # No, we just iterate through the shuffled list.
    # If a candidate was rejected because it was close to word W, it might become valid if W is removed (in swap phase).
    # But in greedy, we only add.
    
    # Let's store rejected candidates for the local search phase.
    rejected = []
    
    for c in all_candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Check c against words_arr
        xors = c ^ words_arr
        dists = popcount[xors]
        
        if np.min(dists) >= MIN_DIST:
            # Add c
            words_arr = np.append(words_arr, c)
        else:
            # Reject c
            # We might want to keep some rejected candidates for local search
            # But storing all 200k is fine.
            rejected.append(c)
            
    # Convert to list for return
    words_list = words_arr.tolist()
    
    # Local Search Phase
    # Try to improve the code by swapping or adding.
    # Since we already added all possible candidates in the greedy pass (in a specific order), 
    # there are no candidates compatible with the *final* set of words that we haven't seen?
    # No, we saw them in a specific order. 
    # A candidate rejected early might be compatible with the final set if some words that blocked it were... 
    # Wait, we only ADD words. We never remove.
    # So if a candidate c was rejected because dist(c, w_i) < 6, and w_i is still in the code, c is still invalid.
    # Since we never remove words in the greedy phase, any rejected candidate is still invalid against the final code.
    # So the greedy phase is actually "Maximal" in the sense that no single candidate can be added.
    # It is not necessarily "Maximum".
    
    # To find a larger code, we need to REMOVE some words to allow adding MORE words.
    # This is the Swap phase.
    
    # Let's implement a simple Swap Local Search.
    # 1. Pick a random word w to remove.
    # 2. Check if any candidate c (from rejected or remaining) is compatible with words \ {w}.
    # 3. If yes, replace w with c.
    # 4. Repeat.
    
    # We need a list of candidates to try.
    # We can use the `rejected` list.
    # We can also generate new random candidates.
    
    # To make this fast, we should vectorize the check.
    # But checking against `words \ {w}` is tricky with NumPy if we do it naively.
    # We can just loop over `words` and skip `w`.
    # Or create a view.
    
    # Given the time budget, let's do a few iterations of Swap.
    
    # Shuffle rejected
    random.shuffle(rejected)
    
    # We will try to swap.
    # To avoid O(N_words) Python loops, we can try to batch the swap checks?
    # Hard to batch "swap w with c" for many c.
    
    # Let's try a simpler improvement:
    # Random Restart?
    # No, time is limited.
    
    # Let's just return the greedy result. It is likely much better than the seed.
    # The seed was 7.3. This should be hundreds.
    
    # One more optimization: 
    # The greedy order matters. 
    # By shuffling, we get a random maximal code.
    # The size of a random maximal code is usually close to the optimal for these parameters.
    
    return words_list
