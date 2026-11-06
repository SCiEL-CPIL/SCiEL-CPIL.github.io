import time
import numpy as np
from itertools import combinations

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0

# Precompute neighbor lists for all possible words? 
# 2^18 * (1 + 18 + 153 + 816 + 3060 + 8568) = 262144 * 12616 ~ 3.3 billion integers. Too much memory.
# Instead, we compute neighbors on the fly but try to vectorize the batch selection.

def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _get_neighbors(center, radius):
    """Generate all words within Hamming distance <= radius of center."""
    results = [center]
    bits = list(range(N))
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results

def _hamming_dist_matrix(a, b):
    """
    Compute pairwise Hamming distances between rows of a and rows of b.
    a: (Na, N) array of bits
    b: (Nb, N) array of bits
    Returns: (Na, Nb) array of distances
    """
    # a * (1 - b) + (1 - a) * b
    # To avoid overflow or slowness, we can use:
    # sum(a, axis=1)[:, None] + sum(b, axis=1)[None, :] - 2 * (a @ b.T)
    sum_a = a.sum(axis=1, keepdims=True)
    sum_b = b.sum(axis=1, keepdims=True).T
    dot = a @ b.T
    return sum_a + sum_b - 2 * dot

def _vectorized_greedy(deadline, seed=12345):
    words = _seed_words()
    
    # Start with seed words
    # We will maintain a list of accepted words.
    # To speed up, we will try to find a large batch of new words that are mutually compatible 
    # and compatible with existing words.
    
    # Initial forbidden mask logic is hard to vectorize for "mutual" exclusivity in a batch 
    # without checking all pairs in the batch.
    # However, we can generate a large pool of candidates, filter against current code, 
    # then solve a maximum independent set (MIS) on the conflict graph of the remaining candidates.
    # MIS is NP-hard, but for small batches (e.g., 1000 candidates), a greedy heuristic on the 
    # conflict matrix is fast.
    
    # Strategy:
    # 1. Generate a large batch of random candidates.
    # 2. Filter out those that conflict with existing `words`.
    # 3. Among the survivors, greedily select a subset that are mutually compatible.
    # 4. Add them to `words`.
    # 5. Repeat until deadline.
    
    # To make this fast, we represent `words` as a bit matrix for fast distance checks.
    # But `words` grows. 
    # Let's stick to the forbidden mask approach but update it in batches?
    # No, the forbidden mask is a boolean array of size 2^18.
    # Updating it for a batch of K words requires marking K * 12616 indices.
    # If K=100, that's 1.2M operations. Vectorized, this is fast.
    
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    # Precompute all indices
    all_indices = np.arange(TOTAL, dtype=np.int64)
    
    while time.time() < deadline:
        # Generate a batch of candidates
        # How big? If we generate 10000, checking against forbidden is fast (vectorized).
        # Then we need to check mutual compatibility among the ~100-1000 survivors.
        batch_size = 10000
        candidates = rng.choice(TOTAL, size=batch_size, replace=False)
        
        # Filter against existing forbidden mask
        valid_mask = ~forbidden[candidates]
        valid_candidates = candidates[valid_mask]
        
        if len(valid_candidates) == 0:
            # If no valid candidates in this random batch, we might be stuck or done.
            # Try a smaller batch or break if we've been stuck for a while?
            # For now, just continue.
            continue
            
        # Now we have a set of candidates that are all compatible with the current code.
        # We need to select a subset of these that are mutually compatible.
        # Let's greedily select from this batch.
        
        # To be efficient, we can just iterate through valid_candidates and add them 
        # one by one, updating a local "batch_forbidden" or just checking against 
        # the newly added ones?
        # Actually, since the batch is random, the density of conflicts is low.
        # We can just iterate and add if not conflicting with *newly added in this batch*.
        
        new_words = []
        # To avoid O(N^2) checks within the batch, we can just use the global forbidden mask 
        # and update it immediately? 
        # If we update global forbidden immediately, subsequent candidates in the batch 
        # will see the update. This is correct.
        
        # However, updating the global mask for every single word in the batch is expensive 
        # if we do it in a Python loop.
        # Let's collect the new words first, then update the mask once for the whole batch.
        # But we need to ensure they are mutually compatible.
        
        # Greedy selection from valid_candidates:
        # 1. Take first candidate c1. Add to new_words.
        # 2. Take next candidate c2. Check if dist(c2, c1) >= 6. If yes, add.
        # 3. This is O(K^2) distance checks. If K is 100, that's 10k checks. Fast.
        #    If K is 1000, that's 1M checks. Might be slow in Python.
        
        # Let's limit the number of candidates we try to pack in one batch to keep it fast.
        # Or, we can just iterate and add, updating a local set of "recently added" 
        # and checking distance only against those?
        # No, distance against ALL previously added in this batch is needed.
        
        # Let's try a simple loop for the batch selection.
        # We assume valid_candidates is sorted or random.
        # We'll just iterate.
        
        # Optimization: If the batch is large, we can subsample or just take the first M.
        # Let's just iterate through all valid_candidates.
        
        # To speed up the distance check, we can precompute the bit patterns?
        # Or just use the `forbidden` mask trick locally?
        # Let's just use a simple loop. If it's too slow, we reduce batch_size.
        
        # Actually, we can vectorize the mutual check!
        # Convert valid_candidates to a bit matrix?
        # valid_candidates is an array of ints.
        # We can compute pairwise distances using the matrix method above.
        
        # But we want a greedy set.
        # Greedy: pick one, remove all its neighbors from the candidate list, repeat.
        # This is efficient if we use the forbidden mask idea locally.
        
        # Let's do this:
        # 1. Create a local boolean array `local_forbidden` of size len(valid_candidates).
        # 2. Iterate:
        #    a. Find first index i where local_forbidden[i] is False.
        #    b. If none, break.
        #    c. Add valid_candidates[i] to new_words.
        #    d. For all j, if dist(valid_candidates[i], valid_candidates[j]) < 6, set local_forbidden[j] = True.
        
        # This is O(K^2) distance calculations in the worst case (if we pick many).
        # But we can vectorize the distance calculation for the picked word against all remaining.
        
        if len(valid_candidates) > 0:
            # We'll just do a simple loop here, it might be the bottleneck.
            # Let's try to vectorize the "mark neighbors" step.
            
            new_words = []
            # We need to check mutual distances.
            # Let's just iterate and maintain a list of accepted words in this batch.
            # For each candidate, check against all accepted in this batch.
            
            # To make this faster, we can break early if the batch is large.
            # Let's just do the simple thing first and see.
            
            # Actually, let's just use the global forbidden mask update in a vectorized way 
            # AFTER we select the batch.
            # So we need to select a mutually compatible subset from valid_candidates.
            
            # Greedy MIS on the conflict graph of valid_candidates.
            # We can represent the conflict graph as a matrix.
            # But building it is O(K^2).
            
            # Let's just do a simple greedy loop.
            # If K is 100, it's fine.
            
            # To avoid O(K^2) Python loops, we can use the fact that we only add ~10-20% of candidates.
            
            # Let's just iterate.
            accepted_in_batch = []
            # We will check distance against accepted_in_batch.
            
            # To speed up, we can break if we've added enough or if time is up.
            
            # Let's just use the simple loop.
            for c in valid_candidates:
                if time.time() > deadline:
                    break
                
                # Check against accepted_in_batch
                ok = True
                for w in accepted_in_batch:
                    if (c ^ w).bit_count() < MIN_DISTANCE:
                        ok = False
                        break
                if ok:
                    accepted_in_batch.append(c)
                    
            if accepted_in_batch:
                words.extend(accepted_in_batch)
                
                # Update global forbidden mask
                # We can do this in a vectorized way?
                # For each w in accepted_in_batch, get neighbors and mark.
                # This is a Python loop over accepted_in_batch.
                # If accepted_in_batch is small (e.g., 10), it's fine.
                for w in accepted_in_batch:
                    neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
                    forbidden[neighbors] = True

    return words

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN

    best_words = None
    best_len = 0

    seeds = [42, 123, 999, 7777, 31415, 27182, 16180, 8675309, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        try:
            w = _vectorized_greedy(deadline, seed=seed)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    if best_words is None:
        best_words = _seed_words()

    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()

    return out
