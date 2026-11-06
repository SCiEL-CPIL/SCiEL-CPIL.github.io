"""Construct largest binary code of length 17, distance >= 6."""
import numpy as np
import time
import random

N = 17
MIN_DIST = 6
TOTAL_WORDS = 1 << N  # 131072
DEADLINE_SECONDS = 85.0  # safety margin under 90s

def construct():
    start_time = time.time()
    
    # Precompute all words as bit arrays: shape (131072, 17), bool
    # word i has bit j set iff (i >> j) & 1
    idx = np.arange(TOTAL_WORDS, dtype=np.uint32)
    # Build bit matrix: bits[j] = (idx >> j) & 1
    bits = np.zeros((TOTAL_WORDS, N), dtype=np.uint8)
    for j in range(N):
        bits[:, j] = (idx >> j) & 1
    
    best_words = []
    best_size = 0
    
    # We'll do multiple restarts. Each restart: greedy MRV packing.
    # To make it faster, we can precompute the popcount of XOR for pairs
    # But full matrix is too big. Instead, we compute distances in batches.
    
    # Strategy: maintain a set of "available" candidates (indices).
    # At each step, for each available candidate, compute its distance to all
    # already-selected words. If min distance >= MIN_DIST, it's a valid addition.
    # MRV: among valid candidates, pick the one that "conflicts" with the fewest
    # other available candidates (i.e., has the fewest available candidates at
    # distance < MIN_DIST from it).
    
    # To speed up: precompute for each word, the list of words at distance < 6.
    # Actually, that's still expensive. Let's use a different approach.
    
    # Alternative: random greedy with MRV approximation.
    # At each step, sample a batch of candidates, compute their distances to
    # selected set, filter valid ones, then among valid ones pick the one with
    # smallest "degree" (number of available neighbors within dist < MIN_DIST).
    
    # Even simpler: use a priority-based greedy where we precompute a "density"
    # score for each word and pick in order, skipping conflicts.
    
    # Let me try: multiple random orderings, greedy first-fit.
    # For each ordering, iterate through candidates, add if compatible with all selected.
    
    # To make MRV work: at each step, among all remaining compatible candidates,
    # pick the one with fewest compatible neighbors remaining.
    # This is expensive per step. Let's approximate: pick randomly from compatible,
    # but bias toward low-degree nodes.
    
    # Actually, let's just do: random permutation greedy, many restarts.
    # And use vectorized distance checks.
    
    def greedy_random_order(order, deadline):
        """Greedy: iterate through order, add word if compatible with all selected."""
        selected = []
        selected_bits = None  # shape (k, N) uint8
        available = list(order)
        
        for i in range(len(available)):
            if time.time() > deadline:
                break
            w_idx = available[i]
            w_bits = bits[w_idx]
            
            if len(selected) == 0:
                selected.append(w_idx)
                selected_bits = w_bits[np.newaxis, :]
                continue
            
            # Check distance to all selected: min distance must be >= MIN_DIST
            # XOR and popcount
            diff = selected_bits ^ w_bits[np.newaxis, :]
            dists = diff.sum(axis=1)
            if dists.min() >= MIN_DIST:
                selected.append(w_idx)
                selected_bits = np.vstack([selected_bits, w_bits[np.newaxis, :]])
        
        return selected
    
    # Better: use a conflict-based approach.
    # Precompute for each word, the set of words at distance < MIN_DIST.
    # This is the "conflict graph". We want a large independent set.
    # Greedy: repeatedly pick a vertex, remove it and all its neighbors.
    # MRV variant: pick the vertex with smallest degree in the remaining graph.
    
    # The conflict graph has 131072 vertices. Edge if dist < 6.
    # Degree of each vertex: number of words within distance 5.
    # For a 17-bit word, the number of words at distance exactly d is C(17,d).
    # Sum for d=0..5: C(17,0)+C(17,1)+C(17,2)+C(17,3)+C(17,4)+C(17,5)
    # = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402
    # So each vertex has degree ~9402 (including itself). That's manageable.
    
    # Precompute adjacency: for each word, list of words at distance < 6.
    # This is 131072 * 9402 ≈ 1.2 billion entries. Too much memory.
    
    # Alternative: don't precompute. Compute on the fly in batches.
    
    # Let's try: MRV greedy where we compute degrees dynamically.
    # At each step:
    #   1. For all remaining candidates, compute min distance to selected set.
    #   2. Filter to those with min_dist >= MIN_DIST (compatible).
    #   3. Among compatible, compute "degree" = number of other compatible candidates
    #      at distance < MIN_DIST from this one.
    #   4. Pick the one with smallest degree.
    #   5. Remove it and all its neighbors (at dist < MIN_DIST) from remaining.
    
    # Step 3 is expensive: O(|compatible|^2 * N). With |compatible| up to 131072,
    # that's ~1.7e10 * 17 ≈ too slow.
    
    # Simplification: skip step 3, just pick randomly from compatible.
    # Or: approximate degree by a precomputed static "density" score.
    
    # Let's just do random-order greedy with many restarts, vectorized.
    
    def greedy_order(order, deadline):
        selected = []
        selected_bits = None
        remaining = np.array(order, dtype=np.int32)
        
        while len(remaining) > 0:
            if time.time() > deadline:
                break
            
            if len(selected) == 0:
                # Pick first
                w_idx = remaining[0]
                selected.append(w_idx)
                selected_bits = bits[w_idx][np.newaxis, :]
                remaining = remaining[1:]
                continue
            
            # Compute distances from all remaining to all selected
            # remaining_bits: (m, N), selected_bits: (k, N)
            # We want min over k of popcount(remaining_bits[i] ^ selected_bits[j])
            # For memory: m can be up to 131072, k up to ~300.
            # m*k*N = 131072 * 300 * 17 ≈ 668M bytes ≈ 668 MB. Too much.
            
            # Process in batches of remaining.
            batch_size = 10000
            compatible = []
            for start in range(0, len(remaining), batch_size):
                end = min(start + batch_size, len(remaining))
                batch_idx = remaining[start:end]
                batch_bits = bits[batch_idx]  # (b, N)
                
                # Compute min distance to selected
                # (b, k, N) -> popcount -> (b, k) -> min over k
                # For memory, process selected in sub-batches if k is large
                min_dists = np.full(len(batch_idx), N + 1, dtype=np.int32)
                for s_start in range(0, len(selected), 100):
                    s_end = min(s_start + 100, len(selected))
                    sel_bits = selected_bits[s_start:s_end]  # (sk, N)
                    # XOR: (b, sk, N)
                    diff = batch_bits[:, np.newaxis, :] ^ sel_bits[np.newaxis, :, :]
                    dists = diff.sum(axis=2)  # (b, sk)
                    min_dists = np.minimum(min_dists, dists.min(axis=1))
                
                comp_mask = min_dists >= MIN_DIST
                if comp_mask.any():
                    compatible.extend(batch_idx[comp_mask].tolist())
            
            if not compatible:
                break
            
            # Pick randomly from compatible
            chosen = random.choice(compatible)
            selected.append(chosen)
            selected_bits = np.vstack([selected_bits, bits[chosen][np.newaxis, :]])
            
            # Remove chosen and its neighbors from remaining
            # Neighbors: words at distance < MIN_DIST from chosen
            # Compute for all remaining
            chosen_bits = bits[chosen]
            remove_mask = np.zeros(len(remaining), dtype=bool)
            for start in range(0, len(remaining), 10000):
                end = min(start + 10000, len(remaining))
                batch = remaining[start:end]
                batch_bits = bits[batch]
                diff = batch_bits ^ chosen_bits[np.newaxis, :]
                dists = diff.sum(axis=1)
                remove_mask[start:end] = dists < MIN_DIST
            
            remaining = remaining[~remove_mask]
        
        return selected
    
    # Do multiple restarts with random orderings
    num_restarts = 50
    deadline = start_time + DEADLINE_SECONDS
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = list(range(TOTAL_WORDS))
        random.shuffle(order)
        
        result = greedy_order(order, deadline)
        
        if len(result) > best_size:
            best_size = len(result)
            best_words = result
    
    # Fallback: if we got nothing, return the seed code
    if not best_words:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    return best_words
