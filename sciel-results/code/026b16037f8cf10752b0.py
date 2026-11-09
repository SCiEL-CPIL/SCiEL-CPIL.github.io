import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a fast greedy approach 
    selecting the best k-superset for a random uncovered t-subset, 
    followed by redundancy removal and multi-start optimization.
    """
    deadline = time.time() + 110.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each t-subset, the list of k-subsets that contain it?
    # Or just generate them on the fly.
    # Number of k-subsets containing a t-subset is C(v-t, k-t).
    # For v=19, k=8, t=5: C(14, 3) = 364.
    # Generating 364 combinations and checking 56 t-subsets each is ~20k ops.
    # This is fine.

    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            S = all_tsubs[start_idx]
            S_set = set(S)
            
            # Generate all k-subsets containing S
            # Points not in S
            remaining_points = [p for p in range(v) if p not in S_set]
            # We need to choose k-t points from remaining_points
            # k-t = 3
            candidates = []
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(S + tuple(extra)))
                candidates.append(block)
            
            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1
            
            # Optimization: Shuffle candidates to break ties randomly
            rng.shuffle(candidates)
            
            for B in candidates:
                # Calculate gain: number of uncovered t-subsets in B
                # B has C(8,5)=56 t-subsets
                gain = 0
                # Using a set for faster lookup? No, we have index map.
                # We can vectorize this if we precompute indices for all blocks? 
                # But we don't have all blocks precomputed.
                # Let's just loop. 56 iterations is small.
                for comb in itertools.combinations(B, t):
                    idx = tsub_to_idx[comb]
                    if uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
                    # If we found a perfect cover for this step (unlikely but possible if few uncovered), 
                    # we could break early if gain == number of uncovered? No, just max gain.
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask
            for comb in itertools.combinations(best_block, t):
                idx = tsub_to_idx[comb]
                uncovered[idx] = False
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Count coverage of each t-subset
        # Use a list of counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        
        # Precompute t-subset indices for each block to speed up updates
        block_indices = []
        for b in blocks:
            idxs = [tsub_to_idx[comb] for comb in itertools.combinations(b, t)]
            block_indices.append(idxs)
            cover_count[idxs] += 1
        
        valid_blocks = []
        valid_indices = []
        
        for i, b in enumerate(blocks):
            if time.time() > deadline:
                # Return what we have so far, but it might be invalid if we stopped mid-way?
                # The prompt says "Valid when every t-subset... lies in at least one block".
                # If we break early in removal, the remaining blocks still cover everything 
                # because we only remove if redundant.
                # So returning current valid_blocks + rest? 
                # Actually, if we break, we haven't processed all blocks.
                # The unprocessed blocks are still in the "candidate" set.
                # It is safer to just return the original blocks if we run out of time, 
                # or the partially reduced set if we are sure it's valid.
                # Since we only remove redundant blocks, the set of kept blocks + unprocessed blocks 
                # is valid. But we don't have the unprocessed ones in `valid_blocks`.
                # Let's just break and return the original list if time runs out during reduction.
                return blocks
            
            # Check if block i is redundant
            # It is redundant if for all idx in block_indices[i], cover_count[idx] > 1
            # (because block i itself contributes 1 to the count)
            idxs = block_indices[i]
            if np.all(cover_count[idxs] > 1):
                # Redundant: remove it
                cover_count[idxs] -= 1
            else:
                # Keep it
                valid_blocks.append(b)
                valid_indices.append(idxs)
        
        return valid_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Multi-start
    # How many starts? 
    # One run takes ~0.5s. We have 110s. So ~200 starts possible.
    # But reduction takes time too.
    # Let's try 50 starts first, then use remaining time for more if needed.
    
    num_starts = 0
    start_time = time.time()
    
    # We will loop until time runs out
    while time.time() < deadline - 5.0: # Leave 5s for final reduction
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        # Reduce
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            
        num_starts += 1
        
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)

    return best_blocks
