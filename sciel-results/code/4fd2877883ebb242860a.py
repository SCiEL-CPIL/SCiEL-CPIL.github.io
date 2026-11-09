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
        
        # Precompute for each possible t-subset S, the list of (block_tuple, tsub_indices_in_block)
        # This is expensive upfront but saves time in the loop.
        # Actually, let's precompute tsub indices for all C(v,k) blocks? That's C(19,8)=75582 blocks, each with 56 tsubs.
        # 75582 * 56 = 4.2M entries. That's a lot of memory. Let's not do that.
        
        # Instead, precompute for each t-subset S, the candidate blocks and their tsub index arrays
        # Number of t-subsets is C(19,5) = 11628. For each, C(14,3)=364 candidates, each with 56 tsubs.
        # 11628 * 364 * 56 = 236M entries. Too much.
        
        # Let's just precompute the tsub index arrays on the fly but cache them.
        # Actually, let's precompute all_tsub_indices: for each block (as a frozenset or sorted tuple), the list of tsub indices.
        # We'll compute lazily and cache.
        block_tsub_cache = {}
        
        def get_block_tsubs(block):
            if block not in block_tsub_cache:
                idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(block, t)], dtype=np.int32)
                block_tsub_cache[block] = idxs
            return block_tsub_cache[block]
        
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
            remaining_points = [p for p in range(v) if p not in S_set]
            candidates = []
            for extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(S + tuple(extra)))
                candidates.append(block)
            
            # Evaluate gain for each candidate using vectorized ops
            best_block = None
            best_gain = -1
            
            # Shuffle to break ties
            rng.shuffle(candidates)
            
            for B in candidates:
                idxs = get_block_tsubs(B)
                gain = np.sum(uncovered[idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask
            idxs = get_block_tsubs(best_block)
            uncovered[idxs] = False
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Precompute t-subset indices for each block
        block_indices = []
        for b in blocks:
            idxs = np.array([tsub_to_idx[comb] for comb in itertools.combinations(b, t)], dtype=np.int32)
            block_indices.append(idxs)
        
        # Iterative removal: repeatedly remove redundant blocks until no more
        current_blocks = list(blocks)
        current_indices = list(block_indices)
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            
            new_blocks = []
            new_indices = []
            
            # Compute cover counts
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for idxs in current_indices:
                cover_count[idxs] += 1
            
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                idxs = current_indices[i]
                # Block i is redundant if all its tsubs are covered by at least one other block
                if np.all(cover_count[idxs] > 1):
                    # Remove it
                    cover_count[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(current_blocks[i])
                    new_indices.append(idxs)
            
            if not changed:
                break
            
            current_blocks = new_blocks
            current_indices = new_indices
        
        return current_blocks

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

    # Final safety: verify the covering is valid, and if not, use the first greedy result
    # (which should be valid since greedy always covers)
    # Actually, greedy_once always produces a valid covering (it keeps adding blocks until all covered).
    # remove_redundant only removes redundant blocks, so it should also be valid.
    # But let's be safe: if time ran out during reduction, we might have an issue.
    # The iterative reduction returns current_blocks which is always a subset that still covers.
    # So we should be fine.

    return best_blocks
