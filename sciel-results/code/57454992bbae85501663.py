"""Bitset-accelerated greedy with randomized redundancy reduction for C(24,7,3)."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s kill
    
    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Map each t-set to its index
    # For each point p, bitset of t-sets containing p
    point_to_tsets = np.zeros(n_tsets, dtype=np.int64)  # Not used directly, but structure helps
    # Actually, we need: for a candidate block B, which t-sets are covered?
    # tset is covered if all its elements are in B.
    
    # Precompute for each t-set, the bitmask of points it contains? No, v=24 fits in int.
    # Let's represent a t-set as a bitmask of points.
    tset_masks = []
    for ts in all_tsets:
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tset_masks.append(mask)
    tset_masks = np.array(tset_masks, dtype=np.int64)
    
    # For each point p, list of tset indices containing p
    # But for gain calculation:
    # Gain(B) = count of t-sets T such that T subset B and T is uncovered.
    # We can maintain an array `uncovered` of booleans.
    # To calculate gain of adding x to current block S:
    # New t-sets covered are those T subset (S U {x}) but not subset S.
    # These are exactly the t-sets that contain x and (t-1) elements from S.
    
    # Let's precompute for each pair (x, y) ... no, that's for t=2.
    # For t=3, gain of x given S is:
    # Sum over pairs (a,b) in S of: is (x,a,b) uncovered?
    
    # We can precompute for each pair (a,b) the index of the t-set (a,b,x) for each x?
    # No, let's just use the tset_index dictionary.
    
    # Optimization: Precompute for each point x, and each subset S of size t-1, 
    # the index of the t-set S U {x}.
    # Since v=24, t=3, k=7.
    # Number of pairs (a,b) is C(24,2) = 276.
    # For each pair, and each x not in pair, there is a t-set.
    
    # Let's create a lookup: pair_mask -> list of (x, tset_idx)
    # Or simpler: tset_idx[ (a,b,c) ]
    
    # We will maintain `uncovered` as a numpy boolean array of length n_tsets.
    
    def greedy_once(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks = []
        
        # Precompute for fast gain calculation
        # For each point x, we want to quickly check which t-sets containing x are uncovered.
        # A t-set T is covered by block B if T subset B.
        
        # Let's precompute for each t-set T, the bitmask of points.
        # Then T subset B is (T_mask & ~B_mask) == 0.
        
        # But we need to do this for many candidates x.
        # Let's precompute for each point x, the list of tset indices containing x.
        tsets_by_point = [[] for _ in range(v)]
        for i, ts in enumerate(all_tsets):
            for p in ts:
                tsets_by_point[p].append(i)
        
        # To calculate gain of x given current block S (list of points):
        # We need to count uncovered t-sets T such that x in T and T \ {x} subset S.
        # Let S_mask be bitmask of S.
        # For each x, iterate through tsets_by_point[x].
        # For each tset_idx, check if (tset_masks[tset_idx] & ~S_mask) == 0 and uncovered[tset_idx].
        # This is still O(v * t) per candidate, which is small.
        
        # Even better: Precompute for each x, an array of tset indices.
        # And use numpy to check masks?
        # tset_masks is int64. S_mask is int.
        # (tset_masks[tsets_by_point[x]] & ~S_mask) == 0
        # This vectorizes nicely.
        
        # Let's precompute tset_masks_by_point: list of arrays of tset masks for each point.
        tset_masks_by_point = []
        for p in range(v):
            idxs = tsets_by_point[p]
            tset_masks_by_point.append(tset_masks[idxs])
        
        # And tset_idxs_by_point
        tset_idxs_by_point = [np.array(idxs, dtype=np.int64) for idxs in tsets_by_point]
        
        deadline = start_time + time_limit
        current_block = []
        
        # Initial block: pick a random uncovered t-set? Or just start empty?
        # Standard greedy: pick block with max gain.
        # But starting from empty, gain of any 7-set is C(7,3)=35 if all uncovered.
        # Let's start by picking a random point to break symmetry?
        # Or just pick the first candidate with max gain.
        
        # To add diversity, we can randomize the tie-breaking or the starting point.
        # Let's start with a random point.
        if v > 0:
            first_point = rng.randint(0, v - 1)
            current_block.append(first_point)
        
        while len(current_block) < k:
            S_mask = 0
            for p in current_block:
                S_mask |= (1 << p)
            
            best_gain = -1
            best_x = -1
            best_score = -1.0
            
            for x in range(v):
                if x in current_block:
                    continue
                
                # Calculate gain
                # tsets containing x: tset_idxs_by_point[x]
                # masks: tset_masks_by_point[x]
                # Condition: (mask & ~S_mask) == 0
                # And uncovered[idx]
                
                idxs = tset_idxs_by_point[x]
                masks = tset_masks_by_point[x]
                
                # Vectorized check
                # ~S_mask in int64. Careful with sign bits? v=24, so bits 0-23 used.
                # ~S_mask will have 1s in high bits.
                # mask & ~S_mask. If mask has 0 in high bits, result is 0 in high bits.
                # So (mask & ~S_mask) == 0 works if we only care about low bits.
                # But mask is int64 positive. ~S_mask is negative.
                # It's safer to use: (mask | S_mask) == S_mask ? No.
                # (mask & S_mask) == mask ?
                # Yes, T subset S iff (mask & S_mask) == mask.
                
                # Wait, we want T subset (S U {x}).
                # And T contains x.
                # So T \ {x} subset S.
                # Let T_mask be mask.
                # T contains x, so x bit is set in T_mask.
                # We want (T_mask & ~(S_mask | (1<<x))) == 0.
                # Since x not in S, S_mask | (1<<x) is just S_mask with x bit added.
                # Let New_S_mask = S_mask | (1 << x).
                # Condition: (T_mask & ~New_S_mask) == 0.
                # Or (T_mask | New_S_mask) == New_S_mask.
                
                New_S_mask = S_mask | (1 << x)
                
                # Check subset
                # (masks & New_S_mask) == masks
                # This is vectorizable.
                is_subset = (masks & New_S_mask) == masks
                
                # Also need uncovered
                unc = uncovered[idxs]
                
                # Count
                gain = np.sum(is_subset & unc)
                
                # Diversity: penalize if x is already in many blocks?
                # For now, pure gain + random noise.
                score = gain + rng.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                break
                
            current_block.append(best_x)
            
            if time.time() > deadline:
                break
        
        if len(current_block) == k:
            current_block.sort()
            blocks.append(tuple(current_block))
            
            # Update uncovered
            B_mask = 0
            for p in current_block:
                B_mask |= (1 << p)
            
            # Find all t-sets subset of B
            # Iterate through all t-sets? No, too slow.
            # Iterate through t-sets containing each point in B?
            # Or just iterate through all t-sets and check mask?
            # n_tsets = C(24,3) = 2024.
            # 2024 checks is fast.
            
            for i in range(n_tsets):
                if uncovered[i]:
                    if (tset_masks[i] & B_mask) == tset_masks[i]:
                        uncovered[i] = False
            
            # Early exit if all covered
            if not np.any(uncovered):
                break
        
        return blocks
    
    def reduce_redundancy(blocks, seed, time_limit):
        """
        Iteratively remove blocks that are redundant.
        A block is redundant if removing it doesn't uncover any t-set.
        """
        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]
        deadline = start_time + time_limit
        
        # Build index: tset -> list of block indices
        # n_tsets = 2024
        tset_to_blocks = [[] for _ in range(n_tsets)]
        
        def build_index(blocks):
            for i in range(n_tsets):
                tset_to_blocks[i] = []
            for bi, b in enumerate(blocks):
                B_mask = 0
                for p in b:
                    B_mask |= (1 << p)
                for i in range(n_tsets):
                    if (tset_masks[i] & B_mask) == tset_masks[i]:
                        tset_to_blocks[i].append(bi)
        
        build_index(blocks)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle blocks to break symmetry
            # We can't just shuffle the list if we want to keep indices stable?
            # Let's just pick a random block to test.
            
            for bi in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                # Check if block bi is redundant
                # For each t-set covered by bi, is it covered by another block?
                b = blocks[bi]
                B_mask = 0
                for p in b:
                    B_mask |= (1 << p)
                
                redundant = True
                for i in range(n_tsets):
                    if (tset_masks[i] & B_mask) == tset_masks[i]:
                        # t-set i is covered by bi
                        # Check if covered by others
                        covering = tset_to_blocks[i]
                        if covering == [bi]:
                            redundant = False
                            break
                
                if redundant:
                    # Remove block bi
                    blocks.pop(bi)
                    build_index(blocks)
                    improved = True
                    break  # Restart loop with new indices
        
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy runs
    # We have ~100s budget.
    # Each greedy run is fast.
    # Let's try 50 runs.
    num_runs = 50
    per_run_time = 1.5  # 50 * 1.5 = 75s
    
    for i in range(num_runs):
        if time.time() - start_time > 100.0:
            break
        seed = 1000 + i
        blocks = greedy_once(seed, per_run_time)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        blocks = greedy_once(0, 10.0)
        best_blocks = blocks
        best_len = len(blocks)
    
    # Phase 2: Redundancy reduction on best and some others
    # Try to reduce the best one
    if time.time() - start_time < 140.0:
        reduced = reduce_redundancy(best_blocks, 42, 5.0)
        if len(reduced) < best_len:
            best_blocks = reduced
            best_len = len(reduced)
        
        # Try to reduce some other good ones?
        # Maybe re-run greedy with different seeds and reduce them?
        # Let's just do a few more greedy + reduce cycles.
        
        for i in range(10):
            if time.time() - start_time > 140.0:
                break
            seed = 2000 + i
            blocks = greedy_once(seed, 1.0)
            if len(blocks) < best_len + 2: # Only if promising
                reduced = reduce_redundancy(blocks, seed, 1.0)
                if len(reduced) < best_len:
                    best_blocks = reduced
                    best_len = len(reduced)
    
    return best_blocks

# Helper for debugging if needed, but not required by contract
if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Found covering with {len(blocks)} blocks")
    # Verify
    all_tsets = list(itertools.combinations(range(24), 3))
    covered = set()
    for b in blocks:
        bset = set(b)
        for ts in itertools.combinations(b, 3):
            if set(ts).issubset(bset):
                covered.add(ts)
    missing = [ts for ts in all_tsets if ts not in covered]
    print(f"Missing: {len(missing)}")
    if missing:
        print("First missing:", missing[0])
