import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    # Precompute all t-subsets and map to indices for fast lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)

    best_blocks = None
    best_len = float('inf')

    # Precompute for each point, which t-subsets contain it (for potential future use)
    # More importantly, precompute the list of t-subset indices for each possible block
    # That's C(v,k) blocks * C(k,t) t-subsets each. For v=24,k=7: C(24,7)=346104, too many.
    # Instead, we'll compute on the fly but use a faster method.

    # For greedy: we pick an anchor t-subset, then enumerate all k-subsets containing it.
    # For each candidate block, we need to count how many uncovered t-subsets it covers.
    # Key optimization: maintain a boolean array `is_uncovered` and use numpy for counting.

    def _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
        rng = random.Random(seed_val)
        
        # Use a numpy boolean array for uncovered status
        is_uncovered = np.ones(n_tsubs, dtype=bool)
        # Also maintain a list of uncovered indices for random selection
        uncovered_list = list(range(n_tsubs))
        uncovered_set = set(uncovered_list)
        
        blocks = []
        
        while uncovered_list:
            if time.time() > deadline:
                return None, False
            
            # Pick a random uncovered t-subset to anchor our search
            pos = rng.randrange(len(uncovered_list))
            anchor_idx = uncovered_list[pos]
            anchor_tsub = all_tsubsets[anchor_idx]
            anchor_set = set(anchor_tsub)
            
            # Generate all k-subsets containing anchor_tsub
            remaining_points = [x for x in range(v) if x not in anchor_set]
            
            best_gain = -1
            best_block = None
            best_tiebreakers = []
            
            # Precompute the t-subsets of the anchor to avoid recomputing
            anchor_tsubs = set(anchor_tsub)
            
            # Iterate over all combinations of (k-t) points from remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(list(anchor_tsub) + list(combo)))
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                gain = 0
                for tsub in itertools.combinations(candidate_block, t):
                    idx = tsub_to_idx[tsub]
                    if is_uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
                    best_tiebreakers = [candidate_block]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(candidate_block)
            
            if best_block is None or best_gain == 0:
                return None, False
                
            # If there are tiebreakers, pick one randomly
            chosen_block = rng.choice(best_tiebreakers)
            
            blocks.append(chosen_block)
            
            # Update uncovered status
            for tsub in itertools.combinations(chosen_block, t):
                idx = tsub_to_idx[tsub]
                if is_uncovered[idx]:
                    is_uncovered[idx] = False
                    uncovered_set.discard(idx)
            
            # Rebuild uncovered_list periodically or use a different structure
            # For efficiency, rebuild when list gets large
            if len(uncovered_set) < len(uncovered_list) * 0.7:
                uncovered_list = list(uncovered_set)
            else:
                # Remove the ones we just covered
                covered_this_round = set()
                for tsub in itertools.combinations(chosen_block, t):
                    covered_this_round.add(tsub_to_idx[tsub])
                uncovered_list = [x for x in uncovered_list if x not in covered_this_round]
        
        return blocks, True

    def _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
        if not blocks:
            return blocks
        
        # Build coverage count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                cover_count[idx] += 1
        
        improved = list(blocks)
        
        # 1. Remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(improved) - 1, -1, -1):
                block = improved[i]
                redundant = True
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for tsub in itertools.combinations(block, t):
                        idx = tsub_to_idx[tsub]
                        cover_count[idx] -= 1
                    improved.pop(i)
                    changed = True
        
        # 2. Try to replace a block with a better one (swap)
        # For each block, try to find a different k-subset that covers at least the same
        # essential t-subsets and possibly more, allowing us to keep the same count or reduce.
        # Actually, since we can't reduce count with a 1-for-1 swap unless we make another block redundant,
        # let's try: for each block B, find an alternative B' such that after replacing B with B',
        # some other block becomes redundant.
        
        # Simpler: try to find if any block can be replaced such that the total count stays the same
        # but we enable further removals. This is complex. Let's try a simpler heuristic:
        # For each block, try random alternatives and see if it helps.
        
        rng_local = random.Random(42)
        n_improved = len(improved)
        
        # Try swap: replace block i with a random k-subset, check if coverage is maintained
        # and if it enables removal of another block.
        for _iter in range(5):  # Limit iterations
            if time.time() > deadline:
                break
            
            made_progress = False
            for i in range(len(improved)):
                if time.time() > deadline:
                    break
                
                orig_block = improved[i]
                # Try a few random alternatives
                for _attempt in range(3):
                    # Generate a random k-subset
                    candidate = tuple(sorted(rng_local.sample(range(v), k)))
                    
                    # Check if replacing improved[i] with candidate maintains coverage
                    # First, remove coverage from orig_block
                    for tsub in itertools.combinations(orig_block, t):
                        idx = tsub_to_idx[tsub]
                        cover_count[idx] -= 1
                    
                    # Check if any t-subset becomes uncovered
                    # Find t-subsets of orig_block that are now uncovered
                    critical = []
                    for tsub in itertools.combinations(orig_block, t):
                        idx = tsub_to_idx[tsub]
                        if cover_count[idx] == 0:
                            critical.append(idx)
                    
                    if critical:
                        # Candidate must cover all critical t-subsets
                        cand_tsubs = set()
                        for tsub in itertools.combinations(candidate, t):
                            cand_tsubs.add(tsub_to_idx[tsub])
                        
                        # Check if candidate covers all critical
                        if all(idx in cand_tsubs for idx in critical):
                            # Apply the swap
                            for tsub in itertools.combinations(candidate, t):
                                idx = tsub_to_idx[tsub]
                                cover_count[idx] += 1
                            improved[i] = candidate
                            made_progress = True
                            break
                        else:
                            # Revert
                            for tsub in itertools.combinations(orig_block, t):
                                idx = tsub_to_idx[tsub]
                                cover_count[idx] += 1
                    else:
                        # No critical t-subsets, so we can swap freely
                        for tsub in itertools.combinations(candidate, t):
                            idx = tsub_to_idx[tsub]
                            cover_count[idx] += 1
                        improved[i] = candidate
                        made_progress = True
                        break
                
                # After a swap, try to remove redundant blocks again
                if made_progress:
                    break
            
            # Try to remove redundant blocks after swaps
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                for i in range(len(improved) - 1, -1, -1):
                    block = improved[i]
                    redundant = True
                    for tsub in itertools.combinations(block, t):
                        idx = tsub_to_idx[tsub]
                        if cover_count[idx] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        for tsub in itertools.combinations(block, t):
                            idx = tsub_to_idx[tsub]
                            cover_count[idx] -= 1
                        improved.pop(i)
                        changed = True
        
        return improved

    # Main loop
    trial = 0
    while time.time() < deadline:
        trial += 1
        
        # Use different seeds for diversity
        seed = trial
        
        blocks, valid = _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
        
        if valid:
            # Local improvement
            improved_blocks = _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
            
            if len(improved_blocks) < best_len:
                best_len = len(improved_blocks)
                best_blocks = improved_blocks
                
            # Early stop if we hit the known best (76)
            if best_len <= 76:
                break

    if best_blocks is None:
        # Fallback
        blocks, valid = _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, 0)
        if valid:
            best_blocks = blocks
            best_blocks = _local_improve(best_blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        else:
            best_blocks = []
            
    return best_blocks
