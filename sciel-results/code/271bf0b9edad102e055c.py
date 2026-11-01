"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Precompute t-subsets for speed in the specific case if needed, 
# but we keep it generic as per contract.
# For v=19, t=5, k=8:
# Total t-subsets: C(19,5) = 11625
# Total k-subsets: C(19,8) = 75582

def _greedy_cover(v, k, t, rng):
    """Randomized greedy set cover: pick uncovered t-subset, build block greedily."""
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered_set = set(all_tsubs)
    blocks = []
    
    # Precompute for faster gain calculation? 
    # For v=19, k=8, t=5, calculating gain for each x in range(v) 
    # involves C(7,4)=35 checks per x, 19 x's -> 665 checks per step.
    # This is manageable in Python if optimized slightly.
    
    while uncovered_set:
        # Pick a random uncovered t-subset to diversify
        if rng is not None and rng.random() < 0.5:
            target = all_tsubs[rng.randrange(len(all_tsubs))]
            # Ensure it's actually uncovered? 
            # If we pick from all_tsubs, it might be covered. 
            # Better to pick from a list of uncovered if maintained, 
            # but maintaining a list is expensive. 
            # Let's stick to picking min or random from the set.
            # Actually, picking min(uncovered_set) is O(N) on a set of size ~10k.
            # Let's use a random sample from the set to avoid O(N) min.
            target = next(iter(uncovered_set)) # Arbitrary, but fast
            if rng.random() < 0.3:
                 # Try to pick a "random" one by sampling
                 sample_size = min(10, len(uncovered_set))
                 samples = rng.sample(list(uncovered_set), sample_size)
                 target = samples[0]
        else:
            target = min(uncovered_set)
            
        block = list(target)
        block_set = set(block)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: how many uncovered t-subsets would be covered
                gain = 0
                # We need to check subsets of block + x of size t
                # block has size b-1. We need to choose t-1 from block, add x.
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # Pick a random unused point
                unused = [x for x in range(v) if x not in block_set]
                if unused:
                    best_x = unused[rng.randrange(len(unused))] if rng else unused[0]
                else:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            uncovered_set.discard(ts)
    
    return blocks


def _local_search_numpy(v, k, t, blocks, rng, time_limit=110.0):
    """
    Local search using NumPy for coverage counts.
    This is significantly faster than dict-based coverage for v=19, t=5.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    # Map t-subsets to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    def build_coverage(blocks):
        """Return NumPy array of coverage counts."""
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[tsub_to_idx[ts]] += 1
        return cov
    
    cov = build_coverage(blocks)
    
    # Precompute t-subsets for each possible block? 
    # C(19,8) is 75582. Too many to precompute all.
    # But we only care about blocks in the current solution + potential new ones.
    
    while time.time() - start_time < time_limit:
        removed_any = False
        # Shuffle blocks to try different orders
        indices = list(range(len(blocks)))
        if rng:
            rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            block_i = blocks[i]
            
            # Check if removing this block leaves all t-subsets covered
            # i.e., min(cov[ts] for ts in block_i) > 1
            min_cov = 1000
            for ts in itertools.combinations(block_i, t):
                c = cov[tsub_to_idx[ts]]
                if c < min_cov:
                    min_cov = c
                if min_cov == 1:
                    break
            
            if min_cov > 1:
                # Safe to remove
                for ts in itertools.combinations(block_i, t):
                    cov[tsub_to_idx[ts]] -= 1
                blocks.pop(i)
                removed_any = True
                # Restart the scan from beginning to catch other redundancies
                break
        
        if not removed_any:
            # Try a more aggressive move: remove a block and re-cover gaps
            # This is where we can potentially improve over greedy.
            # We pick a random block, remove it, see what gaps appear.
            # Then try to cover those gaps with a new block.
            # If the new block covers ALL gaps, we swapped 1 block for 1 block (no gain in count, but maybe better structure for future removals?).
            # Actually, to reduce count, we need to remove a block and cover its gaps with EXISTING blocks? No, that's impossible if it's not redundant.
            # Or remove 1 block and add 0? No.
            # The standard move is: Remove block A. If gaps exist, try to cover them with a NEW block B.
            # If B covers all gaps of A, we have same number of blocks.
            # However, if we then find that B is redundant, we can remove it.
            # So the sequence is: Remove A (create gaps) -> Add B (cover gaps) -> Check if B is redundant -> Remove B.
            # Net result: Remove A, Add B, Remove B => Remove A. 
            # This effectively allows us to remove A even if it wasn't initially redundant, provided we can temporarily cover its gaps.
            
            if len(blocks) > 0:
                i = rng.randrange(len(blocks)) if rng else 0
                block_i = blocks[i]
                
                # Simulate removal
                gap_indices = []
                for ts in itertools.combinations(block_i, t):
                    idx = tsub_to_idx[ts]
                    if cov[idx] <= 1:
                        gap_indices.append(idx)
                
                if gap_indices:
                    # We have gaps. Try to build a block that covers them.
                    # Greedily build a block to cover as many gap t-subsets as possible.
                    gap_set = set(gap_indices)
                    new_block = []
                    new_block_set = set()
                    
                    # Start with a random gap t-subset
                    # We need to map index back to tuple to get elements
                    # It's faster to keep gap_set as tuples?
                    # Let's convert gap_indices to tuples for the builder
                    gap_tuples = [all_tsubs[i] for i in gap_indices]
                    gap_tuple_set = set(gap_tuples)
                    
                    target = gap_tuple_set.pop() if gap_tuple_set else None
                    if target:
                        new_block = list(target)
                        new_block_set = set(new_block)
                    
                    while len(new_block) < k and gap_tuple_set:
                        best_x = -1
                        best_gain = -1
                        for x in range(v):
                            if x in new_block_set:
                                continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t - 1):
                                new_ts = tuple(sorted(rest + (x,)))
                                if new_ts in gap_tuple_set:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x == -1 or best_gain == 0:
                            break
                        new_block.append(best_x)
                        new_block_set.add(best_x)
                        # Update gap_tuple_set
                        for rest in itertools.combinations(new_block, t):
                            gap_tuple_set.discard(tuple(sorted(rest)))
                    
                    # Check if this new block covers all gaps
                    if not gap_tuple_set:
                        new_block.sort()
                        # Accept: remove old block, add new one
                        # Update coverage
                        for ts in itertools.combinations(block_i, t):
                            cov[tsub_to_idx[ts]] -= 1
                        
                        # Remove old block
                        blocks.pop(i)
                        
                        # Add new block
                        for ts in itertools.combinations(new_block, t):
                            cov[tsub_to_idx[ts]] += 1
                        blocks.append(new_block)
                        
                        removed_any = True # Mark as changed to restart scan
                        break
        
        if not removed_any:
            # No improvement found in this pass
            # To avoid infinite loops, we can break or do a random perturbation
            # Let's break to save time for other restarts or just finish
            break
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Multiple randomized greedy restarts
    best_blocks = None
    best_count = float('inf')
    
    # Determine number of restarts based on time
    # Greedy is ~5-10s per run for v=19?
    # Local search is the expensive part.
    
    num_restarts = 3 # Conservative to leave time for local search
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        rng = random.Random(r * 7919 + 13)
        blocks = _greedy_cover(v, k, t, rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on the best solution
    if best_blocks is not None:
        if time.time() < deadline - 5:
            rng = random.Random(42)
            remaining = min(100.0, deadline - time.time() - 3.0)
            best_blocks = _local_search_numpy(v, k, t, best_blocks, rng, time_limit=remaining)
    
    # Final validation and cleanup
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover(v, k, t, None)
    
    # Ensure blocks are distinct and valid
    seen = set()
    final_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
    
    return final_blocks
# EVOLVE-BLOCK-END
