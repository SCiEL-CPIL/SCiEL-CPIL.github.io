"""Covering design C(19,8,5) — improved greedy with block-local search and time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

_DEADLINE_MARGIN = 15  # seconds before the 150s wall-clock kill


def _all_tsubsets(block, t):
    """Return list of t-subsets (as tuples) of the given block."""
    return [tuple(c) for c in itertools.combinations(block, t)]


def construct(v, k, t):
    start_time = time.time()

    # Precompute all t-subsets of the v points
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # uncovered[i] = True if t-subset i is not yet covered
    uncovered = [True] * n_tsubs
    num_uncovered = n_tsubs

    # For each point, list of t-subset indices that contain it
    # This lets us quickly compute how many new t-subsets a point adds to a block
    point_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(i)

    blocks = []
    seen_blocks = set()

    def elapsed():
        return time.time() - start_time

    def remaining_time():
        return 150.0 - elapsed() - _DEADLINE_MARGIN

    # Greedy: repeatedly pick a block that covers the most uncovered t-subsets
    # Use a smarter block construction: start from the most "constrained" uncovered t-subset
    while num_uncovered > 0:
        if remaining_time() <= 0:
            break

        # Pick the uncovered t-subset with the fewest candidate points not yet in any block
        # Actually, for covering design, pick the uncovered t-subset that appears in the fewest
        # possible blocks (i.e., the most constrained). Since all blocks are k-subsets,
        # every t-subset is contained in C(v-t, k-t) blocks. So they're all equally constrained.
        # Instead, pick the first uncovered t-subset (or a random one for diversity).
        
        # Find an uncovered t-subset
        # Use a random selection to diversify
        uncovered_indices = [i for i in range(n_tsubs) if uncovered[i]]
        if not uncovered_indices:
            break
        
        # Pick the uncovered t-subset that has the most "scarcity" - 
        # i.e., the one whose points are least frequently used together in uncovered sets
        # Simple heuristic: pick randomly from uncovered
        seed_ts_idx = random.choice(uncovered_indices)
        seed_ts = all_tsubs[seed_ts_idx]

        # Build a block starting from the seed t-subset
        block = list(seed_ts)
        
        # Greedily add points that maximize the number of newly covered t-subsets
        while len(block) < k:
            if remaining_time() <= 0:
                break
            
            block_set = set(block)
            # For each candidate point not in block, compute how many uncovered t-subsets it would help cover
            # A t-subset is covered by the block if all t of its elements are in the block
            # When we add point x, new t-subsets covered are those that:
            #   - are currently uncovered
            #   - contain x
            #   - have their other t-1 elements in the current block
            
            # Precompute: for each point x, which uncovered t-subsets contain x
            # We need t-subsets where x is in the t-subset and the other t-1 elements are in block
            
            best_x = None
            best_gain = -1
            
            # For efficiency, iterate over candidate points
            for x in all_points:
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets that contain x and whose remaining t-1 elements are in block
                # i.e., t-subsets of form (x, a, b, c, d) where {a,b,c,d} ⊆ block and |block ∩ (a,b,c,d)| = t-1
                # More precisely: t-subsets containing x where all other t-1 elements are in block
                
                gain = 0
                for ts_idx in point_to_tsubs[x]:
                    if not uncovered[ts_idx]:
                        continue
                    ts = all_tsubs[ts_idx]
                    # Check if all other elements of ts are in block
                    all_in_block = True
                    for elem in ts:
                        if elem == x:
                            continue
                        if elem not in block_set:
                            all_in_block = False
                            break
                    if all_in_block:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                # No point adds coverage; pick a random point to complete the block
                available = [x for x in all_points if x not in block_set]
                if available:
                    block.append(random.choice(available))
                break
            
            block.append(best_x)
        
        if len(block) < k:
            # Complete with random points
            block_set = set(block)
            for x in all_points:
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
        
        block.sort()
        block_tuple = tuple(block)
        
        if block_tuple in seen_blocks:
            continue
            
        seen_blocks.add(block_tuple)
        
        # Update uncovered
        covered_count = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(ts)]
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count += 1
        
        if covered_count > 0:
            blocks.append(block)
            num_uncovered -= covered_count
    
    # Verify and return
    if remaining_time() > 2:
        # Final check: ensure all t-subsets are covered
        covered = [False] * n_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[tuple(ts)]
                covered[idx] = True
        
        # If some are uncovered, add blocks for them
        for i in range(n_tsubs):
            if not covered[i]:
                if remaining_time() <= 2:
                    break
                ts = all_tsubs[i]
                # Create a block containing this t-subset
                block = list(ts)
                block_set = set(block)
                for x in all_points:
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                block.sort()
                block_tuple = tuple(block)
                
                if block_tuple in seen_blocks:
                    # This block is already there, but it didn't cover this t-subset?
                    # That implies a logic error in coverage update or this is a new t-subset
                    # that wasn't marked covered. But if the block is in seen_blocks,
                    # it was added to `blocks`, so its t-subsets should have been marked.
                    # Unless `covered` list is stale? No, we just built it.
                    # So if `covered[i]` is False, but block is in `blocks`, it means
                    # the block in `blocks` does NOT contain `ts`.
                    # But we constructed `block` from `ts`, so it MUST contain `ts`.
                    # So if `block_tuple` is in `seen_blocks`, it means we added this exact
                    # block before. If we added it before, `ts` (which is subset of it)
                    # should have been marked covered then.
                    # Contradiction: `covered[i]` is False.
                    # This can happen if the previous addition of this block failed to
                    # update `uncovered` correctly? No, `uncovered` is separate from `covered`.
                    # Ah, `covered` is built from `blocks`. If `block` is in `blocks`,
                    # and `ts` is in `block`, then `covered[i]` MUST be True.
                    # So if `covered[i]` is False, `block` is NOT in `blocks`.
                    # So `block_tuple` CANNOT be in `seen_blocks` if `seen_blocks` is
                    # consistent with `blocks`.
                    # However, if we skip adding here, we leave `ts` uncovered.
                    # To be safe, if it's a duplicate, we just skip, but this case
                    # theoretically shouldn't happen if state is consistent.
                    # If it does happen, we have a bug. For now, skip to avoid duplicate error.
                    continue
                
                seen_blocks.add(block_tuple)
                blocks.append(block)
    
    return blocks
# EVOLVE-BLOCK-END
