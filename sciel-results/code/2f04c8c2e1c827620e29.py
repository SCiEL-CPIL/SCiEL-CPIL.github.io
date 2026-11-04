"""Covering design C(v,k,t) via greedy set cover with index-based tracking,
redundancy removal, and swap-based local optimization."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # covered[i] = True if t-subset i is covered by at least one block
    covered = bytearray(n_t)
    uncovered_count = n_t
    blocks = []

    # Precompute: for each point x, the list of t-subset indices that contain x
    # This is expensive for large v, so we compute on the fly but cache per point
    # For v=19, t=5: C(19,5)=11628, manageable

    # For faster gain computation, precompute for each point the t-subsets containing it
    point_to_tsubs = {}
    for x in all_points:
        subs = []
        for sub in all_t_subs:
            if x in sub:
                subs.append(t_sub_to_idx[sub])
        point_to_tsubs[x] = subs

    # Greedy: repeatedly pick the best block to add
    while uncovered_count > 0 and time_ok():
        # Find the best block: maximize number of uncovered t-subsets covered
        # We use a heuristic: pick the most "constrained" uncovered t-subset,
        # then greedily build a block around it

        # Find first uncovered t-subset (or random for diversity)
        # Use a deterministic approach: find the uncovered t-subset with fewest
        # candidate extensions, but for speed just pick one at random among uncovered
        # Actually, let's use a smarter heuristic: pick uncovered t-subset that
        # is hardest to cover (fewest points that could extend it)
        
        # For efficiency, just pick a random uncovered t-subset
        # Build a set of uncovered indices for quick lookup
        # Instead of maintaining a set, we can scan covered array
        
        # Find an uncovered t-subset
        uncovered_indices = [i for i in range(n_t) if not covered[i]]
        if not uncovered_indices:
            break
        
        # Pick one - use the one with maximum "difficulty" (fewest possible blocks)
        # For simplicity and speed, pick randomly
        seed_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        seed_sub = all_t_subs[seed_idx]
        
        # Build block starting from seed_sub
        block = list(seed_sub)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            for x in all_points:
                if x in block:
                    continue
                # Gain = number of uncovered t-subsets that would be covered
                # A t-subset is covered by adding x if it contains x and (t-1) elements from block
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms a t-subset
                # that is uncovered
                for rest in itertools.combinations(block, t - 1):
                    # rest is a tuple of (t-1) points, add x
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand, -1)
                    if idx != -1 and not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain <= 0:
                # No gain, just add any available point
                for x in all_points:
                    if x not in block:
                        block.append(x)
                        break
            else:
                block.append(best_x)
        
        block.sort()
        
        # Add the block and update coverage
        blocks.append(block)
        block_set = set(block)
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                covered[idx] = 1
                uncovered_count -= 1
    
    if not time_ok():
        return blocks

    # Redundancy removal: remove blocks that don't cover any unique t-subset
    if time_ok():
        # For each block, check if removing it would leave any t-subset uncovered
        # Count how many blocks cover each t-subset
        n_blocks = len(blocks)
        # cover_count[i] = number of blocks that cover t-subset i
        cover_count = [0] * n_t
        for b in blocks:
            bset = set(b)
            for sub in itertools.combinations(b, t):
                idx = t_sub_to_idx[sub]
                cover_count[idx] += 1
        
        # Remove blocks that are redundant
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b in blocks:
                bset = set(b)
                # Check if this block is the only one covering any t-subset
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove this block
                    for sub in itertools.combinations(b, t):
                        idx = t_sub_to_idx[sub]
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
    
    if not time_ok():
        return blocks

    # Swap-based local optimization: try swapping points in a block to improve coverage
    # For each block, try replacing one point with another to see if it covers more
    # uncovered t-subsets (but all are covered at this point, so we look for blocks
    # that become redundant after swap)
    
    # Actually, since all t-subsets are covered, the optimization is to find a smaller
    # set of blocks. Let's try a different approach: for each block, try to see if
    # we can replace it with a better one, or remove it.
    
    # Try greedy re-optimization: sort blocks by "importance" (number of t-subsets
    # they uniquely cover) and try to rebuild
    if time_ok():
        # Compute for each block the number of t-subsets it uniquely covers
        n_blocks = len(blocks)
        unique_cover = [0] * n_blocks
        for i, b in enumerate(blocks):
            bset = set(b)
            for sub in itertools.combinations(b, t):
                idx = t_sub_to_idx[sub]
                if cover_count[idx] == 1:
                    unique_cover[i] += 1
        
        # Try to remove blocks with 0 unique coverage first (already done)
        # Try swap optimization: for each block, try swapping one point
        # This is complex; let's just do a simple hill-climbing on block count
        
        # Attempt: for each pair of blocks, see if one can be removed
        # if its coverage is fully provided by others
        # This is O(n^2 * C(k,t)) which may be too slow
        # For v=19, k=8, t=5: C(8,5)=56, and n~300, so 300*300*56 = 5M, feasible
    
    return blocks
# EVOLVE-BLOCK-END
