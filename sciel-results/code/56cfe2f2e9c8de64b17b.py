"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """
    Optimized Greedy Covering.
    Uses vectorized gain calculation to speed up block construction.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each point x, which t-subsets contain x?
    # Actually, we need: for a current block B, gain(x) = # of uncovered tsubs in B + {x}
    # Let's precompute a list of tsub indices for each point.
    # point_to_tsubs[x] = list of indices of tsubs containing x
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_to_tsubs[x].append(i)

    uncovered_mask = np.ones(n_tsubs, dtype=bool)
    n_uncovered = n_tsubs
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset
        uncovered_indices = np.where(uncovered_mask)[0]
        start_idx = rng.choice(uncovered_indices)
        start_tsub = all_tsubs[start_idx]
        block = list(start_tsub)
        block_set = set(block)
        
        # Precompute t-1 subsets of the current block
        # As block grows, we update this.
        # For efficiency, we can just compute combos on the fly if k is small, 
        # but for v=24, k=7, t=3, it's manageable.
        
        while len(block) < k:
            # Calculate gain for each candidate point x
            # gain(x) = sum(1 for ts in C(block, t-1) if (ts + x) is uncovered)
            
            # To speed this up, we can maintain an array `gain` of size v.
            # However, recalculating from scratch for each step is O(v * C(|B|, t-1)).
            # With v=24, |B|<=6, t=3, C(6,2)=15. 24*15 = 360 ops. Very fast.
            
            block_combos = list(itertools.combinations(block, t - 1))
            gains = np.zeros(v, dtype=np.int32)
            
            for rest in block_combos:
                for x in range(v):
                    if x in block_set:
                        continue
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and uncovered_mask[idx]:
                        gains[x] += 1
            
            # Find best point
            # Mask out points already in block
            for x in block_set:
                gains[x] = -1
            
            best_gain = np.max(gains)
            if best_gain <= 0:
                # Pick random
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            else:
                best_points = np.where(gains == best_gain)[0]
                best_point = rng.choice(best_points)
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered mask
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None and uncovered_mask[idx]:
                uncovered_mask[idx] = False
                n_uncovered -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """
    Fast removal of redundant blocks.
    A block is redundant if all its t-subsets are covered by other blocks.
    """
    if not blocks:
        return blocks
    
    # Map tsub -> count of blocks covering it
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_tsubs = []
        for i, block in enumerate(blocks):
            # Check if any tsub in this block is uniquely covered by it
            is_redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                changed = True
                for ts in block_tsubs[i]:
                    cover_count[ts] -= 1
            else:
                new_blocks.append(block)
                new_block_tsubs.append(block_tsubs[i])
        
        blocks = new_blocks
        block_tsubs = new_block_tsubs
        
    return blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    # Optimization: if len(covered) == C(v,t), it's a covering
    # But we need to be sure it's a subset of valid tsubs.
    # Since blocks are valid, all generated tsubs are valid.
    return len(covered) == len(list(itertools.combinations(range(v), t)))


def _local_search(blocks, v, k, t, rng, deadline):
    """
    Optimized Local Search.
    1. Remove redundant blocks.
    2. Try to replace blocks with ones that cover more 'new' t-subsets (greedy swap).
    3. Remove redundant blocks again.
    """
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Precompute tsub indices for fast lookup
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # cover_count[i] = number of blocks covering tsub i
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    
    def update_cover_count(block, sign):
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                cover_count[idx] += sign
    
    for block in current_blocks:
        update_cover_count(block, 1)
    
    # 1. Initial cleanup
    current_blocks = _remove_redundant_blocks(current_blocks, v, t)
    # Rebuild cover_count for the cleaned blocks
    cover_count[:] = 0
    for block in current_blocks:
        update_cover_count(block, 1)
    
    if len(current_blocks) < best_count:
        best_count = len(current_blocks)
        best_blocks = [list(b) for b in current_blocks]
    
    # 2. Swap heuristic
    # For each block, see if we can find a replacement that covers at least the same
    # unique tsubs but potentially more total tsubs (which might allow removing others later).
    # Actually, a simpler effective heuristic:
    # Try to remove a block. If it breaks coverage, try to "fix" the broken tsubs by
    # modifying other blocks or adding a new block? No, we want to minimize blocks.
    
    # Let's try a "2-opt" like move:
    # Remove block i.
    # If coverage is broken, identify the set of broken tsubs.
    # Try to find a single block (from a pool of candidates or constructed) that covers
    # all broken tsubs AND allows us to keep the same number of blocks or reduce it?
    # This is complex.
    
    # Alternative: Simulated Annealing style swaps.
    # Pick a block B_i. Generate a candidate B_new.
    # Replace B_i with B_new.
    # If the new solution is valid (it is, if B_new covers all tsubs that B_i uniquely covered? No, that's not enough).
    # Validity check is expensive.
    
    # Let's stick to a robust "Remove and Re-add" strategy.
    # 1. Remove a block.
    # 2. If invalid, identify minimal set of tsubs to cover.
    # 3. Try to cover them with a single new block (greedy construction constrained to cover these).
    # 4. If successful, we have same # blocks.
    # 5. Then try to remove another block.
    
    # This is effectively what the current slow version tried to do.
    # Let's optimize the "Re-add" part.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order for randomness
        indices = list(range(len(current_blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try to remove block i
            old_block = current_blocks[i]
            update_cover_count(old_block, -1)
            
            # Check if any tsub has count 0
            # We can track min count or just check the tsubs of old_block?
            # No, other blocks might have been covering them.
            # We need to check if ANY tsub has count 0.
            # Efficient way: keep a count of how many tsubs have count 0?
            # Or just check if min(cover_count) > 0.
            
            # Since n_tsubs is ~2024, min() is fast.
            if np.min(cover_count) > 0:
                # Removal was successful!
                current_blocks.pop(i)
                improved = True
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
                # Restart loop as indices changed
                break
            else:
                # Restoration needed
                update_cover_count(old_block, 1)
                
                # Try to find a replacement block that covers the "critical" tsubs
                # Critical tsubs are those with count 1 that are covered by old_block?
                # Actually, if we remove old_block, the tsubs with count 1 that were covered by it become 0.
                # Let's find them.
                critical_tsubs = []
                for ts in itertools.combinations(old_block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and cover_count[idx] == 1:
                        critical_tsubs.append(ts)
                
                if not critical_tsubs:
                    continue # Should not happen if min(cover_count)==0, but safety
                
                # Try to construct a block that covers all critical_tsubs
                # This is a set packing / covering subproblem.
                # Greedy: Start with one critical tsub, add points to cover others.
                
                best_candidate = None
                # Try starting with each critical tsub (or a sample if many)
                # Since critical_tsubs are all in old_block, they are compatible.
                # We can just try to complete old_block to a better block?
                # Or generate a new block from scratch that covers them.
                
                # Simple heuristic:
                # 1. Start with a random critical tsub.
                # 2. Greedily add points that cover most other critical tsubs.
                # 3. Then greedily add points to cover other uncovered tsubs (to maximize utility).
                
                for _attempt in range(5):
                    candidate = list(rng.choice(critical_tsubs))
                    cand_set = set(candidate)
                    
                    # Phase 1: Cover all critical tsubs
                    while True:
                        # Check if all critical tsubs are covered
                        all_covered = True
                        for cts in critical_tsubs:
                            if not set(cts).issubset(cand_set):
                                all_covered = False
                                break
                        
                        if all_covered:
                            break
                        
                        if len(candidate) >= k:
                            break # Failed to cover all criticals in k points?
                            # Actually, if they are in old_block (size k), they are coverable.
                            # But my greedy might fail.
                        
                        # Find point that covers most uncovered critical tsubs
                        best_gain = -1
                        best_point = -1
                        for x in range(v):
                            if x in cand_set:
                                continue
                            gain = 0
                            for cts in critical_tsubs:
                                if x in cts and not set(cts).issubset(cand_set):
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = x
                            
                        if best_point == -1:
                            break # Stuck
                            
                        candidate.append(best_point)
                        cand_set.add(best_point)
                    
                    # Check if we covered all criticals
                    if len(candidate) < k:
                        # Phase 2: Maximize total coverage
                        while len(candidate) < k:
                            # Standard greedy gain
                            block_combos = list(itertools.combinations(candidate, t - 1))
                            gains = np.zeros(v, dtype=np.int32)
                            for rest in block_combos:
                                for x in range(v):
                                    if x in cand_set:
                                        continue
                                    ts = tuple(sorted(rest + (x,)))
                                    idx = tsub_to_idx.get(ts)
                                    if idx is not None and cover_count[idx] == 0:
                                        gains[x] += 1
                            
                            for x in cand_set:
                                gains[x] = -1
                            
                            best_gain = np.max(gains)
                            if best_gain <= 0:
                                candidates = [x for x in range(v) if x not in cand_set]
                                best_point = rng.choice(candidates)
                            else:
                                best_points = np.where(gains == best_gain)[0]
                                best_point = rng.choice(best_points)
                            
                            candidate.append(best_point)
                            cand_set.add(best_point)
                    
                    candidate.sort()
                    
                    # Evaluate candidate
                    # Does it cover all critical tsubs?
                    cand_tsubs = set(itertools.combinations(candidate, t))
                    if all(ts in cand_tsubs for ts in critical_tsubs):
                        # Calculate "score": how many tsubs does it cover that are currently uncovered?
                        # And does it allow removing other blocks?
                        # For now, just accept if it covers criticals.
                        # To be smarter, check if it covers MORE tsubs than old_block?
                        old_tsubs = set(itertools.combinations(old_block, t))
                        if len(cand_tsubs) >= len(old_tsubs):
                            best_candidate = candidate
                            break
                        elif best_candidate is None:
                            best_candidate = candidate
                            break # Take first valid one to save time?
                            # Better: collect a few and pick best?
                            # Let's just take the first valid one for speed in this inner loop.
                    
                if best_candidate is not None:
                    # Replace old_block with best_candidate
                    # Note: cover_count still has old_block subtracted.
                    update_cover_count(best_candidate, 1)
                    current_blocks[i] = best_candidate
                    improved = True
                    # Don't break, continue to next block?
                    # Or break to restart?
                    # Breaking is safer for index consistency.
                    break
    
    # Final cleanup
    current_blocks = _remove_redundant_blocks(current_blocks, v, t)
    if len(current_blocks) < best_count:
        best_blocks = [list(b) for b in current_blocks]
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with varied seeds
    # Since greedy is now faster, we can afford more restarts or deeper local search.
    max_restarts = 100 
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Use different seed for each restart
        rng_restart = random.Random(42 + restart * 13)
        
        blocks = _greedy_cover(v, k, t, rng_restart, deadline)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search refinement
        # Allocate remaining time to local search
        local_deadline = min(deadline, time.time() + 10) # Cap local search time per restart initially?
        # Actually, let's just pass the global deadline.
        blocks = _local_search(blocks, v, k, t, rng_restart, deadline)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        # This should rarely be needed
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
