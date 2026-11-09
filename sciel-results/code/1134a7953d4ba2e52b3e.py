import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a two-phase approach:
    1. Fast Greedy Construction to get a valid initial covering.
    2. Iterative Local Search (Reduction) to remove redundant blocks.
    
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # --- Phase 1: Greedy Construction ---
    # We maintain a count of how many times each t-subset is covered.
    # Initially all are 0. We need to make all >= 1.
    cover_count = [0] * num_t_subsets
    blocks = []
    
    # To speed up the greedy phase, we can precompute for each point x, 
    # the list of t-subsets containing x? No, that's too big.
    # Instead, we track uncovered subsets.
    uncovered_set = set(range(num_t_subsets))
    
    # Heuristic for greedy:
    # Pick the t-subset with the highest "potential" or just the first one?
    # Standard greedy: pick the block that covers the most uncovered subsets.
    # To do this efficiently without checking all C(v,k) blocks:
    # We can construct a block around a specific uncovered t-subset.
    
    # Let's use a randomized greedy with a fixed seed for reproducibility in this run,
    # but maybe multiple restarts if time permits? 
    # For now, a single strong greedy + reduction is usually better than many weak ones.
    
    random.seed(42)
    
    while uncovered_set:
        # Pick a random uncovered t-subset to anchor the block
        # Picking the "most constrained" one (smallest candidate set) is better but expensive.
        # Let's just pick the minimum index for determinism or random for variety.
        # Random selection often leads to better diversity in the initial cover.
        target_idx = random.choice(list(uncovered_set))
        target_subset = all_t_subsets[target_idx]
        
        # Build a block of size k containing target_subset
        # Start with the points in target_subset
        block_points = set(target_subset)
        
        # Add k - t more points to maximize coverage of other uncovered subsets
        # We need to choose points from the remaining v - t points.
        remaining_points = [x for x in range(v) if x not in block_points]
        
        # Greedily add points
        for _ in range(k - t):
            best_point = -1
            best_gain = -1
            # For each candidate point, calculate how many NEW uncovered subsets it covers
            # A new subset is formed by (t-1) existing points in the block + the new point.
            # We need to check if that subset is in uncovered_set.
            
            # Optimization: Precompute (t-1)-combinations of current block_points?
            # Current block size is small (t to k-1).
            # Let's just iterate.
            
            # To speed up, we can sample if v is large, but v=19 is small.
            # Number of (t-1) subsets of block_points is C(len(block_points), t-1).
            # For v=19, k=8, t=5. Max block size 8. C(8,4)=70.
            # Candidates: ~11. 70*11 = 770 checks per point addition. Very fast.
            
            for x in remaining_points:
                gain = 0
                # Iterate over all (t-1) subsets of current block_points
                for rest in itertools.combinations(block_points, t - 1):
                    # Form the t-subset
                    new_sub = tuple(sorted(rest + (x,)))
                    idx = t_subset_to_idx.get(new_sub)
                    if idx is not None and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block_points.add(best_point)
                remaining_points.remove(best_point)
            else:
                # Fallback: add any random point
                if remaining_points:
                    p = random.choice(remaining_points)
                    block_points.add(p)
                    remaining_points.remove(p)
        
        # Finalize block
        block = tuple(sorted(block_points))
        blocks.append(block)
        
        # Update uncovered_set
        # Remove all t-subsets contained in this block
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            if idx in uncovered_set:
                uncovered_set.remove(idx)
        
        # Check time
        if time.time() > deadline - 30.0: # Leave 30s for reduction
             break

    # If we broke early and still have uncovered, we must finish quickly to be valid
    # But usually greedy is fast enough for v=19.
    while uncovered_set and time.time() < deadline - 5.0:
         # Emergency greedy
         target_idx = next(iter(uncovered_set))
         target_subset = all_t_subsets[target_idx]
         block_points = set(target_subset)
         remaining_points = [x for x in range(v) if x not in block_points]
         random.shuffle(remaining_points)
         for x in remaining_points:
             if len(block_points) >= k: break
             block_points.add(x)
         block = tuple(sorted(block_points))
         blocks.append(block)
         for ts in itertools.combinations(block, t):
             idx = t_subset_to_idx[ts]
             if idx in uncovered_set:
                 uncovered_set.remove(idx)
         
         if not uncovered_set:
             break
         
         # If stuck, force add
         if not uncovered_set: break

    if uncovered_set:
        # Cannot finish in time, return what we have (will be invalid, score 0)
        # But we should try to minimize this risk.
        pass

    # --- Phase 2: Reduction (Local Search) ---
    # Goal: Remove any block B if all t-subsets in B are covered by other blocks.
    # We maintain cover_count again.
    cover_count = [0] * num_t_subsets
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cover_count[t_subset_to_idx[ts]] += 1
    
    # Sort blocks by "criticality" or just iterate?
    # A block is removable if for all ts in B, cover_count[ts] > 1.
    
    # We will iterate through the list of blocks and try to remove them.
    # We can do multiple passes until no more blocks can be removed.
    
    # To make it faster, we can prioritize blocks that are "least useful".
    # A block is "least useful" if it contains many t-subsets that are covered > 1 times.
    
    def block_redundancy_score(b):
        # Count how many t-subsets in b are covered exactly once (critical)
        # If 0, it can be removed immediately.
        # If > 0, it's "critical".
        critical_count = 0
        for ts in itertools.combinations(b, t):
            if cover_count[t_subset_to_idx[ts]] == 1:
                critical_count += 1
        return critical_count

    # We will repeatedly scan the list and remove the first removable block found.
    # To be more effective, we can try to find the block with the "least impact" if we were to remove it?
    # No, removal is binary.
    
    # Let's sort blocks initially by redundancy score (ascending) to remove easy ones first?
    # Or just iterate.
    
    # Optimization: Instead of recalculating scores for all blocks every time,
    # we can just check blocks in a random order or fixed order.
    
    # Let's do a loop:
    # 1. Shuffle blocks? No, keep order.
    # 2. Check each block. If removable, remove it and update counts.
    # 3. Repeat until no changes or time runs out.
    
    # To speed up convergence, we can prioritize blocks that are "more redundant".
    # Let's compute initial scores.
    
    # However, calculating the score requires iterating all t-subsets of the block.
    # C(8,5) = 56 subsets per block.
    # If we have ~300 blocks, 300 * 56 = 16800 checks.
    # This is very fast.
    
    while time.time() < deadline - 2.0:
        changed = False
        # We can randomize the order of checking to avoid getting stuck in a local minimum
        # where a specific block order prevents removal of a set of blocks.
        # Actually, removing a block only increases cover counts for others (or keeps same),
        # making them *less* likely to be removable?
        # Wait. If I remove block A, the cover count for subsets in A drops by 1.
        # This might make subsets in A become critical (count=1) if they were 2 before.
        # This doesn't affect other blocks' removability directly unless they share subsets.
        # If block B shares a subset with A, and that subset had count 2, now it has count 1.
        # If B was removable (all its subsets had count > 1), it might *not* be removable anymore
        # if one of its subsets was shared with A and had count exactly 2.
        # So removal can make other blocks *harder* to remove.
        # Therefore, the order matters.
        
        # Strategy: Try to remove blocks that are "most redundant" first?
        # If I remove a block that has many subsets covered by many others, it's safe.
        # If I remove a block that has some subsets covered by only 1 other block,
        # those subsets become critical for that other block.
        
        # Let's just do a greedy removal:
        # Find a block that is currently removable. Remove it.
        # If none, break.
        
        # To find one quickly, we can scan.
        # We can also try "swaps" later if we have time, but removal is step 1.
        
        # Let's scan in a randomized order to escape potential ordering biases.
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(blocks):
                continue
            b = blocks[i]
            removable = True
            for ts in itertools.combinations(b, t):
                if cover_count[t_subset_to_idx[ts]] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove block i
                # Update cover counts
                for ts in itertools.combinations(b, t):
                    cover_count[t_subset_to_idx[ts]] -= 1
                
                blocks.pop(i)
                changed = True
                # Break the inner loop? 
                # If we continue, we might remove more.
                # But indices are stale.
                # Let's restart the scan or just continue with valid indices.
                # Restarting is safer and often fast enough.
                break
        
        if not changed:
            # No single block can be removed.
            # We can try a more complex move: Replace a block with a better one?
            # Or accept the local minimum.
            # Given the time limit, let's stop if no simple removals are found.
            # We could try a few random restarts of the reduction if time permits,
            # but usually, the first pass cleans up most redundancy.
            
            # Let's try one more pass with a different random order just in case?
            # No, if no block is removable in the current state, the state is a local minimum.
            # Different orders might find a different local minimum, but not necessarily a better one.
            # However, since removal is monotonic in terms of "difficulty" (counts go down),
            # the set of removable blocks shrinks.
            # Actually, if no block is removable, we are stuck.
            break

    return [list(b) for b in blocks]

# Keep the original function signature
# The prompt asks to keep function names and signatures.
# The seed program had `construct(v, k, t)`.
