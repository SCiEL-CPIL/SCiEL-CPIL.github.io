"""Simulated annealing approach for covering design C(24,7,3)."""
import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using simulated annealing.
    
    Strategy:
    1. Start with a greedy covering (or random blocks).
    2. Use simulated annealing to swap points within blocks to reduce the number of uncovered t-subsets.
    3. Periodically remove redundant blocks if possible.
    
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Count how many times each t-subset is covered
    def count_coverages(blocks):
        """Return an array where count[i] = number of blocks covering all_t_subsets[i]."""
        counts = [0] * n_t_subsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                counts[idx] += 1
        return counts
    
    def uncovered_count(counts):
        """Return the number of uncovered t-subsets."""
        return sum(1 for c in counts if c == 0)
    
    def get_uncovered_indices(counts):
        """Return list of indices of uncovered t-subsets."""
        return [i for i, c in enumerate(counts) if c == 0]
    
    # Step 1: Generate initial covering using greedy algorithm
    def greedy_covering():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            # Pick the first uncovered t-subset
            target = min(uncovered)
            block = list(target)
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks
    
    blocks = greedy_covering()
    counts = count_coverages(blocks)
    current_uncovered = uncovered_count(counts)
    
    # If greedy already covers everything, we can try to improve by removing redundant blocks
    # But first, let's try simulated annealing to reduce the number of blocks
    
    # Step 2: Simulated annealing to reduce block count
    # We'll work with a fixed number of blocks initially, then try to remove redundant ones
    
    # Let's start with the current number of blocks
    n_blocks = len(blocks)
    
    # Temperature schedule
    start_time = time.time()
    max_time = 115.0  # Leave 5 seconds buffer
    
    T = 10.0  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    
    best_blocks = blocks[:]
    best_uncovered = current_uncovered
    
    # Precompute for each block, which t-subsets it covers
    # This will be updated during moves
    
    # For efficiency, let's maintain the counts array and update it incrementally
    
    # Define a move: pick a random block, pick a random point in it, replace it with a random other point
    # Or: pick two blocks and swap points between them
    
    # Let's use the following moves:
    # 1. Replace a point in a random block with another point
    # 2. Swap a point between two random blocks
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        # Randomly choose a move type
        move_type = random.random()
        
        if move_type < 0.5 and n_blocks >= 1:
            # Move 1: Replace a point in a random block
            bi = random.randint(0, n_blocks - 1)
            block = blocks[bi]
            pi = random.randint(0, k - 1)
            old_point = block[pi]
            
            # Choose a new point not already in the block
            available = [x for x in range(v) if x not in block]
            if not available:
                continue
            new_point = random.choice(available)
            
            # Calculate the change in uncovered count
            # Remove old point, add new point
            # t-subsets affected: those containing old_point but not new_point, and vice versa
            
            # Count how many t-subsets contain old_point from this block
            old_covered = []
            for ts in itertools.combinations(block, t):
                if old_point in ts:
                    old_covered.append(t_subset_to_idx[ts])
            
            # After removing old_point and adding new_point
            new_block = block[:]
            new_block[pi] = new_point
            new_block.sort()
            
            new_covered = []
            for ts in itertools.combinations(new_block, t):
                if new_point in ts:
                    new_covered.append(t_subset_to_idx[ts])
            
            # The change in coverage:
            # t-subsets that were covered by old_point in this block but not by new_point
            # and t-subsets that are covered by new_point but not by old_point
            
            # For each t-subset, we need to check if its count goes from 0 to >0 or >0 to 0
            
            delta = 0
            # t-subsets that lose coverage (only if count was 1)
            for idx in old_covered:
                if new_point not in all_t_subsets[idx]:
                    # This t-subset contains old_point but not new_point
                    # It loses one coverage from this block
                    if counts[idx] == 1:
                        delta += 1  # Becomes uncovered
            
            # t-subsets that gain coverage (only if count was 0)
            for idx in new_covered:
                if old_point not in all_t_subsets[idx]:
                    # This t-subset contains new_point but not old_point
                    # It gains one coverage from this block
                    if counts[idx] == 0:
                        delta -= 1  # Becomes covered
            
            # Accept or reject
            if delta <= 0:
                # Improvement or neutral
                # Update counts
                for idx in old_covered:
                    if new_point not in all_t_subsets[idx]:
                        counts[idx] -= 1
                for idx in new_covered:
                    if old_point not in all_t_subsets[idx]:
                        counts[idx] += 1
                blocks[bi] = new_block
                current_uncovered += delta
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
            else:
                # Deterioration: accept with probability exp(-delta/T)
                if random.random() < (2.718281828 ** (-delta / T)):
                    # Update counts
                    for idx in old_covered:
                        if new_point not in all_t_subsets[idx]:
                            counts[idx] -= 1
                    for idx in new_covered:
                        if old_point not in all_t_subsets[idx]:
                            counts[idx] += 1
                    blocks[bi] = new_block
                    current_uncovered += delta
        
        elif move_type < 0.8 and n_blocks >= 2:
            # Move 2: Swap a point between two random blocks
            bi1 = random.randint(0, n_blocks - 1)
            bi2 = random.randint(0, n_blocks - 1)
            if bi1 == bi2:
                continue
            
            block1 = blocks[bi1]
            block2 = blocks[bi2]
            
            pi1 = random.randint(0, k - 1)
            pi2 = random.randint(0, k - 1)
            
            p1 = block1[pi1]
            p2 = block2[pi2]
            
            # Ensure p1 is not in block2 and p2 is not in block1
            if p1 in block2 or p2 in block1:
                continue
            
            new_block1 = block1[:]
            new_block1[pi1] = p2
            new_block1.sort()
            
            new_block2 = block2[:]
            new_block2[pi2] = p1
            new_block2.sort()
            
            # Calculate delta
            # For block1: remove p1, add p2
            # For block2: remove p2, add p1
            
            delta = 0
            
            # Block1 changes
            old1_covered = [t_subset_to_idx[ts] for ts in itertools.combinations(block1, t) if p1 in ts]
            new1_covered = [t_subset_to_idx[ts] for ts in itertools.combinations(new_block1, t) if p2 in ts]
            
            for idx in old1_covered:
                if p2 not in all_t_subsets[idx]:
                    if counts[idx] == 1:
                        delta += 1
            
            for idx in new1_covered:
                if p1 not in all_t_subsets[idx]:
                    if counts[idx] == 0:
                        delta -= 1
            
            # Block2 changes
            old2_covered = [t_subset_to_idx[ts] for ts in itertools.combinations(block2, t) if p2 in ts]
            new2_covered = [t_subset_to_idx[ts] for ts in itertools.combinations(new_block2, t) if p1 in ts]
            
            for idx in old2_covered:
                if p1 not in all_t_subsets[idx]:
                    if counts[idx] == 1:
                        delta += 1
            
            for idx in new2_covered:
                if p2 not in all_t_subsets[idx]:
                    if counts[idx] == 0:
                        delta -= 1
            
            # Accept or reject
            if delta <= 0:
                # Update counts
                for idx in old1_covered:
                    if p2 not in all_t_subsets[idx]:
                        counts[idx] -= 1
                for idx in new1_covered:
                    if p1 not in all_t_subsets[idx]:
                        counts[idx] += 1
                for idx in old2_covered:
                    if p1 not in all_t_subsets[idx]:
                        counts[idx] -= 1
                for idx in new2_covered:
                    if p2 not in all_t_subsets[idx]:
                        counts[idx] += 1
                blocks[bi1] = new_block1
                blocks[bi2] = new_block2
                current_uncovered += delta
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in blocks]
            else:
                if random.random() < (2.718281828 ** (-delta / T)):
                    for idx in old1_covered:
                        if p2 not in all_t_subsets[idx]:
                            counts[idx] -= 1
                    for idx in new1_covered:
                        if p1 not in all_t_subsets[idx]:
                            counts[idx] += 1
                    for idx in old2_covered:
                        if p1 not in all_t_subsets[idx]:
                            counts[idx] -= 1
                    for idx in new2_covered:
                        if p2 not in all_t_subsets[idx]:
                            counts[idx] += 1
                    blocks[bi1] = new_block1
                    blocks[bi2] = new_block2
                    current_uncovered += delta
        
        else:
            # Move 3: Try to remove a redundant block
            # A block is redundant if removing it doesn't uncover any t-subset
            # i.e., all t-subsets it covers are covered by at least one other block
            
            if n_blocks > 1:
                bi = random.randint(0, n_blocks - 1)
                block = blocks[bi]
                
                # Check if all t-subsets in this block are covered by other blocks
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this block
                    # Update counts
                    for ts in itertools.combinations(block, t):
                        idx = t_subset_to_idx[ts]
                        counts[idx] -= 1
                    
                    blocks.pop(bi)
                    n_blocks -= 1
                    current_uncovered = uncovered_count(counts)
                    
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_blocks = [b[:] for b in blocks]
        
        # Cool down
        T *= cooling_rate
    
    # Use the best solution found
    blocks = best_blocks
    
    # Final cleanup: remove any redundant blocks
    changed = True
    while changed:
        changed = False
        counts = count_coverages(blocks)
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break
    
    # Ensure all blocks are sorted
    for i in range(len(blocks)):
        blocks[i].sort()
    
    # Verify coverage
    counts = count_coverages(blocks)
    if uncovered_count(counts) > 0:
        # Fallback to greedy if SA didn't find a valid covering
        blocks = greedy_covering()
    
    return blocks
