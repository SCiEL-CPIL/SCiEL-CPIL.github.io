"""Simulated annealing approach for covering design C(19,7,4)."""
import itertools
import random
import math
import time
import sys

# Use a fixed seed for reproducibility within a single run, but allow randomness
random.seed(42)


def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each possible block (k-subset), precompute which t-subsets it covers
    # But C(19,7) = 50388, and each covers C(7,4)=35 t-subsets. 
    # We can precompute a list of (block, covered_t_indices) for all blocks.
    
    # Actually, let's be smarter. We'll use a greedy initial solution, then improve with SA.
    
    # First, generate a good initial solution using a greedy heuristic
    def greedy_initial():
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        # Precompute for each k-subset, the list of t-subset indices it covers
        # This is expensive but C(19,7) = 50388, each with 35 entries, so ~1.7M entries
        # Let's compute on the fly but cache
        
        # Better: precompute all blocks and their covered t-subsets
        all_blocks = list(itertools.combinations(range(v), k))
        block_to_t_idx = []
        for b in all_blocks:
            t_idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_to_t_idx.append(t_idxs)
        
        # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
        while uncovered:
            best_block_idx = -1
            best_count = -1
            for bi, t_idxs in enumerate(block_to_t_idx):
                count = sum(1 for ti in t_idxs if ti in uncovered)
                if count > best_count:
                    best_count = count
                    best_block_idx = bi
                # Early break optimization: if we've already covered all, skip
                # But this is O(num_blocks * k) per iteration, which is slow
                # Let's just do it straightforwardly for now
            if best_block_idx == -1:
                break
            blocks.append(all_blocks[best_block_idx])
            for ti in block_to_t_idx[best_block_idx]:
                uncovered.discard(ti)
        
        return blocks
    
    # The greedy above is too slow for C(19,7) with full precomputation in the loop.
    # Let's use a different greedy: pick t-subset, then pick block that covers most uncovered.
    # But we need to be efficient.
    
    # Alternative: Use a randomized greedy with local search
    
    # Let's start with the simple greedy from the original, then improve
    def simple_greedy():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to break ties
            target = random.choice(list(uncovered))
            # Find the best block containing this target that covers most uncovered
            # Enumerate all k-subsets containing target
            rest_points = [x for x in range(v) if x not in target]
            best_block = None
            best_count = -1
            for rest in itertools.combinations(rest_points, k - t):
                block = tuple(sorted(target + rest))
                count = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
        return blocks
    
    # Start with simple greedy
    blocks = simple_greedy()
    
    # Now apply simulated annealing to improve
    # State: list of blocks
    # Moves:
    #   1. Remove a block if all its t-subsets are covered by other blocks
    #   2. Add a random block
    #   3. Replace a block with another that covers its unique t-subsets
    
    # Precompute coverage: for each t-subset, how many blocks cover it
    # And for each block, which t-subsets it covers
    
    def compute_coverage(blocks):
        """Return dict: t_subset -> count of blocks covering it, and list of block indices"""
        t_cover_count = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                t_cover_count[idx] += 1
        return t_cover_count
    
    def is_valid(blocks):
        """Check if every t-subset is covered by at least one block"""
        covered = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                covered[idx] += 1
        return all(c >= 1 for c in covered)
    
    def get_unique_t_subsets(blocks, block_idx):
        """Get t-subsets covered only by block_idx"""
        b = blocks[block_idx]
        # Count how many blocks cover each t-subset in b
        counts = {}
        for bi, other_b in enumerate(blocks):
            if bi == block_idx:
                continue
            for ts in itertools.combinations(other_b, t):
                if ts in counts:
                    counts[ts] += 1
                else:
                    counts[ts] = 1
        unique = []
        for ts in itertools.combinations(b, t):
            if ts not in counts:
                unique.append(ts)
        return unique
    
    # SA parameters
    start_time = time.time()
    timeout = 110  # Leave 10 seconds buffer
    
    current_blocks = blocks
    current_cost = len(current_blocks)
    
    # Precompute all possible blocks for random addition
    all_blocks_list = list(itertools.combinations(range(v), k))
    
    # SA loop
    temp = 10.0
    cooling = 0.995
    
    while time.time() - start_time < timeout:
        if not is_valid(current_blocks):
            # Restore validity: add blocks until valid
            uncovered = set()
            covered = [0] * num_t_subsets
            for b in current_blocks:
                for ts in itertools.combinations(b, t):
                    idx = t_subset_to_idx[ts]
                    covered[idx] += 1
            for i in range(num_t_subsets):
                if covered[i] == 0:
                    uncovered.add(all_t_subsets[i])
            while uncovered:
                target = random.choice(list(uncovered))
                rest_points = [x for x in range(v) if x not in target]
                best_block = None
                best_count = -1
                for rest in itertools.combinations(rest_points, k - t):
                    block = tuple(sorted(target + rest))
                    count = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
                    if count > best_count:
                        best_count = count
                        best_block = block
                if best_block is None:
                    break
                current_blocks.append(list(best_block))
                for ts in itertools.combinations(best_block, t):
                    uncovered.discard(ts)
            current_cost = len(current_blocks)
        
        # Choose a move
        move_type = random.random()
        
        new_blocks = list(current_blocks)
        
        if move_type < 0.4 and len(new_blocks) > 0:
            # Try to remove a block
            # Pick a random block and check if it's removable
            idx = random.randint(0, len(new_blocks) - 1)
            b = new_blocks[idx]
            # Check if all t-subsets of b are covered by other blocks
            other_blocks = new_blocks[:idx] + new_blocks[idx+1:]
            covered_by_others = set()
            for ob in other_blocks:
                for ts in itertools.combinations(ob, t):
                    covered_by_others.add(ts)
            removable = all(ts in covered_by_others for ts in itertools.combinations(b, t))
            if removable:
                new_blocks.pop(idx)
                new_cost = len(new_blocks)
                if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / temp):
                    current_blocks = new_blocks
                    current_cost = new_cost
        elif move_type < 0.7:
            # Try to replace a block
            if len(new_blocks) > 0:
                idx = random.randint(0, len(new_blocks) - 1)
                old_block = new_blocks[idx]
                # Find t-subsets uniquely covered by old_block
                other_blocks = new_blocks[:idx] + new_blocks[idx+1:]
                covered_by_others = set()
                for ob in other_blocks:
                    for ts in itertools.combinations(ob, t):
                        covered_by_others.add(ts)
                unique_ts = [ts for ts in itertools.combinations(old_block, t) if ts not in covered_by_others]
                if not unique_ts:
                    # Can just remove
                    new_blocks.pop(idx)
                    new_cost = len(new_blocks)
                    if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / temp):
                        current_blocks = new_blocks
                        current_cost = new_cost
                else:
                    # Find a replacement block that covers all unique_ts
                    # This is hard; skip for now or try random
                    pass
        else:
            # Add a random block
            new_block = list(random.choice(all_blocks_list))
            new_blocks.append(new_block)
            new_cost = len(new_blocks)
            # Only accept if it helps (reduces uncovered) or with low probability
            # For simplicity, only accept if we can then remove something
            # Actually, adding a block increases cost, so only do this if we expect to remove later
            # Skip this move type for now, or make it rare
            pass
        
        temp *= cooling
        if temp < 0.01:
            break
    
    # Final cleanup: remove any redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks)):
            b = current_blocks[i]
            other_blocks = current_blocks[:i] + current_blocks[i+1:]
            covered_by_others = set()
            for ob in other_blocks:
                for ts in itertools.combinations(ob, t):
                    covered_by_others.add(ts)
            if all(ts in covered_by_others for ts in itertools.combinations(b, t)):
                current_blocks.pop(i)
                changed = True
                break
    
    # Verify validity
    if not is_valid(current_blocks):
        # Fallback to greedy
        current_blocks = simple_greedy()
    
    return [list(b) for b in current_blocks]
