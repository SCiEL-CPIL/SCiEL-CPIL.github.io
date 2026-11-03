import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized local search heuristic.
    
    Strategy:
    1. Generate an initial covering using a fast greedy algorithm.
    2. Iteratively improve the covering by:
       a. Removing redundant blocks (blocks that cover no unique t-subsets).
       b. Replacing low-utility blocks with better candidates that maximize new coverage.
    
    Parameters:
    v: number of points
    k: block size
    t: subset size to cover
    
    Returns:
    A list of blocks (lists of integers) forming a covering.
    """
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index for fast array access
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def greedy_cover():
        """Fast greedy to get an initial valid covering."""
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to anchor the block
            anchor_idx = random.choice(list(uncovered))
            anchor = all_t_subsets[anchor_idx]
            
            # Grow the block greedily to maximize coverage of remaining uncovered sets
            current_block = set(anchor)
            
            while len(current_block) < k:
                best_point = None
                best_gain = -1
                best_tie_breaker = -1
                
                for p in all_points:
                    if p in current_block:
                        continue
                    
                    # Calculate gain: how many new t-subsets would be covered?
                    # A t-subset T is covered if T is a subset of current_block U {p}
                    # and T was not a subset of current_block.
                    # This is equivalent to: T contains p, and T \ {p} is a subset of current_block.
                    # And T must be currently uncovered.
                    
                    gain = 0
                    # We need to check t-subsets of (current_block U {p}) that contain p.
                    # These are formed by choosing t-1 elements from current_block.
                    for rest in itertools.combinations(tuple(current_block), t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        idx = t_subset_to_idx[ts]
                        if idx in uncovered:
                            gain += 1
                    
                    # Tie-breaker: prefer points that appear in more uncovered sets generally? 
                    # Or just random. Let's add a small random factor to diversify.
                    if gain > best_gain or (gain == best_gain and random.random() > 0.5):
                        best_gain = gain
                        best_point = p
                
                if best_point is None:
                    # Fallback: pick any point not in block
                    remaining = [p for p in all_points if p not in current_block]
                    if remaining:
                        best_point = random.choice(remaining)
                    else:
                        break
                
                if best_point is not None:
                    current_block.add(best_point)
                else:
                    break
            
            if len(current_block) == k:
                block_tuple = tuple(sorted(current_block))
                blocks.append(block_tuple)
                
                # Update uncovered set
                for ts in itertools.combinations(block_tuple, t):
                    idx = t_subset_to_idx[ts]
                    uncovered.discard(idx)
        
        return blocks

    def calculate_block_coverage(blocks):
        """
        Returns:
        - coverage_count: array of length num_t_subsets, count of blocks covering each t-subset
        - block_uniqueness: list of length len(blocks), number of t-subsets uniquely covered by that block
        """
        coverage_count = [0] * num_t_subsets
        block_uniqueness = [0] * len(blocks)
        
        # First pass: total coverage
        for b_idx, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                coverage_count[idx] += 1
        
        # Second pass: unique coverage
        for b_idx, block in enumerate(blocks):
            unique_count = 0
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                if coverage_count[idx] == 1:
                    unique_count += 1
            block_uniqueness[b_idx] = unique_count
            
        return coverage_count, block_uniqueness

    def get_uncovered_indices(coverage_count):
        """Return set of indices of t-subsets with coverage 0."""
        return {i for i, c in enumerate(coverage_count) if c == 0}

    def try_remove_redundant(blocks, coverage_count, block_uniqueness):
        """
        Attempts to remove blocks that have 0 unique coverage.
        Returns updated blocks, coverage_count, block_uniqueness if a removal was successful.
        """
        # Find blocks with 0 unique coverage
        candidates_to_remove = [i for i, u in enumerate(block_uniqueness) if u == 0]
        
        if not candidates_to_remove:
            return blocks, coverage_count, block_uniqueness
        
        # Try removing one at a time, starting with the first one found
        # In a real local search, we might try all, but for speed, we pick one randomly
        # Actually, if a block has 0 unique coverage, removing it is always safe 
        # IF we recalculate coverage. But wait, if I remove block A, 
        # does block B become redundant? Yes, potentially.
        # So we should loop until no more can be removed.
        
        changed = True
        while changed:
            changed = False
            # Recalculate uniqueness is expensive if done every time.
            # But if we only remove blocks with 0 unique coverage, we can just check 
            # if any block has 0 unique coverage in the CURRENT state.
            
            # To be efficient, we just pick one candidate from the previous list 
            # or re-scan. Re-scanning is O(N_blocks * C(k,t)).
            # Let's just pick one random candidate from the current list of 0-unique blocks.
            
            # We need to update coverage_count and block_uniqueness after removal.
            # For simplicity in this heuristic, we'll just try to remove one block
            # that has 0 unique coverage in the current snapshot.
            
            # Identify 0-unique blocks in current state
            zero_unique = [i for i, u in enumerate(block_uniqueness) if u == 0]
            
            if zero_unique:
                idx_to_remove = random.choice(zero_unique)
                block_to_remove = blocks[idx_to_remove]
                
                # Update coverage_count
                for ts in itertools.combinations(block_to_remove, t):
                    t_idx = t_subset_to_idx[ts]
                    coverage_count[t_idx] -= 1
                    
                # Check if any t-subset became uncovered (should not happen if logic is correct)
                # If coverage_count[t_idx] becomes 0, we have a problem. 
                # But by definition of "0 unique coverage", every t-subset in this block 
                # is covered by at least one other block. So coverage_count should stay >= 1.
                
                # Remove block
                blocks.pop(idx_to_remove)
                
                # Recalculate block_uniqueness for the remaining blocks
                # This is O(N_blocks * C(k,t)). 
                # Since we are doing local search, this cost is acceptable if N_blocks is small.
                _, block_uniqueness = calculate_block_coverage(blocks)
                changed = True
                # We return immediately after one removal to allow for other operations
                # or we can loop. Let's loop inside the function to clean up all redundancies.
                
        return blocks, coverage_count, block_uniqueness

    def try_replace_block(blocks, coverage_count, block_uniqueness):
        """
        Attempts to replace the block with the lowest unique coverage 
        with a new block that maximizes coverage of uncovered/under-covered sets.
        """
        if not blocks:
            return blocks, coverage_count, block_uniqueness
            
        # Find the block with the minimum unique coverage (excluding 0, as those are removed)
        # If there are 0-unique blocks, they should have been removed.
        # Let's find the block with the least "value".
        # Value can be defined as unique coverage.
        
        min_unique = min(block_uniqueness)
        candidates_to_replace = [i for i, u in enumerate(block_uniqueness) if u == min_unique]
        
        if not candidates_to_replace:
            return blocks, coverage_count, block_uniqueme
            
        # Pick a random candidate to replace
        idx_to_replace = random.choice(candidates_to_replace)
        old_block = blocks[idx_to_replace]
        
        # Generate a new candidate block
        # Strategy: Pick a random uncovered t-subset (or one with low coverage) 
        # and grow a block around it.
        
        # Find t-subsets with coverage == 1 (most vulnerable) or 0 (invalid, shouldn't exist)
        # Actually, we want to maximize the number of t-subsets that are currently 
        # covered by ONLY the old block (which we are removing) or by few blocks.
        
        # Simpler heuristic:
        # 1. Identify t-subsets covered by old_block.
        # 2. For each, check its coverage count.
        # 3. We want the new block to cover these "critical" subsets if possible,
        #    or cover new ones.
        
        # Let's just try to build a block that covers the most uncovered sets 
        # (sets with coverage 0, though there shouldn't be any) or sets with coverage 1.
        
        # To save time, we'll sample a few candidate blocks.
        best_new_block = None
        best_score = -1
        
        # Sample 10 candidate blocks
        for _ in range(10):
            # Pick a random t-subset to anchor
            # Prefer t-subsets with low coverage (1)
            low_cov_ts = [ts for i, ts in enumerate(all_t_subsets) if coverage_count[i] == 1]
            if not low_cov_ts:
                low_cov_ts = all_t_subsets
                
            anchor = random.choice(low_cov_ts)
            
            # Grow block greedily from anchor
            current_block = set(anchor)
            while len(current_block) < k:
                best_p = None
                best_g = -1
                for p in all_points:
                    if p in current_block:
                        continue
                    # Gain: number of t-subsets in (current_block U {p}) that are 
                    # currently covered by 1 block (and that block is likely the one we're removing? 
                    # No, we just want to maximize coverage of "weak" sets).
                    # Let's just count how many t-subsets containing p and subset of current_block U {p}
                    # have coverage_count == 1.
                    
                    g = 0
                    for rest in itertools.combinations(tuple(current_block), t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        idx = t_subset_to_idx[ts]
                        if coverage_count[idx] == 1:
                            g += 1
                            
                    if g > best_g or (g == best_g and random.random() > 0.5):
                        best_g = g
                        best_p = p
                        
                if best_p is None:
                    remaining = [p for p in all_points if p not in current_block]
                    if remaining:
                        best_p = random.choice(remaining)
                    else:
                        break
                if best_p is not None:
                    current_block.add(best_p)
                else:
                    break
                    
            if len(current_block) == k:
                candidate_block = tuple(sorted(current_block))
                
                # Score the candidate block
                # Score = number of t-subsets in candidate_block that are covered by 
                # the old_block (to ensure we don't lose coverage) 
                # PLUS number of t-subsets in candidate_block that are covered by 
                # few blocks (to strengthen coverage).
                
                # Actually, the simplest robust score:
                # If we replace old_block with candidate_block, 
                # any t-subset covered by old_block but NOT by candidate_block 
                # must be covered by at least one OTHER block.
                # If not, the replacement is invalid.
                
                valid = True
                # Check coverage for t-subsets in old_block
                for ts in itertools.combinations(old_block, t):
                    idx = t_subset_to_idx[ts]
                    # Coverage count currently includes old_block.
                    # If we remove old_block, coverage becomes coverage_count[idx] - 1.
                    # If coverage_count[idx] - 1 == 0, we need candidate_block to cover it.
                    if coverage_count[idx] == 1:
                        # Must be covered by candidate_block
                        if ts not in set(itertools.combinations(candidate_block, t)):
                            valid = False
                            break
                            
                if valid:
                    # Score: maximize number of t-subsets in candidate_block that are 
                    # currently covered by exactly 1 block (strengthening)
                    score = 0
                    for ts in itertools.combinations(candidate_block, t):
                        idx = t_subset_to_idx[ts]
                        if coverage_count[idx] == 1:
                            score += 1
                    # Add a small bonus for covering more total t-subsets?
                    # Or just random tie-breaker.
                    score += random.random() * 0.001
                    
                    if score > best_score:
                        best_score = score
                        best_new_block = candidate_block
                        
        if best_new_block is not None:
            # Perform replacement
            blocks[idx_to_replace] = best_new_block
            
            # Update coverage counts
            # Subtract old block
            for ts in itertools.combinations(old_block, t):
                idx = t_subset_to_idx[ts]
                coverage_count[idx] -= 1
            # Add new block
            for ts in itertools.combinations(best_new_block, t):
                idx = t_subset_to_idx[ts]
                coverage_count[idx] += 1
                
            # Recalculate block_uniqueness
            _, block_uniqueness = calculate_block_coverage(blocks)
            
            return blocks, coverage_count, block_uniqueness
            
        return blocks, coverage_count, block_uniqueness

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for overhead
    
    # Initial greedy
    blocks = greedy_cover()
    coverage_count, block_uniqueness = calculate_block_coverage(blocks)
    
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = [list(b) for b in blocks]
        
    while time.time() - start_time < time_limit:
        # Step 1: Remove redundancies
        blocks, coverage_count, block_uniqueness = try_remove_redundant(blocks, coverage_count, block_uniqueness)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Step 2: Try to replace a weak block
        blocks, coverage_count, block_uniqueness = try_replace_block(blocks, coverage_count, block_uniqueness)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Optional: Restart if stuck? 
        # For now, just keep improving.
        
    return best_blocks

if __name__ == "__main__":
    # Test with the specific parameters from the task
    v, k, t = 19, 8, 5
    blocks = construct(v, k, t)
    print(f"Found covering with {len(blocks)} blocks")
    
    # Validation
    all_t_subsets = set(itertools.combinations(range(v), t))
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
            
    if all_t_subsets == covered:
        print("Valid covering.")
    else:
        print("INVALID covering.")
        missing = all_t_subsets - covered
        print(f"Missing {len(missing)} subsets.")
