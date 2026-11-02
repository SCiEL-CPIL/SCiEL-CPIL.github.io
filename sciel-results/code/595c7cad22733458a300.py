"""Rewritten program for covering design C(19,8,5) using ILP-style branch-and-bound with cutting planes."""
# EVOLVE-BLOCK-START
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design using a greedy set-cover approach with
    randomized restarts and local improvement, inspired by ILP cutting-plane heuristics.
    
    The core idea:
    1. Generate candidate blocks (k-subsets) intelligently.
    2. Use a greedy set-cover to pick blocks that cover the most uncovered t-subsets.
    3. Use randomized restarts with different orderings to find better solutions.
    4. Apply local search to swap blocks for better coverage.
    """
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    total_t_subsets = len(all_t_subsets)
    
    # Precompute for each t-subset, which potential blocks could cover it
    # A block covers a t-subset if the t-subset is a subset of the block
    
    # We'll use a greedy approach with multiple strategies
    
    def greedy_cover(candidates, uncovered_set, max_blocks=None):
        """Greedy set cover: repeatedly pick the candidate that covers the most uncovered t-subsets."""
        # uncovered_set is a set of t-subsets (as tuples)
        if not uncovered_set:
            return []
        
        # Count how many uncovered t-subsets each candidate covers
        # To speed this up, we track for each t-subset which candidates cover it
        # But since candidates can be large, we'll compute on the fly
        
        blocks = []
        remaining_uncovered = set(uncovered_set)
        
        # Precompute: for each candidate, the list of t-subsets it covers
        # This is expensive, so we'll be selective about which candidates we consider
        
        # Strategy: only consider candidates that cover at least one uncovered t-subset
        # And prioritize those that cover many
        
        # For efficiency, we'll use a different approach:
        # Start from an uncovered t-subset, and build a block around it
        
        while remaining_uncovered:
            # Pick a random uncovered t-subset to anchor our new block
            anchor = random.choice(list(remaining_uncovered))
            
            # Build a block containing this anchor, choosing additional points
            # to maximize coverage of remaining uncovered t-subsets
            block_set = set(anchor)
            
            # Greedily add points to the block
            for _ in range(k - t):
                best_point = None
                best_gain = -1
                for p in all_points:
                    if p in block_set:
                        continue
                    # Calculate how many additional uncovered t-subsets would be covered
                    # if we add p to the block
                    new_block_set = block_set | {p}
                    # A t-subset is covered by the block if it's a subset of new_block_set
                    # We need to count uncovered t-subsets that are subsets of new_block_set
                    # but were not subsets of block_set (i.e., they require p)
                    
                    # More efficient: count uncovered t-subsets that contain p and are 
                    # subsets of new_block_set
                    gain = 0
                    for ts in remaining_uncovered:
                        if p in ts and set(ts).issubset(new_block_set):
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point is None:
                    break
                block_set.add(best_point)
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                remaining_uncovered.discard(tuple(ts))
            
            if max_blocks and len(blocks) >= max_blocks:
                break
        
        return blocks
    
    def greedy_cover_v2(uncovered_set):
        """Improved greedy: pick the block that covers the most uncovered t-subsets."""
        if not uncovered_set:
            return []
        
        remaining_uncovered = set(uncovered_set)
        blocks = []
        
        # For each possible block (k-subset), compute how many uncovered t-subsets it covers
        # This is too expensive for all C(v,k) blocks, so we use a heuristic
        
        # Heuristic: start with the uncovered t-subsets, and for each, consider blocks 
        # that contain it. Pick the block that covers the most uncovered t-subsets.
        
        while remaining_uncovered:
            # Sample some uncovered t-subsets
            sample_size = min(len(remaining_uncovered), 50)
            sample = random.sample(list(remaining_uncovered), sample_size)
            
            best_block = None
            best_count = -1
            
            for anchor in sample:
                # Generate candidate blocks containing this anchor
                # A block must contain all t points of the anchor
                remaining_points = [p for p in all_points if p not in anchor]
                need = k - t
                
                # Try a few random completions
                for _ in range(5):
                    if need <= 0:
                        block_set = set(anchor)
                    else:
                        extra = random.sample(remaining_points, min(need, len(remaining_points)))
                        block_set = set(anchor) | set(extra)
                    
                    if len(block_set) != k:
                        continue
                    
                    # Count how many uncovered t-subsets this block covers
                    count = 0
                    for ts in itertools.combinations(sorted(block_set), t):
                        if tuple(ts) in remaining_uncovered:
                            count += 1
                    
                    if count > best_count:
                        best_count = count
                        best_block = sorted(block_set)
            
            if best_block is None:
                # Fallback: just use the first anchor
                best_block = sorted(list(sample[0]) + random.sample(
                    [p for p in all_points if p not in sample[0]], k - t))
            
            blocks.append(best_block)
            
            # Remove covered t-subsets
            for ts in itertools.combinations(best_block, t):
                remaining_uncovered.discard(tuple(ts))
        
        return blocks
    
    def local_search(blocks, uncovered_set):
        """Try to improve the solution by swapping blocks."""
        if not blocks:
            return blocks, uncovered_set
        
        remaining_uncovered = set(uncovered_set)
        current_blocks = list(blocks)
        
        # Compute coverage for each block
        def compute_coverage(block):
            """Return the set of t-subsets covered by this block."""
            return set(tuple(ts) for ts in itertools.combinations(block, t))
        
        # Try to replace each block with a better one
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(current_blocks)):
                # Remove block i
                removed = current_blocks[i]
                temp_blocks = current_blocks[:i] + current_blocks[i+1:]
                
                # Find newly uncovered t-subsets
                newly_uncovered = set()
                for ts in itertools.combinations(removed, t):
                    ts_tuple = tuple(ts)
                    # Check if still covered by other blocks
                    still_covered = False
                    for b in temp_blocks:
                        if set(ts).issubset(set(b)):
                            still_covered = True
                            break
                    if not still_covered:
                        newly_uncovered.add(ts_tuple)
                
                if not newly_uncovered:
                    # Block i is redundant, remove it
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all newly_uncovered
                # and ideally covers additional uncovered t-subsets
                # For simplicity, try a few random blocks
                for _ in range(10):
                    # Build a block that covers as many of newly_uncovered as possible
                    # Start with one of the newly uncovered t-subsets
                    anchor = random.choice(list(newly_uncovered))
                    block_set = set(anchor)
                    remaining_points = [p for p in all_points if p not in block_set]
                    need = k - len(block_set)
                    
                    if need > 0:
                        # Greedily add points
                        for _ in range(need):
                            best_p = None
                            best_gain = -1
                            for p in remaining_points:
                                if p in block_set:
                                    continue
                                new_set = block_set | {p}
                                gain = sum(1 for ts in newly_uncovered 
                                          if p in ts and set(ts).issubset(new_set))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            if best_p is None:
                                break
                            block_set.add(best_p)
                            remaining_points.remove(best_p)
                    
                    if len(block_set) == k:
                        candidate = sorted(block_set)
                        # Check if this candidate covers all newly_uncovered
                        covers_all = all(set(ts).issubset(set(candidate)) for ts in newly_uncovered)
                        if covers_all:
                            current_blocks[i] = candidate
                            improved = True
                            break
                if improved:
                    break
        
        return current_blocks, remaining_uncovered
    
    # Main algorithm: multiple random restarts with greedy + local search
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 20
    random.seed(42)
    
    for restart in range(num_restarts):
        # Randomize the order of t-subsets to get different solutions
        shuffled_t_subsets = all_t_subsets[:]
        random.shuffle(shuffled_t_subsets)
        
        uncovered = set(shuffled_t_subsets)
        
        # Greedy cover
        blocks = greedy_cover_v2(uncovered)
        
        # Local search
        blocks, _ = local_search(blocks, uncovered)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_solution = blocks[:]
            
            # Check if we've found a perfect solution (no uncovered)
            if not uncovered:
                break
    
    if best_solution is None:
        # Fallback: simple greedy
        best_solution = greedy_cover_v2(set(all_t_subsets))
    
    # Ensure blocks are distinct
    seen = set()
    unique_blocks = []
    for b in best_solution:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
    
    return unique_blocks
# EVOLVE-BLOCK-END
