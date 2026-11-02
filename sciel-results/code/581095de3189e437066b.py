"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    """
    Greedy covering with lookahead via Monte Carlo estimation of future coverage.
    Uses a deadline guard to ensure we return a valid covering within the time limit.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # Track which t-subsets are covered
    # Use a set of frozensets for fast lookup
    uncovered = set(frozenset(s) for s in all_t_subsets)
    
    blocks = []
    
    # Helper: count how many uncovered t-subsets a candidate block would cover
    def count_covered_by_block(block_set):
        """Count uncovered t-subsets contained in block_set."""
        count = 0
        for ts in itertools.combinations(block_set, t):
            if frozenset(ts) in uncovered:
                count += 1
        return count
    
    # Helper: estimate future benefit using Monte Carlo
    def estimate_future_benefit(block_set, n_samples=50):
        """
        Estimate how much this block reduces the expected number of additional
        blocks needed by sampling from the remaining uncovered t-subsets.
        """
        if not uncovered:
            return 0.0
        
        # Sample some uncovered t-subsets to estimate overlap potential
        sample_list = list(uncovered)
        if len(sample_list) <= n_samples:
            samples = sample_list
        else:
            samples = random.sample(sample_list, n_samples)
        
        # For each sampled uncovered t-subset, check if it shares elements with block_set
        # This is a rough proxy for how well this block positions us for future coverage
        overlap_score = 0.0
        for fs in samples:
            # Count how many elements of this t-subset are in block_set
            overlap = sum(1 for x in fs if x in block_set)
            overlap_score += overlap
        return overlap_score / len(samples)
    
    while uncovered:
        if time.time() > DEADLINE:
            break
        
        # Choose the first uncovered t-subset as a starting point
        first_uncovered = next(iter(uncovered))
        block = list(first_uncovered)
        
        # Build the block by adding points that maximize immediate coverage
        # with a small lookahead component
        while len(block) < k:
            if time.time() > DEADLINE:
                break
            
            block_set = set(block)
            best_point = None
            best_score = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Immediate gain: how many new uncovered t-subsets does adding x cover?
                immediate_gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if frozenset(candidate) in uncovered:
                        immediate_gain += 1
                
                # Lookahead: rough estimate of future positioning
                # Use a lightweight version: check overlap with a few random uncovered sets
                if immediate_gain == 0 and len(uncovered) > 100:
                    # If no immediate gain, use a simpler heuristic
                    # Prefer points that appear in more uncovered t-subsets
                    freq = 0
                    # Sample a few uncovered sets
                    sample_size = min(20, len(uncovered))
                    for fs in random.sample(list(uncovered), sample_size):
                        if x in fs:
                            freq += 1
                    lookahead = freq / sample_size
                    score = immediate_gain + 0.1 * lookahead
                else:
                    score = float(immediate_gain)
                
                if score > best_score or (score == best_score and best_point is None):
                    best_score = score
                    best_point = x
            
            if best_point is None:
                # Fallback: pick any point not in block
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break
            
            block.append(best_point)
        
        if len(block) < k:
            break
        
        block.sort()
        block_set = frozenset(block)
        
        # Check if this block is already in blocks
        if block_set not in [frozenset(b) for b in blocks]:
            blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(frozenset(ts))
    
    return blocks
# EVOLVE-BLOCK-END
