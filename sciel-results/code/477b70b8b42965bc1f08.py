"""Stronger covering design algorithm for C(v, k, t)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

import numpy as np


def construct(v, k, t):
    """
    Greedy covering design with bottleneck heuristic and multiple strategies.
    
    Strategy:
    1. Bottleneck greedy: pick the t-subset that is "hardest" to cover, then greedily
       build a block maximizing coverage gain with a soft-min surrogate for bottleneck.
    2. Local search polish: try to remove redundant blocks.
    3. Multiple restarts with different random seeds if time permits.
    """
    start_time = time.time()
    time_limit = 115  # seconds, leave margin before 120s kill
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def time_elapsed():
        return time.time() - start_time
    
    def build_greedy_block_bottleneck(uncovered_set, block_seed, rng, beta=2.0):
        """
        Build a k-block starting from block_seed (a t-subset) using a soft-min surrogate.
        At each step, choose the point that maximizes a smooth combination of:
        - Raw gain (number of new uncovered t-subsets covered)
        - Soft-min of per-subset gains (encourages covering constrained subsets)
        """
        block = set(block_seed)
        remaining = k - t
        
        # Precompute uncovered t-subsets as a set for fast membership
        uncovered = set(uncovered_set)
        
        for _ in range(remaining):
            best_point = -1
            best_score = -float('inf')
            candidates = [x for x in range(v) if x not in block]
            
            # Shuffle for tie-breaking randomness
            rng.shuffle(candidates)
            
            for x in candidates:
                # Count newly covered t-subsets
                gain = 0
                for rest in itertools.combinations(sorted(block), t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                
                # Soft-min surrogate: use log-sum-exp style to create a smooth
                # approximation of the minimum constraint level
                # Here we approximate by weighting gain with a concave function
                # that rewards spreading coverage across many subsets
                if gain > 0:
                    # Smooth surrogate: gain * (1 - exp(-beta/gain)) approximates
                    # a soft-min that favors solutions with more balanced coverage
                    soft_min_score = gain * (1.0 - np.exp(-beta / gain))
                else:
                    soft_min_score = 0.0
                
                # Combine raw gain with soft-min surrogate
                score = 0.7 * gain + 0.3 * soft_min_score
                
                if score > best_score:
                    best_score = score
                    best_point = x
            
            block.add(best_point)
        
        return sorted(block)
    
    def greedy_cover_bottleneck(uncovered_set, rng, use_random_seed=False):
        """
        Greedy set cover with bottleneck heuristic: repeatedly pick the hardest
        uncovered t-subset to cover, build a block around it, and remove covered subsets.
        """
        blocks = []
        uncovered = set(uncovered_set)
        
        # Precompute frequency of each point in uncovered t-subsets for bottleneck estimation
        point_freq = defaultdict(int)
        for ts in uncovered:
            for p in ts:
                point_freq[p] += 1
        
        while uncovered:
            if time_elapsed() > time_limit:
                break
            
            if use_random_seed:
                # Pick randomly for diversity
                ts = rng.choice(list(uncovered))
            else:
                # Bottleneck heuristic: pick the t-subset with the highest total point frequency
                best_ts = None
                best_freq = -1
                # Sample a subset of uncovered for speed
                if len(uncovered) > 1000:
                    sample = rng.sample(list(uncovered), 1000)
                else:
                    sample = list(uncovered)
                for ts_candidate in sample:
                    freq = sum(point_freq[p] for p in ts_candidate)
                    if freq > best_freq:
                        best_freq = freq
                        best_ts = ts_candidate
                ts = best_ts
            
            # Build a block centered on this t-subset using bottleneck heuristic
            block = build_greedy_block_bottleneck(uncovered, ts, rng)
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for covered_ts in itertools.combinations(block, t):
                uncovered.discard(covered_ts)
            
            # Update point frequencies
            for p in block:
                point_freq[p] = max(0, point_freq[p] - 1)
        
        return blocks
    
    def greedy_cover_standard(uncovered_set, rng):
        """Standard greedy: pick min uncovered t-subset, build block maximizing raw gain."""
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered:
            if time_elapsed() > time_limit:
                break
            
            ts = min(uncovered)
            
            # Build block maximizing raw coverage
            block = set(ts)
            remaining = k - t
            
            for _ in range(remaining):
                best_point = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(sorted(block), t - 1):
                        ts_new = tuple(sorted(rest + (x,)))
                        if ts_new in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                block.add(best_point)
            
            blocks.append(sorted(block))
            
            for covered_ts in itertools.combinations(block, t):
                uncovered.discard(covered_ts)
        
        return blocks
    
    def try_remove_redundant(blocks):
        """Try to remove blocks that are redundant (all their t-subsets are covered by other blocks)."""
        if not blocks:
            return blocks
        
        # Build a coverage count: for each t-subset, how many blocks cover it
        coverage_count = defaultdict(int)
        block_t_subsets = []
        for block in blocks:
            ts_list = list(itertools.combinations(block, t))
            block_t_subsets.append(ts_list)
            for ts in ts_list:
                coverage_count[ts] += 1
        
        new_blocks = []
        for i, block in enumerate(blocks):
            if all(coverage_count[ts] > 1 for ts in block_t_subsets[i]):
                # This block is redundant
                for ts in block_t_subsets[i]:
                    coverage_count[ts] -= 1
            else:
                new_blocks.append(block)
        
        return new_blocks
    
    def local_search_swap(blocks, rng, max_iterations=50):
        """
        Local search: try to swap out one block for another that covers the same
        critical t-subsets but allows removing another block.
        """
        if len(blocks) < 2:
            return blocks
        
        # Build coverage count
        coverage_count = defaultdict(int)
        block_t_subsets = []
        for block in blocks:
            ts_list = list(itertools.combinations(block, t))
            block_t_subsets.append(ts_list)
            for ts in ts_list:
                coverage_count[ts] += 1
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time_elapsed() > time_limit - 10:
                break
            
            # Try to find a redundant block to remove
            for i in range(len(blocks)):
                if all(coverage_count[ts] > 1 for ts in block_t_subsets[i]):
                    # Remove this block
                    for ts in block_t_subsets[i]:
                        coverage_count[ts] -= 1
                    blocks.pop(i)
                    block_t_subsets.pop(i)
                    improved = True
                    break
        
        return blocks
    
    # Run multiple greedy strategies and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Bottleneck greedy (deterministic)
    rng1 = random.Random(42)
    blocks1 = greedy_cover_bottleneck(set(all_t_subsets), rng1, use_random_seed=False)
    if len(blocks1) < best_count:
        best_count = len(blocks1)
        best_blocks = blocks1
    
    # Strategy 2: Bottleneck greedy with random seed selection
    if time_elapsed() < time_limit - 30:
        rng2 = random.Random(123)
        blocks2 = greedy_cover_bottleneck(set(all_t_subsets), rng2, use_random_seed=True)
        if len(blocks2) < best_count:
            best_count = len(blocks2)
            best_blocks = blocks2
    
    # Strategy 3: Standard greedy
    if time_elapsed() < time_limit - 30:
        rng3 = random.Random(7)
        blocks3 = greedy_cover_standard(set(all_t_subsets), rng3)
        if len(blocks3) < best_count:
            best_count = len(blocks3)
            best_blocks = blocks3
    
    # Try local search polish on the best solution
    if best_blocks and time_elapsed() < time_limit - 20:
        polished = try_remove_redundant(best_blocks)
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = polished
        
        # Additional local search pass
        if time_elapsed() < time_limit - 15:
            rng_ls = random.Random(999)
            polished2 = local_search_swap(list(best_blocks), rng_ls, max_iterations=30)
            if len(polished2) < best_count:
                best_count = len(polished2)
                best_blocks = polished2
    
    # More restarts if time permits
    if best_blocks and time_elapsed() < time_limit - 40:
        for seed in [99, 2024, 555]:
            if time_elapsed() > time_limit - 30:
                break
            rng4 = random.Random(seed)
            blocks4 = greedy_cover_bottleneck(set(all_t_subsets), rng4, use_random_seed=(seed % 2 == 0))
            if len(blocks4) < best_count:
                best_count = len(blocks4)
                best_blocks = blocks4
            # Try polish
            if time_elapsed() < time_limit - 25:
                polished4 = try_remove_redundant(blocks4)
                if len(polished4) < best_count:
                    best_count = len(polished4)
                    best_blocks = polished4
    
    # Final check: ensure we have a valid cover
    if best_blocks is None:
        # Fallback: simple greedy
        rng5 = random.Random(0)
        best_blocks = greedy_cover_standard(set(all_t_subsets), rng5)
    
    # Verify coverage
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    # If not fully covered (due to time), extend with additional blocks
    if len(covered) < n_t_subsets:
        uncovered = set(all_t_subsets) - covered
        while uncovered:
            if time_elapsed() > time_limit:
                break
            ts = min(uncovered)
            block = build_greedy_block_bottleneck(uncovered, ts, random.Random())
            best_blocks.append(block)
            for covered_ts in itertools.combinations(block, t):
                uncovered.discard(covered_ts)
    
    return best_blocks
# EVOLVE-BLOCK-END
