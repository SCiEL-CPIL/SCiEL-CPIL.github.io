"""Covering design C(19,8,5) - improved greedy with urgency and diversity."""
import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    """Construct a covering design: choose k-subsets of v points so every t-subset is covered."""
    deadline = time.time() + 115  # safety margin before 120s limit
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Precompute index for each t-subset
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_once(seed):
        """Run one greedy pass with given random seed."""
        rng = random.Random(seed)
        
        # Track uncovered t-subsets
        uncovered_set = set(range(num_t_subsets))
        uncovered_list = list(uncovered_set)  # maintain a list for sampling
        
        # Track how many times each point has been used
        point_freq = [0] * v
        
        blocks = []
        block_set = set()  # for duplicate detection
        
        while uncovered_list:
            if time.time() > deadline:
                break
            
            # Choose the most "urgent" uncovered t-subset
            # Sample a few candidates and pick the one with least-used points
            sample_size = min(30, len(uncovered_list))
            candidates = []
            for _ in range(sample_size):
                idx = rng.choice(uncovered_list)
                sub = all_t_subsets[idx]
                # Score: prefer subsets with less-used points
                score = sum(1.0 / (1 + point_freq[p]) for p in sub)
                candidates.append((score, idx))
            
            if not candidates:
                break
            
            # Pick the highest-scoring (most urgent) candidate
            _, target_idx = max(candidates, key=lambda x: x[0])
            target_subset = all_t_subsets[target_idx]
            
            # Build a block containing the target subset
            block = list(target_subset)
            
            # Add remaining points greedily to maximize coverage of uncovered subsets
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                block_set_local = set(block)
                available = [p for p in all_points if p not in block_set_local]
                
                if not available:
                    break
                
                best_point = None
                best_gain = -1
                
                # Compute gain for each available point
                for x in available:
                    # Count how many uncovered t-subsets contain x and (t-1) other points in block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = t_subset_index.get(s)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    
                    # Add a small diversity bonus for less-used points
                    diversity_bonus = 0.01 / (1 + point_freq[x])
                    total_score = gain + diversity_bonus
                    
                    if total_score > best_gain:
                        best_gain = total_score
                        best_point = x
                
                if best_point is None or best_gain <= 0:
                    # No good point found, just pick a random available point
                    best_point = rng.choice(available)
                
                block.append(best_point)
                point_freq[best_point] += 1
            
            # Finalize the block
            block.sort()
            block_tuple = tuple(block)
            
            # Check if this block is new
            if block_tuple not in block_set:
                block_set.add(block_tuple)
                blocks.append(block)
                
                # Update uncovered set and list
                newly_covered = []
                for s in itertools.combinations(block, t):
                    idx = t_subset_index.get(s)
                    if idx is not None and idx in uncovered_set:
                        uncovered_set.discard(idx)
                        newly_covered.append(idx)
                
                # Remove newly covered from uncovered_list
                if newly_covered:
                    newly_covered_set = set(newly_covered)
                    uncovered_list = [idx for idx in uncovered_list if idx not in newly_covered_set]
            
            if not uncovered_list:
                break
        
        return blocks
    
    # Try multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    
    # First, do a deterministic greedy to get a baseline
    baseline = greedy_once(0)
    if baseline:
        best_blocks = baseline
        best_count = len(baseline)
    
    # Now try randomized restarts
    seed = 1
    while time.time() < deadline - 5:  # leave 5s buffer
        blocks = greedy_once(seed)
        if blocks and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Return the best result found
    if best_blocks is None:
        return []
    
    return best_blocks
