"""Randomized greedy covering for C(19,7,4) with weighted block selection."""
import itertools
import random
import time
import numpy as np
import math

# Precompute all 4-subsets of 19 points as tuples for index mapping
# We need a fast way to map a 4-subset to an index in a flat array.
# There are C(19,4) = 3876 subsets.
_ALL_T_SUBSETS = list(itertools.combinations(range(19), 4))
_T_SUBSET_INDEX = {s: i for i, s in enumerate(_ALL_T_SUBSETS)}
_N_T_SUBSETS = len(_ALL_T_SUBSETS)


def _subsets_to_indices(subsets):
    """Convert a list of tuples to an array of indices."""
    return np.array([_T_SUBSET_INDEX[s] for s in subsets], dtype=np.int32)


def _generate_candidate_blocks(v, k, t, n_candidates, rng):
    """
    Generate a list of candidate blocks.
    Mix of random blocks and blocks completed via greedy heuristic.
    """
    candidates = []
    # Generate some purely random blocks
    for _ in range(n_candidates // 3):
        block = sorted(rng.sample(range(v), k))
        candidates.append(block)
    
    # Generate blocks started from random t-subsets and completed greedily
    for _ in range(2 * n_candidates // 3):
        seed = rng.sample(range(v), t)
        block_set = set(seed)
        block = list(seed)
        
        # Greedily add points to maximize coverage of potential t-subsets
        # Note: This is a local heuristic for block generation, not the main selection.
        for _ in range(k - t):
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Estimate gain: how many new t-subsets are formed?
                # Actually, for generation, we just want diverse blocks.
                # A simple heuristic: pick a point that is not in the block yet.
                # To make it slightly smarter, we can pick randomly from remaining.
                # Or we can use a quick check against a small sample of uncovered sets?
                # For speed, let's just pick randomly from remaining for now, 
                # but bias towards points that appear in many potential extensions?
                # Let's stick to random for generation speed, the selection step handles quality.
                pass
            
            # Random selection for the rest of the block
            remaining = [x for x in range(v) if x not in block_set]
            if remaining:
                x = rng.choice(remaining)
                block.append(x)
                block_set.add(x)
        
        candidates.append(sorted(block))
        
    return candidates


def construct(v, k, t):
    """
    Randomized greedy set cover with weighted selection.
    """
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for evaluation overhead
    
    # Precompute indices for all t-subsets
    all_t_indices = np.arange(_N_T_SUBSETS, dtype=np.int32)
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of candidate blocks to generate per step
    n_candidates = 200 
    # Temperature for probability selection: P(i) = gain(i)^temp / sum(gain^temp)
    # temp > 1 favors high gain, temp < 1 favors exploration
    temp = 0.5 
    
    num_trials = 0
    for trial in range(100):  # Max trials, will break on time
        if time.time() - start_time > time_limit - 5:
            break
        
        rng = random.Random(trial * 7919 + 13)
        
        # Initialize uncovered mask
        uncovered = np.ones(_N_T_SUBSETS, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Generate candidate blocks
            candidates = _generate_candidate_blocks(v, k, t, n_candidates, rng)
            
            # Calculate gain for each candidate
            # Gain = number of currently uncovered t-subsets covered by the block
            gains = np.zeros(len(candidates), dtype=np.int32)
            
            for i, block in enumerate(candidates):
                # Get indices of all t-subsets in this block
                block_t_subsets = list(itertools.combinations(block, t))
                if not block_t_subsets:
                    continue
                indices = _subsets_to_indices(block_t_subsets)
                # Count how many of these are currently uncovered
                gains[i] = np.sum(uncovered[indices])
            
            # Filter out candidates with 0 gain
            valid_mask = gains > 0
            if not np.any(valid_mask):
                # Fallback: pick a random block or break if stuck
                # This shouldn't happen if v, k, t are valid and uncovered is non-empty
                # unless all candidates are bad. Force a random block.
                block = sorted(rng.sample(range(v), k))
                blocks.append(block)
                indices = _subsets_to_indices(list(itertools.combinations(block, t)))
                uncovered[indices] = False
                continue
            
            valid_gains = gains[valid_mask]
            valid_candidates = [c for c, m in zip(candidates, valid_mask) if m]
            
            # Select block with probability proportional to gain^temp
            weights = valid_gains ** temp
            total_w = weights.sum()
            
            if total_w == 0:
                # Fallback to uniform
                idx = rng.randint(0, len(valid_candidates) - 1)
            else:
                # Roulette wheel selection
                r = rng.random() * total_w
                cumsum = 0
                idx = len(valid_candidates) - 1
                for j, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        idx = j
                        break
            
            chosen_block = valid_candidates[idx]
            blocks.append(chosen_block)
            
            # Update uncovered mask
            block_t_subsets = list(itertools.combinations(chosen_block, t))
            indices = _subsets_to_indices(block_t_subsets)
            uncovered[indices] = False
        
        if not np.any(uncovered) and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        num_trials += 1
    
    # If we never completed a full cover (time ran out), fall back to a simple greedy
    if best_blocks is None:
        uncovered = set(_ALL_T_SUBSETS)
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to seed a block
            seed = min(uncovered)
            block = list(seed)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for combo in itertools.combinations(block, t):
                uncovered.discard(combo)
        best_blocks = blocks
    
    return best_blocks
