"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _build_tsub_index(v, t):
    """Build a mapping from t-subset to index, and return the list of all t-subsets."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    return all_tsubs, tsub_to_idx


def _greedy_cover_fast(v, k, t, rng, deadline, all_tsubs, tsub_to_idx):
    """Optimized greedy covering with precomputed gain tables."""
    n_tsubs = len(all_tsubs)
    
    # Precompute: for each (t-1)-subset, map to list of t-subset indices that extend it
    # For t=3: for each pair (a,b), list of triples containing both a and b
    # This is: for each (t-1)-combo, the t-subsets that contain it
    combo_to_tsubs = {}
    for i, ts in enumerate(all_tsubs):
        for rest in itertools.combinations(ts, t - 1):
            if rest not in combo_to_tsubs:
                combo_to_tsubs[rest] = []
            combo_to_tsubs[rest].append(i)
    
    # Uncovered mask
    uncovered_mask = np.ones(n_tsubs, dtype=bool)
    n_uncovered = n_tsubs
    
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            return None  # timeout
        
        # Pick a random uncovered t-subset to start a new block
        uncovered_indices = np.where(uncovered_mask)[0]
        start_idx = rng.choice(uncovered_indices)
        start_tsub = all_tsubs[start_idx]
        block = list(start_tsub)
        block_set = set(block)
        
        while len(block) < k:
            n_block = len(block)
            
            # Generate all (t-1)-subsets of block
            if t - 1 <= n_block:
                block_combos = list(itertools.combinations(block, t - 1))
            else:
                block_combos = []
            
            if not block_combos:
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
                block.append(best_point)
                block_set.add(best_point)
                continue
            
            # For each candidate x, compute gain using precomputed tables
            best_gain = -1
            best_point = -1
            best_points = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for rest in block_combos:
                    # Get all t-subsets containing this (t-1)-combo
                    tsub_indices = combo_to_tsubs.get(rest, [])
                    for idx in tsub_indices:
                        if uncovered_mask[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            if best_point == -1 or best_gain == 0:
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            elif len(best_points) > 1:
                best_point = rng.choice(best_points)
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered mask
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts, -1)
            if idx >= 0 and uncovered_mask[idx]:
                uncovered_mask[idx] = False
                n_uncovered -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset. Iterative until stable."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        changed = False
        kept_blocks = []
        for i, block in enumerate(blocks):
            all_redundant = True
            for ts in block_tsubs[i]:
                if cover_count[ts] <= 1:
                    all_redundant = False
                    break
            if not all_redundant:
                kept_blocks.append(block)
            else:
                # Remove this block
                changed = True
                for ts in block_tsubs[i]:
                    cover_count[ts] -= 1
        
        if changed:
            # Rebuild block_tsubs for remaining blocks
            blocks = kept_blocks
            block_tsubs = []
            for block in blocks:
                tsubs = list(itertools.combinations(block, t))
                block_tsubs.append(tsubs)
    
    return blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def _local_search(blocks, v, k, t, rng, deadline, all_tsubs, tsub_to_idx):
    """Local search: try removing blocks and replacing blocks to reduce total block count."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Compute coverage counts for current solution
    cover_count = defaultdict(int)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # Precompute combo_to_tsubs for faster gain calculation
    combo_to_tsubs = {}
    for i, ts in enumerate(all_tsubs):
        for rest in itertools.combinations(ts, t - 1):
            if rest not in combo_to_tsubs:
                combo_to_tsubs[rest] = []
            combo_to_tsubs[rest].append(i)
    
    # First: try to remove redundant blocks iteratively
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            is_redundant = True
            for ts in itertools.combinations(current_blocks[i], t):
                if cover_count[ts] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for ts in itertools.combinations(current_blocks[i], t):
                    cover_count[ts] -= 1
                current_blocks.pop(i)
                changed = True
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
    
    # Second: try block replacement - for each block, see if we can find a better block
    # that covers the same unique t-subsets plus more
    # Reduced iterations and trials for speed
    max_iterations = 20
    max_trials = 10
    
    for _iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break
            
            # Find t-subsets that are only covered by block i
            unique_tsubs = []
            for ts in itertools.combinations(current_blocks[i], t):
                if cover_count[ts] == 1:
                    unique_tsubs.append(ts)
            
            if not unique_tsubs:
                continue
            
            # Try to find a new block that covers all unique_tsubs and maximizes additional coverage
            best_new_block = None
            best_new_gain = 0
            
            # Sample some candidate blocks
            for _trial in range(max_trials):
                # Start with a random point from one of the unique tsubs
                start_ts = rng.choice(unique_tsubs)
                candidate = list(start_ts)
                cand_set = set(candidate)
                
                while len(candidate) < k:
                    # Pick point that maximizes coverage of uncovered t-subsets
                    block_combos = list(itertools.combinations(candidate, t - 1))
                    best_gain = -1
                    best_point = -1
                    
                    for x in range(v):
                        if x in cand_set:
                            continue
                        gain = 0
                        for rest in block_combos:
                            tsub_indices = combo_to_tsubs.get(rest, [])
                            for idx in tsub_indices:
                                if cover_count.get(all_tsubs[idx], 0) == 0:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    
                    if best_point == -1:
                        candidates = [x for x in range(v) if x not in cand_set]
                        best_point = rng.choice(candidates)
                    
                    candidate.append(best_point)
                    cand_set.add(best_point)
                
                candidate.sort()
                
                # Check if this candidate covers all unique_tsubs
                cand_tsubs = set(itertools.combinations(candidate, t))
                if all(ts in cand_tsubs for ts in unique_tsubs):
                    # Count additional coverage
                    add_gain = sum(1 for ts in cand_tsubs if cover_count.get(ts, 0) == 0)
                    if add_gain > best_new_gain:
                        best_new_gain = add_gain
                        best_new_block = candidate
            
            if best_new_block is not None:
                # Replace block i with best_new_block
                old_block = current_blocks[i]
                for ts in itertools.combinations(old_block, t):
                    cover_count[ts] -= 1
                for ts in itertools.combinations(best_new_block, t):
                    cover_count[ts] += 1
                current_blocks[i] = best_new_block
                improved = True
        
        if not improved:
            break
    
    # Final redundancy removal
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            is_redundant = True
            for ts in itertools.combinations(current_blocks[i], t):
                if cover_count[ts] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for ts in itertools.combinations(current_blocks[i], t):
                    cover_count[ts] -= 1
                current_blocks.pop(i)
                changed = True
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
    
    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 145  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Precompute t-subset indexing
    all_tsubs, tsub_to_idx = _build_tsub_index(v, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with varied seeds
    # Reduced restarts to allow more time per restart for local search
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Use different seed for each restart
        rng_restart = random.Random(42 + restart * 7)
        
        blocks = _greedy_cover_fast(v, k, t, rng_restart, deadline, all_tsubs, tsub_to_idx)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search refinement
        blocks = _local_search(blocks, v, k, t, rng_restart, deadline, all_tsubs, tsub_to_idx)
        
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
