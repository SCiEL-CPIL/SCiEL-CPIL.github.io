import itertools
import random
import time
import sys
import numpy as np

def _precompute(v, k, t):
    """
    Precompute all t-sets, their indices, and a mapping from t-set index to bitmask
    of which points are in it (though we mostly need the t-set index to block mask mapping).
    Actually, we need:
    1. List of all t-sets.
    2. For any block (k-subset), compute the bitmask of t-sets it covers.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Precompute masks for all possible blocks? 
    # C(24, 7) = 346,104.
    # Storing 346k masks of 2024 bits each.
    # 346k * (2024/8) bytes ~ 346k * 253 bytes ~ 87 MB.
    # This is acceptable memory-wise and makes evaluation O(1) for any block.
    
    block_to_mask = {}
    # To speed up generation, we can iterate over all combinations.
    # However, generating 346k masks might take a few seconds.
    # Let's see if we can generate them on the fly or cache them.
    # Given 150s limit, precomputing once is fine if it takes < 5s.
    
    # Optimization: Use numpy for vectorized mask generation?
    # Or just loop.
    # Looping 346k times, each taking ~30 t-sets to check -> 10M ops. Python is slow.
    # Let's try to be smarter.
    
    # Alternative: Don't precompute ALL blocks. 
    # In SA, we propose random blocks. We can compute the mask of a proposed block on the fly.
    # Computing mask of one block: C(7,3)=35 lookups in tset_to_idx. Very fast.
    # So we don't need to precompute all blocks.
    
    return all_tsets, n_tsets, tset_to_idx

def _block_to_mask(block, tset_to_idx, t):
    """Compute bitmask for a specific block."""
    mask = 0
    for ts in itertools.combinations(block, t):
        idx = tset_to_idx[ts]
        mask |= (1 << idx)
    return mask

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    all_tsets, n_tsets, tset_to_idx = _precompute(v, k, t)
    full_mask = (1 << n_tsets) - 1
    
    # 1. Greedy Initialization
    # Fast greedy to get a starting solution.
    rng = random.Random(42)
    current_blocks = []
    current_covered_mask = 0
    uncovered_count = n_tsets
    
    # Greedy loop
    # To speed up, we don't scan ALL blocks. We use a heuristic candidate generation.
    # Or we can scan a large random sample.
    # Let's do a standard greedy but with a time limit per step? No, greedy is fast enough if we are smart.
    # Actually, scanning 346k blocks every step is too slow.
    # Heuristic: Pick a random uncovered t-set. Generate candidates containing it.
    
    while uncovered_count > 0 and time.time() < deadline - 20.0:
        # Find an uncovered t-set
        # We need to find an index i such that bit i in current_covered_mask is 0.
        # Iterate bits?
        # Or keep a list of uncovered indices? Updating it is O(n_tsets).
        # Let's just find the first 0 bit.
        
        # Efficient way to find a 0 bit:
        # Invert the mask?
        # ~current_covered_mask & full_mask
        # Find least significant set bit.
        inv_mask = ~current_covered_mask & full_mask
        if inv_mask == 0:
            break
            
        # Get index of LSB
        lsb = inv_mask & -inv_mask
        target_idx = lsb.bit_length() - 1
        target_tset = all_tsets[target_idx]
        
        # Generate candidates containing target_tset
        # Points in target_tset: T
        # Remaining points: V \ T
        # Choose k-t points from V \ T
        remaining_points = [p for p in range(v) if p not in target_tset]
        num_to_choose = k - t
        
        # Sample candidates
        # If C(remaining, k-t) is small, enumerate. Else sample.
        # C(21, 4) = 5985. Small enough to enumerate?
        # 5985 candidates * 35 t-sets each = 200k ops.
        # If we do this 70 times (for 70 blocks), that's 14M ops.
        # Python might handle 14M simple ops in ~5-10s.
        # Let's enumerate for accuracy in the greedy phase.
        
        best_gain = -1
        best_block = None
        best_mask = 0
        
        for combo in itertools.combinations(remaining_points, num_to_choose):
            block = tuple(sorted(target_tset + combo))
            mask = _block_to_mask(block, tset_to_idx, t)
            gain = bin(mask & ~current_covered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_mask = mask
                if gain == 0: # Can't be better
                    break # Actually, if gain is 0 for the first, it might be 0 for all? No.
                
        if best_gain <= 0:
            # Should not happen if uncovered_count > 0
            break
            
        current_blocks.append(list(best_block))
        current_covered_mask |= best_mask
        uncovered_count = bin(~current_covered_mask & full_mask).count('1')
        
    # Initial solution found.
    # Now Simulated Annealing.
    
    # State:
    # blocks: list of tuples
    # covered_mask: int
    # cost: number of blocks (we want to minimize this, but SA usually minimizes energy.
    # Energy = len(blocks) * BIG_CONSTANT - covered_count?
    # Or Energy = len(blocks) if covered_count == n_tsets else INF?
    # Better: Objective = len(blocks) * 10000 - covered_count.
    # We want to minimize this.
    
    current_energy = len(current_blocks) * 10000 - bin(current_covered_mask).count('1')
    best_blocks = [b[:] for b in current_blocks]
    best_energy = current_energy
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.9995 # Cooling rate
    # We need to run many iterations.
    
    # Precompute masks for current blocks?
    # current_masks = [_block_to_mask(b, tset_to_idx, t) for b in current_blocks]
    
    # To update mask efficiently when adding/removing/swapping, we just recalculate the union?
    # Union of 70 masks is fast (OR operation).
    
    while time.time() < deadline - 5.0:
        # Propose a move
        move_type = rng.random()
        
        # Copy state for evaluation
        # We need to calculate new covered_mask and new block count.
        
        if move_type < 0.3:
            # Add a random block
            new_block = tuple(sorted(rng.sample(range(v), k)))
            # Check if already in blocks?
            # If yes, skip or treat as no-op.
            # Using a set for blocks is faster for lookup.
            block_set = set(current_blocks)
            if new_block in block_set:
                continue
                
            new_blocks = current_blocks + [list(new_block)]
            new_mask = current_covered_mask | _block_to_mask(new_block, tset_to_idx, t)
            new_energy = len(new_blocks) * 10000 - bin(new_mask).count('1')
            
        elif move_type < 0.6:
            # Remove a random block
            if not current_blocks:
                continue
            idx_to_remove = rng.randint(0, len(current_blocks) - 1)
            new_blocks = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
            
            # Recalculate mask
            # This is O(N_blocks).
            new_mask = 0
            for b in new_blocks:
                new_mask |= _block_to_mask(b, tset_to_idx, t)
                
            new_energy = len(new_blocks) * 10000 - bin(new_mask).count('1')
            
        else:
            # Swap: Remove one, Add one
            if not current_blocks:
                continue
            idx_to_remove = rng.randint(0, len(current_blocks) - 1)
            new_block = tuple(sorted(rng.sample(range(v), k)))
            
            # Check if new_block is the one being removed?
            # Or if it's already in the rest?
            # Simplify: just build the new list and mask.
            new_blocks = [b for i, b in enumerate(current_blocks) if i != idx_to_remove]
            # If new_block is already in new_blocks, skip?
            # Let's just add it. Duplicates are handled by mask OR, but count increases?
            # No, if we add a duplicate, len increases but mask doesn't. Energy increases.
            # So it's naturally penalized.
            
            new_blocks.append(list(new_block))
            
            new_mask = 0
            for b in new_blocks:
                new_mask |= _block_to_mask(b, tset_to_idx, t)
                
            new_energy = len(new_blocks) * 10000 - bin(new_mask).count('1')

        # Acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0:
            # Always accept improvements
            current_blocks = new_blocks
            current_covered_mask = new_mask
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = [b[:] for b in current_blocks]
        else:
            # Accept with probability exp(-delta_E / T)
            if T > 0:
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    current_blocks = new_blocks
                    current_covered_mask = new_mask
                    current_energy = new_energy
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Reheat? Or just keep going at T_min.
            # With T_min, we are doing random walk / local search.
            
    # Final Check and Cleanup
    # Remove redundant blocks from best_blocks to ensure minimal count for the coverage?
    # The SA might have kept redundant blocks if they were part of a "good" energy state
    # (though energy penalizes count, so it should prefer fewer).
    # But if a block is redundant, removing it decreases count and keeps coverage same -> Energy decreases.
    # So SA should have removed it.
    # However, due to stochastic nature, it might miss it.
    # Let's do a final greedy reduction on best_blocks.
    
    final_blocks = best_blocks
    # Build index for final reduction
    # Check each block if it's redundant
    # O(N^2 * C(k,t))
    reduced = True
    while reduced and time.time() < deadline - 1.0:
        reduced = False
        n = len(final_blocks)
        # Precompute coverage counts for each t-set in final_blocks
        # Map t-set to count of blocks covering it
        tset_counts = {}
        for b in final_blocks:
            for ts in itertools.combinations(b, t):
                tset_counts[ts] = tset_counts.get(ts, 0) + 1
        
        for i in range(n):
            if time.time() > deadline - 1.0:
                break
            b = final_blocks[i]
            redundant = True
            for ts in itertools.combinations(b, t):
                if tset_counts[ts] == 1:
                    redundant = False
                    break
            if redundant:
                final_blocks.pop(i)
                reduced = True
                break # Restart loop with new list
    
    return final_blocks

if __name__ == "__main__":
    # Test
    blocks = construct(24, 7, 3)
    print(f"Found covering with {len(blocks)} blocks")
    # Verify
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, 3):
            covered.add(ts)
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered {len(covered)}/{total} t-sets")
