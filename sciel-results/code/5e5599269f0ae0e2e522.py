"""Covering design C(19,7,4) via bitmask vectorized greedy + local search."""
import time
import itertools
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute all t-subsets and map to bitmask
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    
    # For each point x, precompute bitmask of t-subsets containing x
    # Use numpy uint64 bitmask: n_tsubs = C(19,4) = 3876, needs 3876 bits = 61 uint64
    n_words = (n_tsubs + 63) // 64
    point_masks = np.zeros((v, n_words), dtype=np.uint64)
    for x in range(v):
        for i, s in enumerate(all_tsubs):
            if x in s:
                point_masks[x, i // 64] |= (1 << (i % 64))
    
    def mask_and_count(m1, m2):
        """Count bits set in (m1 & m2)."""
        return int(np.sum(np.bitwise_and(m1, m2).view(np.uint8).reshape(-1, 8), axis=1))
    
    def count_overlap(mask, uncovered):
        """Count bits set in (mask & uncovered)."""
        return int(np.sum(np.bitwise_and(mask, uncovered).view(np.uint8).reshape(-1, 8), axis=1))
    
    def update_uncovered(uncovered, mask):
        """Remove bits in mask from uncovered."""
        np.bitwise_and(uncovered, np.invert(mask), out=uncovered)
    
    def greedy_cover(rng, max_blocks=None):
        """Randomized greedy set cover using bitmask operations."""
        uncovered = np.ones(n_words, dtype=np.uint64)
        # Mask out bits beyond n_tsubs
        extra_bits = n_words * 64 - n_tsubs
        if extra_bits > 0:
            uncovered[-1] &= ((1 << extra_bits) - 1)
        
        blocks = []
        total_covered = 0
        
        while True:
            if max_blocks is not None and len(blocks) >= max_blocks:
                break
            if np.all(uncovered == 0):
                break
            if time.time() > deadline - 5:
                break
            
            # Pick a random uncovered t-subset as seed
            # Find an index where uncovered has a bit set
            flat = uncovered.view(np.uint8).flatten()
            set_bits = np.nonzero(flat)[0]
            if len(set_bits) == 0:
                break
            # Pick a random set bit
            byte_idx = set_bits[rng.randint(0, len(set_bits) - 1)]
            byte_val = flat[byte_idx]
            # Pick a random set bit within that byte
            bit_positions = [b for b in range(8) if byte_val & (1 << b)]
            bit_pos = bit_positions[rng.randint(0, len(bit_positions) - 1)]
            tsub_idx = (byte_idx // 8) * 64 + (byte_idx % 8) * 8 + bit_pos
            if tsub_idx >= n_tsubs:
                continue
            
            target = all_tsubs[tsub_idx]
            block = list(target)
            rng.shuffle(block)
            block_set = set(block)
            
            # Current mask of t-subsets covered by current block
            block_mask = np.zeros(n_words, dtype=np.uint64)
            for x in block:
                block_mask |= point_masks[x]
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                rng.shuffle(candidates)
                
                best_x = None
                best_gain = -1
                
                for x in candidates:
                    new_mask = block_mask | point_masks[x]
                    gain = count_overlap(new_mask, uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                block_mask |= point_masks[best_x]
            
            if not block:
                break
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            update_uncovered(uncovered, block_mask)
            total_covered += count_overlap(block_mask, np.ones(n_words, dtype=np.uint64))
        
        # Count actually covered
        covered_mask = np.zeros(n_words, dtype=np.uint64)
        for b in blocks:
            bm = np.zeros(n_words, dtype=np.uint64)
            for x in b:
                bm |= point_masks[x]
            covered_mask |= bm
        n_covered = count_overlap(covered_mask, np.ones(n_words, dtype=np.uint64))
        
        return blocks, n_covered
    
    def local_search(blocks):
        """Remove redundant blocks and try to swap for improvement."""
        if not blocks:
            return blocks
        
        # Build coverage count per t-subset
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            for s in itertools.combinations(b, t):
                cover_count[tsub_to_idx[s]] += 1
        
        # Remove blocks that are fully redundant (all their t-subsets covered by others)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if removing b leaves all its t-subsets covered
                redundant = True
                for s in itertools.combinations(b, t):
                    if cover_count[tsub_to_idx[s]] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b
                    for s in itertools.combinations(b, t):
                        cover_count[tsub_to_idx[s]] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        return blocks
    
    # Run multiple randomized restarts
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    n_restarts = 500  # Will be limited by time
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        blocks, n_covered = greedy_cover(rng)
        
        if n_covered == n_tsubs:
            # Complete covering - apply local search
            refined = local_search(blocks)
            # Verify refinement is still valid
            if _verify(refined, v, t, tsub_to_idx):
                if len(refined) < best_count:
                    best_count = len(refined)
                    best_blocks = refined[:]
        
        # If we've found a solution, keep searching for better
        if best_blocks is not None and restart % 50 == 0:
            if time.time() > deadline - 30:
                break
    
    if best_blocks is not None:
        return best_blocks
    
    # Fallback: one more greedy
    blocks, _ = greedy_cover(rng)
    return blocks


def _verify(blocks, v, t, tsub_to_idx):
    """Verify that all t-subsets are covered."""
    covered = set()
    for b in blocks:
        for s in itertools.combinations(b, t):
            covered.add(tsub_to_idx[s])
    return len(covered) == len(tsub_to_idx)
