"""Bitset-optimized greedy covering with randomized restarts and redundancy pruning."""
import itertools
import time
import random

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    
    # Generate all t-subsets and assign them indices for fast lookup
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute all possible k-blocks
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute bitsets for each block
    # Bit i is 1 if the block covers t-subset i
    block_bitsets = [0] * num_kblocks
    for bi, b in enumerate(all_kblocks):
        bits = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            bits |= (1 << idx)
        block_bitsets[bi] = bits
        
    def bit_count(x):
        return x.bit_count()
        
    best_solution = None
    best_size = float('inf')
    
    # Run greedy multiple times with random tie-breaking
    restarts = 0
    max_restarts = 100 # Limit by time
    
    while time.monotonic() < deadline - 10:
        restarts += 1
        random.seed(restarts)
        
        uncovered = (1 << num_tsubs) - 1
        selected_blocks = []
        cover_counts = [0] * num_tsubs
        
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            tie_breakers = []
            
            for bi in range(num_kblocks):
                b_bits = block_bitsets[bi]
                if not (b_bits & uncovered):
                    continue
                gain = bit_count(b_bits & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    tie_breakers = [bi]
                elif gain == best_gain:
                    tie_breakers.append(bi)
                    
            if best_block_idx == -1:
                break
                
            # Pick randomly from ties to diversify
            if len(tie_breakers) > 1:
                bi = random.choice(tie_breakers)
            else:
                bi = best_block_idx
                
            selected_blocks.append(bi)
            b_bits = block_bitsets[bi]
            
            # Update cover counts
            temp_bits = b_bits
            while temp_bits:
                lsb = temp_bits & -temp_bits
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                temp_bits ^= lsb
                
            uncovered &= ~b_bits
            
            if time.monotonic() > deadline - 5:
                break
                
        # Phase 2: Redundancy Removal
        changed = True
        while changed and time.monotonic() < deadline - 5:
            changed = False
            n_sel = len(selected_blocks)
            for i in range(n_sel - 1, -1, -1):
                bi = selected_blocks[i]
                b_bits = block_bitsets[bi]
                is_redundant = True
                temp_bits = b_bits
                while temp_bits:
                    lsb = temp_bits & -temp_bits
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        is_redundant = False
                        break
                    temp_bits ^= lsb
                    
                if is_redundant:
                    # Remove block i
                    temp_bits = b_bits
                    while temp_bits:
                        lsb = temp_bits & -temp_bits
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        temp_bits ^= lsb
                        
                    selected_blocks.pop(i)
                    changed = True
        
        # Check validity
        final_uncovered = (1 << num_tsubs) - 1
        for bi in selected_blocks:
            final_uncovered &= ~block_bitsets[bi]
            
        if final_uncovered == 0 and len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            
    if best_solution is None:
        # Fallback: return a valid covering if possible, else empty
        # If no solution found, it's likely due to timeout or bug.
        # We return an empty list which is invalid, but better than crashing.
        return []
        
    return [list(all_kblocks[bi]) for bi in best_solution]
