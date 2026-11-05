import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map t-subset to index
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    
    # Precompute for each k-subset, which t-subsets it covers
    # We'll use bitmasks: bit i is set if t-subset i is covered
    # But with n_tsubsets = C(24,3) = 2024, we need 2024-bit integers
    # Python handles big integers natively
    
    # For each point x, precompute which t-subsets contain x
    # And for each (t-1)-subset p, precompute which t-subsets extend p
    
    # Strategy: use bitsets for uncovered t-subsets
    # uncovered_mask: bit i set iff t-subset i is uncovered
    
    best_blocks = None
    
    # Run multiple random trials
    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks, valid = _greedy_cover_bitset(v, k, t, all_tsubsets, tsub_to_idx, deadline)
        if valid and (best_blocks is None or len(blocks) < len(best_blocks)):
            best_blocks = blocks
        # Local search to try to reduce block count
        if best_blocks and len(best_blocks) > 76:
            # Try to remove redundant blocks
            improved = _local_search_remove(v, k, t, all_tsubsets, tsub_to_idx, best_blocks, deadline)
            if improved and len(improved) < len(best_blocks):
                best_blocks = improved
        # If we find a very good solution, we might stop early
        if best_blocks and len(best_blocks) <= 76:
            break
            
    if best_blocks is None:
        return []
    return best_blocks

def _greedy_cover_bitset(v, k, t, all_tsubsets, tsub_to_idx, deadline):
    n_tsubsets = len(all_tsubsets)
    
    # uncovered_mask: bit i set iff t-subset i is uncovered
    uncovered_mask = (1 << n_tsubsets) - 1
    blocks = []
    
    while uncovered_mask != 0:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed
        # Find set bits in uncovered_mask
        # For efficiency, pick a random position among set bits
        # We can use a trick: pick random bit position, check if set
        # Or iterate through set bits
        
        # Simple approach: find one set bit
        # Use bit_length and masking
        temp_mask = uncovered_mask
        # Pick a random offset among set bits
        # Count set bits first? That's expensive.
        # Just pick random positions until we find a set bit
        seed_idx = None
        for _ in range(100):
            candidate = random.randint(0, n_tsubsets - 1)
            if (uncovered_mask >> candidate) & 1:
                seed_idx = candidate
                break
        if seed_idx is None:
            # Fallback: find first set bit
            if uncovered_mask == 0:
                break
            # Find position of lowest set bit
            low = uncovered_mask & (-uncovered_mask)
            seed_idx = low.bit_length() - 1
        
        seed = all_tsubsets[seed_idx]
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            # For each candidate x not in block, compute gain
            # Gain = number of uncovered t-subsets that are subset of block ∪ {x} and contain x
            # Equivalently: t-subsets s such that x ∈ s and (s \ {x}) ⊆ block
            # Since |s| = t, s \ {x} has t-1 elements
            # So we need: for each (t-1)-subset p of block, check if p ∪ {x} is an uncovered t-subset
            
            # Precompute (t-1)-subsets of block
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for p in t_minus_1_subsets:
                    # Create t-subset p ∪ {x}
                    s = tuple(sorted(p + (x,)))
                    idx = tsub_to_idx.get(s)
                    if idx is not None and (uncovered_mask >> idx) & 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # Should not happen if uncovered_mask != 0, but safety check
                # Try to find any x that covers at least one uncovered t-subset
                found = False
                for x in range(v):
                    if x in block_set:
                        continue
                    t_minus_1_subsets = list(itertools.combinations(block, t-1))
                    for p in t_minus_1_subsets:
                        s = tuple(sorted(p + (x,)))
                        idx = tsub_to_idx.get(s)
                        if idx is not None and (uncovered_mask >> idx) & 1:
                            best_x = x
                            best_gain = 1
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return None, False
            
            block.append(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets from uncovered_mask
        for s in itertools.combinations(block, t):
            idx = tsub_to_idx[s]
            uncovered_mask &= ~(1 << idx)
            
    return blocks, True

def _local_search_remove(v, k, t, all_tsubsets, tsub_to_idx, blocks, deadline):
    """Try to remove redundant blocks from the covering."""
    if not blocks:
        return None
    
    n_tsubsets = len(all_tsubsets)
    n_blocks = len(blocks)
    
    # For each t-subset, track which blocks cover it
    # covered_by[i] = bitmask of blocks that cover t-subset i
    covered_by = [0] * n_tsubsets
    
    for bi, block in enumerate(blocks):
        bit = 1 << bi
        for s in itertools.combinations(block, t):
            idx = tsub_to_idx[s]
            covered_by[idx] |= bit
    
    # Try to remove each block if its removal doesn't leave any t-subset uncovered
    improved = True
    result_blocks = list(blocks)
    
    while improved:
        improved = False
        for bi in range(len(result_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if block bi is essential: is there any t-subset covered only by this block?
            bit = 1 << bi
            essential = False
            for idx in range(n_tsubsets):
                if covered_by[idx] == bit:
                    essential = True
                    break
            
            if not essential:
                # Remove block bi
                result_blocks.pop(bi)
                # Update covered_by
                for s in itertools.combinations(blocks[bi], t):
                    idx = tsub_to_idx[s]
                    covered_by[idx] &= ~bit
                improved = True
                break  # Restart from beginning since indices changed
    
    if len(result_blocks) < len(blocks):
        return result_blocks
    return None
