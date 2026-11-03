import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute masks for each block
    # Mask is an integer where bit i is set if triple i is covered by the block
    block_masks = [0] * n_blocks
    # triple_to_blocks: list of lists of block indices
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            triple_to_blocks[idx].append(i)
        block_masks[i] = mask
    
    full_mask = (1 << n_triples) - 1
    
    # Fast Greedy with randomization
    def fast_greedy():
        uncovered_mask = full_mask
        blocks = []
        # Maintain a list of uncovered triple indices for random sampling
        uncovered_list = list(range(n_triples))
        
        while uncovered_mask:
            best_block_idx = -1
            best_gain = -1
            
            # Sample candidates
            candidates = set()
            # Number of samples per step
            n_samples = 20
            
            for _ in range(n_samples):
                if not uncovered_list:
                    break
                # Pick random uncovered triple
                tri_idx = random.choice(uncovered_list)
                # Pick random block covering it
                # triple_to_blocks[tri_idx] is a list
                b_idx = random.choice(triple_to_blocks[tri_idx])
                candidates.add(b_idx)
            
            if not candidates:
                # Fallback if no candidates (should not happen)
                tri_idx = uncovered_list[0]
                candidates.add(random.choice(triple_to_blocks[tri_idx]))
            
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen
                break
            
            blocks.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
            # Update uncovered_list
            # Filter out covered triples
            # Optimization: only rebuild if necessary or use a set?
            # Rebuilding is O(N). N=2024. ~80 steps. 160k ops. Fine.
            uncovered_list = [i for i in uncovered_list if (uncovered_mask >> i) & 1]
            
        return blocks
    
    # Local Search
    def local_search(blocks):
        blocks = list(blocks)
        covered_mask = 0
        for b in blocks:
            covered_mask |= block_masks[b]
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove Redundant
            for i in range(len(blocks)):
                b_i = blocks[i]
                # Check if b_i is redundant
                # Redundant if mask[b_i] is subset of (covered_mask XOR mask[b_i]?) No.
                # Redundant if (mask[b_i] & ~covered_mask) == 0? No.
                # Redundant if mask[b_i] is covered by others.
                # others_mask = covered_mask & ~mask[b_i]
                # If mask[b_i] & ~others_mask == 0, then mask[b_i] is subset of others_mask.
                others_mask = covered_mask & ~block_masks[b_i]
                if (block_masks[b_i] & ~others_mask) == 0:
                    # Remove b_i
                    blocks.pop(i)
                    covered_mask = 0
                    for b in blocks:
                        covered_mask |= block_masks[b]
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. 2-for-1 Replacement
            n = len(blocks)
            for i in range(n):
                for j in range(i + 1, n):
                    b_i = blocks[i]
                    b_j = blocks[j]
                    mask_ij = block_masks[b_i] | block_masks[b_j]
                    others_mask = covered_mask & ~mask_ij
                    excl = mask_ij & ~others_mask
                    
                    if excl == 0:
                        continue
                    
                    # Find a block k that covers all triples in excl
                    # Get triple indices in excl
                    tri_indices = []
                    x = excl
                    # Limit number of triples to check for intersection efficiency
                    while x and len(tri_indices) < 10:
                        lsb = x & -x
                        idx = lsb.bit_length() - 1
                        tri_indices.append(idx)
                        x ^= lsb
                        
                    if not tri_indices:
                        continue
                        
                    # Intersect triple_to_blocks for these triples
                    # Use sets for intersection
                    candidates = set(triple_to_blocks[tri_indices[0]])
                    for t in tri_indices[1:]:
                        candidates &= set(triple_to_blocks[t])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Find a valid candidate not equal to b_i or b_j
                        valid_cands = [c for c in candidates if c != b_i and c != b_j]
                        if valid_cands:
                            new_b = random.choice(valid_cands)
                            # Replace blocks[i] with new_b, remove blocks[j]
                            blocks[i] = new_b
                            blocks.pop(j)
                            covered_mask = 0
                            for b in blocks:
                                covered_mask |= block_masks[b]
                            improved = True
                            break
                if improved:
                    break
                    
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        blocks = fast_greedy()
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        # Optional: If we reach a very good score (e.g., 76), we can try to stop early or keep going
        # But keeping going might find 75.
        
    if best_blocks is None:
        # Fallback: return a simple greedy result
        blocks = fast_greedy()
        best_blocks = blocks
        
    return [list(all_blocks[i]) for i in best_blocks]
