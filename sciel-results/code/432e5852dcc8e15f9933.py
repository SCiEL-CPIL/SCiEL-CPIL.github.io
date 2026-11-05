import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    # 1. All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_index = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # 2. All k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    # 3. Map each block to the indices of t-subsets it covers
    block_to_t_indices = []
    for block in all_k_blocks:
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(t_subset_index[sub])
        block_to_t_indices.append(indices)
        
    # 4. Map each t-subset to the list of block indices covering it
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_to_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    # Build sparse matrix for vectorized operations
    # Matrix M: num_k x num_t, M[b, t] = 1 if block b covers t-subset t
    M_rows = []
    M_cols = []
    for b_idx, indices in enumerate(block_to_t_indices):
        for t_idx in indices:
            M_rows.append(b_idx)
            M_cols.append(t_idx)
    
    M = np.zeros((num_k, num_t), dtype=np.int8)
    M[np.array(M_rows), np.array(M_cols)] = 1
    
    # Precompute block coverage counts
    block_cover_count = M.sum(axis=1)
    
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        uncovered = np.ones(num_t, dtype=bool)
        uncovered_count = num_t
        cover = []
        
        while uncovered_count > 0:
            # Vectorized: find best block by counting newly covered t-subsets
            # new_cover = M[:, uncovered] sum over axis 1
            # But we only need to consider blocks not yet in cover
            # For simplicity, just compute for all blocks
            
            # Compute how many uncovered t-subsets each block covers
            new_cover_counts = M[:, uncovered].sum(axis=1)
            
            # Pick the block with maximum new coverage
            best_block = int(np.argmax(new_cover_counts))
            
            if new_cover_counts[best_block] == 0:
                # No block can cover any remaining uncovered subset
                break
                
            cover.append(best_block)
            
            # Update uncovered
            newly_covered = M[best_block, :]
            uncovered[~uncovered & newly_covered.astype(bool)] = False
            uncovered_count = int(uncovered.sum())
                
        return cover

    def local_search(cover, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        if not cover:
            return cover
            
        # Initialize coverage counts using matrix
        counts = np.zeros(num_t, dtype=np.int8)
        for b_idx in cover:
            counts += M[b_idx, :]
        
        improved = True
        no_improvement_streak = 0
        max_iterations = 100
        
        while improved and no_improvement_streak < 10 and (time.time() < deadline - 5.0):
            improved = False
            
            # Shuffle order to avoid bias
            indices = list(range(len(cover)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cover):
                    continue
                b_idx = cover[i]
                
                # Find essential t-subsets for b_idx (those with count == 1)
                essential_mask = (counts == 1) & M[b_idx, :].astype(bool)
                essential_count = int(essential_mask.sum())
                
                if essential_count == 0:
                    # Block is redundant - remove it
                    counts -= M[b_idx, :]
                    cover.pop(i)
                    improved = True
                    no_improvement_streak = 0
                    break  # Restart loop
                else:
                    # Try to find a replacement block that covers all essential t-subsets
                    # Get indices of essential t-subsets
                    essential_indices = np.where(essential_mask)[0]
                    
                    # Find blocks that cover all essential t-subsets
                    # A block c is valid if M[c, essential_indices].all()
                    # But we need to be careful about performance
                    
                    # Start with blocks that cover the first essential subset
                    first_essential = essential_indices[0]
                    candidate_blocks = np.where(M[:, first_essential] == 1)[0]
                    
                    # Filter candidates that cover all essential subsets
                    valid_candidates = []
                    for c_idx in candidate_blocks:
                        if c_idx == b_idx:
                            continue
                        if M[c_idx, essential_indices].all():
                            valid_candidates.append(c_idx)
                    
                    if valid_candidates:
                        # Score each candidate: count how many t-subsets with count==1 
                        # (but not essential) it covers, to potentially make other blocks redundant
                        best_cand = None
                        best_score = -1
                        
                        for c_idx in valid_candidates:
                            # Count t-subsets that are count==1 and covered by c_idx but not essential
                            non_essential_ones = (counts == 1) & ~essential_mask
                            score = int((M[c_idx, :] & non_essential_ones.astype(np.int8)).sum())
                            
                            if score > best_score:
                                best_score = score
                                best_cand = c_idx
                                
                        if best_cand is not None:
                            # Perform swap
                            counts -= M[b_idx, :]
                            counts += M[best_cand, :]
                            cover[i] = best_cand
                            improved = True
                            no_improvement_streak = 0
                            break  # Restart loop
                            
            if not improved:
                no_improvement_streak += 1

        return cover

    def verify_cover(cover):
        """Verify that the cover is valid"""
        if not cover:
            return num_t == 0
        counts = np.zeros(num_t, dtype=np.int8)
        for b_idx in cover:
            counts += M[b_idx, :]
        return int((counts == 0).sum()) == 0

    best_cover = None
    best_len = float('inf')
    
    # Run multiple instances
    seed_counter = 0
    while time.time() < deadline - 10.0:
        seed = seed_counter
        seed_counter += 1
        
        # Initial greedy
        cover = greedy_cover(seed)
        if not cover:
            continue
            
        # Local search
        cover = local_search(cover, seed + 1000)
        
        # Verify
        if not verify_cover(cover):
            continue
            
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover.copy()
            
        # If we're doing well, we can break early
        if best_len <= 210:
            break

    # Convert block indices back to point lists
    if best_cover is None:
        # Fallback: run one more greedy
        cover = greedy_cover(0)
        if not cover:
            return []
        best_cover = cover
        
    result = []
    for b_idx in best_cover:
        result.append(list(all_k_blocks[b_idx]))
        
    return result

# Test locally if needed
# if __name__ == "__main__":
#     import time
#     start = time.time()
#     blocks = construct(19, 8, 5)
#     end = time.time()
#     print(f"Time: {end - start}, Blocks: {len(blocks)}")
