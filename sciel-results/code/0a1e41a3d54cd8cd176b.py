import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map t-subset to index
    mask_to_idx = {}
    for i, sub in enumerate(all_t_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        mask_to_idx[mask] = i

    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Precompute which t-subsets each k-block covers
    # We will store this as a list of lists of indices for fast access, 
    # and also build a sparse matrix for vectorized gain calculation.
    
    block_t_indices = []
    rows = []
    cols = []
    data = []
    
    for b_idx, block in enumerate(all_k_blocks):
        idxs = []
        for t_sub in itertools.combinations(block, t):
            mask = 0
            for p in t_sub:
                mask |= (1 << p)
            idx = mask_to_idx.get(mask)
            if idx is not None:
                idxs.append(idx)
                rows.append(b_idx)
                cols.append(idx)
                data.append(1)
        block_t_indices.append(idxs)
        
    # Create sparse matrix M where M[b, s] = 1 if block b covers t-subset s
    # Shape: (n_k_blocks, n_t_subs)
    M = sparse.csr_matrix((data, (rows, cols)), shape=(n_k_blocks, n_t_subs))
    
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        # uncovered: boolean array, True if not yet covered
        uncovered = np.ones(n_t_subs, dtype=bool)
        # num_uncovered: count of True values
        num_uncovered = n_t_subs
        
        blocks_selected = []
        used_blocks = set()
        
        while num_uncovered > 0:
            # Calculate gain for each block: number of uncovered subsets it covers
            # M is (n_blocks, n_t_subs), uncovered is (n_t_subs,)
            # gain[b] = sum(M[b, s] * uncovered[s] for s in range(n_t_subs))
            gains = M.dot(uncovered)
            
            # To introduce randomness and diversity, pick from top K candidates
            # If num_uncovered is small, gains will be small.
            # Find indices of top K gains
            # Sort gains descending
            # To avoid sorting all 75k every time (which is ~75k log 75k), 
            # we can use argpartition or just find max. 
            # However, randomizing from top candidates helps escape local minima.
            # Let's try picking from top 10-50 candidates.
            
            # Get top K indices
            K = 20
            if n_k_blocks > K:
                # argpartition returns indices of the K largest elements (not sorted)
                # We need to exclude used blocks? 
                # Ideally, we shouldn't reuse blocks. 
                # But in a covering, reusing a block is useless. 
                # We can mask out used blocks by setting their gains to -1 or 0.
                # Since M is sparse, modifying it is expensive. 
                # Instead, we can just check if the best candidate is used.
                # Or, we can maintain a list of available blocks? 
                # With 75k blocks, a list is fine.
                
                # For simplicity and speed, let's just pick the global max.
                # If it's used, pick the next max.
                # This is effectively greedy. 
                # To add randomness, we can pick from top K *unused* blocks.
                
                # Let's get the indices of the top K gains
                top_k_indices = np.argpartition(gains, -K)[-K:]
                
                # Filter out used blocks
                valid_candidates = [i for i in top_k_indices if i not in used_blocks]
                
                if not valid_candidates:
                    # Fallback: find any unused block with max gain
                    # This is rare if K is large enough and we haven't used many
                    gains_copy = gains.copy()
                    for i in used_blocks:
                        gains_copy[i] = -1
                    best_idx = np.argmax(gains_copy)
                    if gains_copy[best_idx] <= 0:
                         # No gain, break? Or just pick random?
                         # If no gain, we are stuck. But num_uncovered > 0 means we need more blocks.
                         # If max gain is 0, it means no block covers any uncovered subset?
                         # That implies the problem is infeasible or we have a bug.
                         # Given C(19,8,5) is feasible, this shouldn't happen unless we ran out of blocks.
                         break
                    valid_candidates = [best_idx]
                
                # Pick randomly from valid candidates
                best_idx = random.choice(valid_candidates)
            else:
                # If few blocks, just pick max unused
                gains_copy = gains.copy()
                for i in used_blocks:
                    gains_copy[i] = -1
                best_idx = np.argmax(gains_copy)
                if gains_copy[best_idx] <= 0:
                    break

            # Update state
            blocks_selected.append(best_idx)
            used_blocks.add(best_idx)
            
            # Update uncovered mask
            # Get the t-subsets covered by this block
            block_idxs = block_t_indices[best_idx]
            if block_idxs:
                # Set uncovered[block_idxs] = False
                # Count how many were actually uncovered
                # np.sum(uncovered[block_idxs]) gives the gain
                gain = np.sum(uncovered[block_idxs])
                num_uncovered -= gain
                uncovered[block_idxs] = False
                
            # Time check
            if time.time() > deadline - 5.0:
                break
        
        # Convert block indices to actual blocks
        return [all_k_blocks[i] for i in blocks_selected]

    def reduce_covering(blocks):
        """
        Remove redundant blocks using coverage counts.
        """
        if not blocks:
            return blocks
        
        # Map block value to index in the provided list for quick lookup?
        # Or just use the precomputed block_t_indices if the blocks are from all_k_blocks.
        # The blocks from greedy_construct are indices into all_k_blocks.
        # Let's assume input blocks are lists of points.
        # We need to map them back to indices or recompute.
        # Since greedy_construct returns indices, let's change it to return indices, 
        # and convert at the end. Or just keep them as indices in the reduction step.
        
        # Let's modify greedy_construct to return indices, and construct to return blocks.
        # Wait, the current greedy_construct returns blocks. 
        # Let's change it to return indices for efficiency in reduction.
        
        # Actually, let's just do reduction on the indices.
        pass

    def reduce_covering_indices(block_indices):
        """
        Remove redundant blocks given as indices into all_k_blocks.
        """
        if not block_indices:
            return []
        
        # Coverage count for each t-subset
        cov = np.zeros(n_t_subs, dtype=np.int32)
        
        # Initialize coverage
        for b_idx in block_indices:
            for idx in block_t_indices[b_idx]:
                cov[idx] += 1
        
        removed = np.zeros(len(block_indices), dtype=bool)
        changed = True
        
        while changed:
            changed = False
            for i in range(len(block_indices)):
                if removed[i]:
                    continue
                b_idx = block_indices[i]
                idxs = block_t_indices[b_idx]
                
                if not idxs:
                    removed[i] = True
                    changed = True
                    continue
                    
                is_essential = False
                for idx in idxs:
                    if cov[idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    removed[i] = True
                    changed = True
                    for idx in idxs:
                        cov[idx] -= 1
        
        result = [block_indices[i] for i in range(len(block_indices)) if not removed[i]]
        return result

    best_blocks = None
    best_size = float('inf')
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        seed = attempt * 12345 + 67890
        
        # Get indices of blocks from greedy
        block_indices = greedy_construct(seed)
        
        if not block_indices:
            continue
        
        # Reduce
        reduced_indices = reduce_covering_indices(block_indices)
        
        if len(reduced_indices) < best_size:
            best_size = len(reduced_indices)
            best_blocks = reduced_indices
            
        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        # Fallback
        block_indices = greedy_construct(None)
        reduced_indices = reduce_covering_indices(block_indices)
        best_blocks = reduced_indices
        
    # Convert indices back to blocks
    return [all_k_blocks[i] for i in best_blocks]
