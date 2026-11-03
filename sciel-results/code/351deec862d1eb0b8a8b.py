import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build sparse matrix A of size n_k x n_t
    # A[i, j] = 1 if t_subs[j] is subset of k_subs[i]
    row_indices = []
    col_indices = []
    data = []
    
    # For each block, find which t-subsets it contains
    # Pre-compute the list of t-subset indices for each block for faster access later
    block_t_subs = []
    
    for i, block in enumerate(k_subs):
        # Generate t-subsets of this block
        block_t_sub_indices = []
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                block_t_sub_indices.append(j)
        block_t_subs.append(block_t_sub_indices)
                
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)
    
    def greedy_cover(random_seed=None):
        """Greedy Set Cover with optional random tie-breaking."""
        u = np.ones(n_t, dtype=np.int8)
        selected_blocks = []
        selected_set = set()
        
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        # Pre-compute block cols as numpy arrays for faster u[cols] = 0
        # We already have block_t_subs as lists, let's convert to arrays if needed, 
        # but lists are fine for small sizes (56).
        
        while u.sum() > 0:
            if time.time() > deadline:
                break
                
            # Calculate gain for each block
            gains = A @ u
            best_gain = np.max(gains)
            
            if best_gain == 0:
                # Should not happen if covering exists
                break
            
            # Find all blocks with max gain
            # To add randomness, pick randomly among max gain blocks
            candidates = np.where(gains == best_gain)[0]
            
            if len(candidates) > 1:
                best_idx = candidates[rng.randint(0, len(candidates)-1)]
            else:
                best_idx = candidates[0]
            
            if best_idx in selected_set:
                # This shouldn't happen if we update u correctly, but just in case
                # If we pick a block already selected, it would have gain 0 because 
                # its t-subsets are already covered (u[cols]=0).
                # So this check is mostly for safety.
                continue
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered vector
            cols = block_t_subs[best_idx]
            if cols:
                u[cols] = 0
                
        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove redundant blocks from the covering."""
        if not selected_blocks:
            return selected_blocks
            
        # Pre-compute lists of t-subset indices for each selected block
        selected_t_subs = []
        for idx in selected_blocks:
            cols = block_t_subs[idx]
            selected_t_subs.append(cols)
            
        # Initialize cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            if cols:
                cover_count[cols] += 1
            
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                # Check if any col in cols has cover_count == 1
                is_essential = False
                if cols:
                    for c in cols:
                        if cover_count[c] == 1:
                            is_essential = True
                            break
                    
                if not is_essential:
                    to_remove.append(i)
                    
            if to_remove:
                changed = True
                # Remove blocks from the end to maintain indices
                for i in sorted(to_remove, reverse=True):
                    # Update cover_count
                    cols = selected_t_subs[i]
                    if cols:
                        cover_count[cols] -= 1
                    # Remove from lists
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)
                    
        return selected_blocks

    def local_search(selected_blocks):
        """
        Local search: Try to swap blocks or remove blocks to reduce size.
        1. Try to remove a block if it's redundant (already handled by remove_redundant, 
           but we can do it again after swaps).
        2. Try to swap a block B_i with B_j if B_j covers all t-subsets that B_i uniquely covers.
           This maintains coverage but might allow further reductions.
           Actually, a better heuristic for minimizing count:
           Try to remove B_i. If coverage breaks, find a B_j that covers the broken parts.
           If such B_j exists and B_j is not in the set, swap.
           This doesn't reduce count immediately, but it's a move in the search space.
           
           However, a more direct "reduction" heuristic is:
           Check if there is a block B_i that can be removed, and the resulting uncovered sets
           can be covered by adding *fewer* than 1 block? No, we add 1 block.
           So swap doesn't reduce count.
           
           What reduces count is finding that a block is redundant.
           The greedy + redundancy removal is strong.
           
           Let's try a "2-swap" or "add-remove" heuristic:
           1. Add a random block B_new.
           2. Check if any existing block B_old becomes redundant.
           3. If yes, remove B_old. Net change: 0 or -1? 
              If we add B_new, count increases by 1. If we remove B_old, count decreases by 1. Net 0.
              This is a hill-climbing move to escape local optima.
              
           Let's implement a simple 1-swap local search that aims to find a configuration
           where more blocks are redundant.
        """
        if not selected_blocks:
            return selected_blocks
            
        selected_blocks = list(selected_blocks)
        selected_t_subs = [block_t_subs[idx] for idx in selected_blocks]
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            if cols:
                cover_count[cols] += 1
                
        improved = True
        max_iters = 50 # Limit local search iterations to avoid timeout
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # Try to remove each block
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                if not cols:
                    # Empty block? Unlikely, but if so, remove it
                    cover_count[cols] -= 1
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)
                    improved = True
                    break # Restart loop after modification
                    
                # Check if essential
                is_essential = False
                for c in cols:
                    if cover_count[c] == 1:
                        is_essential = True
                        break
                        
                if not is_essential:
                    # Redundant, remove it
                    cover_count[cols] -= 1
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)
                    improved = True
                    break # Restart loop after modification
            else:
                # No redundant blocks found in this pass.
                # Try a swap heuristic?
                # For now, just break if no redundancy found.
                # A more advanced search would try swaps here.
                break
                
        return selected_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy passes with different random seeds
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random seed for this restart
        seed = r * 1000 + 42
        blocks = greedy_cover(random_seed=seed)
        
        if not blocks:
            continue
            
        # Remove redundancies
        blocks = remove_redundant(blocks)
        
        # Local search for further redundancy
        blocks = local_search(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        # If we found a very good solution, we might break early?
        # The theoretical lower bound is 264. If we get close, maybe.
        # But let's use the time.

    # Final redundancy removal on the best block
    if best_blocks:
        best_blocks = remove_redundant(best_blocks)
        best_blocks = local_search(best_blocks)
        
    # Convert indices back to blocks
    if best_blocks:
        final_blocks = [list(k_subs[i]) for i in best_blocks]
    else:
        # Fallback: if no blocks found (should not happen), return empty
        final_blocks = []
        
    return final_blocks
