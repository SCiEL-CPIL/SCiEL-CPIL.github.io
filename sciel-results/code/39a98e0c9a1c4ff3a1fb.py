import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_subsets_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Precompute all k-subsets and their t-subset index lists
    # C(19,8) = 75582 blocks, each has C(8,5)=56 t-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each k-subset to its t-subset indices
    k_to_t_indices = {}
    for ks in all_k_subsets:
        k_to_t_indices[ks] = [t_subsets_idx[ts] for ts in itertools.combinations(ks, t)]
    
    # For greedy: map each t-subset to list of k-subsets containing it
    t_to_k_subsets = [[] for _ in range(num_t_subsets)]
    for ks, t_idx_list in k_to_t_indices.items():
        for idx in t_idx_list:
            t_to_k_subsets[idx].append(ks)
    
    # Greedy construction
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    while uncovered:
        # Pick random uncovered t-subset
        s_idx = random.choice(list(uncovered))
        s = t_subsets[s_idx]
        
        # Get all k-subsets containing s
        candidates = t_to_k_subsets[s_idx]
        
        best_block = None
        best_gain = -1
        
        for block in candidates:
            # Count uncovered t-subsets in this block
            gain = sum(1 for idx in k_to_t_indices[block] if idx in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        blocks.append(best_block)
        
        # Update uncovered
        for idx in k_to_t_indices[best_block]:
            uncovered.discard(idx)
    
    # Optimization phase
    ref_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b in blocks:
        for idx in k_to_t_indices[b]:
            ref_count[idx] += 1
    
    current_size = len(blocks)
    best_size = current_size
    best_blocks = list(blocks)
    
    block_indices = {b: k_to_t_indices[b] for b in blocks}
    block_set = set(blocks)
    
    deadline = time.time() + 115  # Safety margin
    
    # Aggressive deletion pass
    def deletion_pass(blocks, ref_count, block_indices, block_set):
        changed = False
        i = 0
        while i < len(blocks):
            b = blocks[i]
            indices = block_indices[b]
            
            # Tentatively remove
            for idx in indices:
                ref_count[idx] -= 1
            
            if np.min(ref_count) > 0:
                # Redundant
                blocks.pop(i)
                del block_indices[b]
                block_set.discard(b)
                changed = True
            else:
                # Restore
                for idx in indices:
                    ref_count[idx] += 1
                i += 1
        
        return changed
    
    # Swap strategy: try to replace a block with one that enables deletion
    def try_swap(blocks, ref_count, block_indices, block_set):
        if len(blocks) == 0:
            return False
        
        # Pick random block
        i = random.randint(0, len(blocks) - 1)
        b_old = blocks[i]
        old_indices = block_indices[b_old]
        
        # Find critical t-subsets (count == 1 after removing b_old)
        criticals = []
        for idx in old_indices:
            if ref_count[idx] == 1:
                criticals.append(idx)
        
        if not criticals:
            return False
        
        # Union of points in critical t-subsets
        u_set = set()
        for idx in criticals:
            u_set.update(t_subsets[idx])
        
        if len(u_set) > k:
            return False
        
        # Generate candidate blocks containing u_set
        remaining_points = [p for p in points if p not in u_set]
        if len(remaining_points) < k - len(u_set):
            return False
        
        # Try multiple random completions
        for _ in range(20):
            if len(remaining_points) >= k - len(u_set):
                extra = random.sample(remaining_points, k - len(u_set))
                b_new = tuple(sorted(list(u_set) + extra))
            else:
                break
            
            if b_new == b_old:
                continue
            
            # Check if b_new is already in blocks (would be no-op)
            if b_new in block_set:
                continue
            
            new_indices = k_to_t_indices[b_new]
            
            # Check validity
            for idx in old_indices:
                ref_count[idx] -= 1
            for idx in new_indices:
                ref_count[idx] += 1
            
            if np.min(ref_count) > 0:
                # Valid swap - perform it
                blocks[i] = b_new
                block_indices[b_new] = new_indices
                del block_indices[b_old]
                block_set.discard(b_old)
                block_set.add(b_new)
                return True
            else:
                # Revert
                for idx in new_indices:
                    ref_count[idx] -= 1
                for idx in old_indices:
                    ref_count[idx] += 1
        
        return False
    
    # Main optimization loop
    while time.time() < deadline:
        # Deletion pass
        changed = deletion_pass(blocks, ref_count, block_indices, block_set)
        
        if current_size < best_size:
            best_size = current_size
            best_blocks = list(blocks)
        
        # Try swaps
        num_swap_attempts = 50
        for _ in range(num_swap_attempts):
            if time.time() > deadline:
                break
            try_swap(blocks, ref_count, block_indices, block_set)
        
        # Another deletion pass after swaps
        changed = deletion_pass(blocks, ref_count, block_indices, block_set)
        
        if current_size < best_size:
            best_size = current_size
            best_blocks = list(blocks)
    
    return best_blocks
