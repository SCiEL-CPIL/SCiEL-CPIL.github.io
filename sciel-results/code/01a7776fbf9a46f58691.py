import itertools
import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    deadline = time.time() + 145.0
    
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets and their t-subset memberships
    all_ksubs = list(itertools.combinations(range(v), k))
    num_ksubs = len(all_ksubs)
    
    # block_to_tsubs[b] = list of t-subset indices contained in block b
    block_to_tsubs = []
    for b in all_ksubs:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(tsubs_in_b)
    
    # tsub_to_blocks[t] = list of block indices that contain t-subset t
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsub_list in enumerate(block_to_tsubs):
        for ts_idx in tsub_list:
            tsub_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    # For each t-subset, store the list of blocks as a numpy array
    tsub_blocks_np = []
    for i in range(num_tsubs):
        tsub_blocks_np.append(np.array(tsub_to_blocks[i], dtype=np.int32))
    
    def greedy_construction(seed):
        """
        Greedy construction using bitset-like operations with numpy.
        Returns a list of block indices.
        """
        rng = random.Random(seed)
        
        # covered[i] = True if t-subset i is covered
        covered = np.zeros(num_tsubs, dtype=bool)
        selected_blocks = []
        
        # Precompute for each block: the t-subsets it covers as a boolean mask
        # This would be memory intensive, so we'll compute gains on the fly
        
        while not covered.all():
            if time.time() > deadline - 2.0:
                return None
            
            # Find an uncovered t-subset to anchor
            # Use random probing for speed
            while True:
                idx = rng.randint(0, num_tsubs - 1)
                if not covered[idx]:
                    break
            
            anchor_tsub_idx = idx
            # Get all blocks containing this t-subset
            candidate_blocks = tsub_blocks_np[anchor_tsub_idx]
            
            if len(candidate_blocks) == 0:
                return None
            
            # For each candidate block, compute the gain (number of new t-subsets covered)
            # We want to pick the block with maximum gain
            best_block = None
            best_gain = -1
            
            # Shuffle candidates to break ties randomly
            shuffled = candidate_blocks.copy()
            rng.shuffle(shuffled)
            
            for b_idx in shuffled:
                tsub_list = block_to_tsubs[b_idx]
                gain = 0
                for ts_idx in tsub_list:
                    if not covered[ts_idx]:
                        gain += 1
                        # Early termination if we can't beat best
                        # max possible gain is len(tsub_list), but we can break if
                        # even completing all remaining won't help... not quite.
                        # Actually, we can't easily early terminate without knowing total.
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    # If gain is maximum possible (all tsubs in block uncovered), we can break
                    # Maximum possible for a block is C(k,t) = 56
                    if best_gain >= len(tsub_list):
                        break
            
            if best_block is None or best_gain <= 0:
                # Should not happen if there's an uncovered t-subset
                # But just in case, pick randomly
                best_block = shuffled[0]
            
            selected_blocks.append(best_block)
            
            # Update covered
            for ts_idx in block_to_tsubs[best_block]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
        
        return selected_blocks
    
    def reduce_blocks(block_indices):
        """
        Remove redundant blocks. A block is redundant if every t-subset it covers
        is also covered by at least one other selected block.
        """
        if not block_indices:
            return block_indices
        
        selected = set(block_indices)
        
        # Count how many selected blocks cover each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in selected:
            for ts_idx in block_to_tsubs[b_idx]:
                cover_count[ts_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_selected = set()
            
            for b_idx in list(selected):
                # Check if this block is essential
                is_essential = False
                for ts_idx in block_to_tsubs[b_idx]:
                    if cover_count[ts_idx] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_selected.add(b_idx)
                else:
                    # Remove this block
                    changed = True
                    for ts_idx in block_to_tsubs[b_idx]:
                        cover_count[ts_idx] -= 1
            
            selected = new_selected
        
        return sorted(selected)
    
    def local_search(block_indices, seed, time_limit=5.0):
        """
        Try to improve the solution by removing blocks or swapping.
        Returns improved list of block indices.
        """
        rng = random.Random(seed)
        current = list(block_indices)
        current_set = set(current)
        
        # Compute cover counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsubs[b_idx]:
                cover_count[ts_idx] += 1
        
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            rng.shuffle(current)
            
            for i, b_idx in enumerate(current):
                if time.time() - start_time > time_limit:
                    break
                
                # Check if we can remove b_idx
                can_remove = True
                for ts_idx in block_to_tsubs[b_idx]:
                    if cover_count[ts_idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for ts_idx in block_to_tsubs[b_idx]:
                        cover_count[ts_idx] -= 1
                    current.pop(i)
                    current_set.discard(b_idx)
                    improved = True
                    break
            
            if not improved:
                # Try to swap: remove one block and add another
                for i in range(len(current)):
                    if time.time() - start_time > time_limit:
                        break
                    
                    b_remove = current[i]
                    
                    # Find t-subsets only covered by b_remove
                    exclusive_tsubs = []
                    for ts_idx in block_to_tsubs[b_remove]:
                        if cover_count[ts_idx] == 1:
                            exclusive_tsubs.append(ts_idx)
                    
                    if not exclusive_tsubs:
                        # We can remove and don't need to add
                        for ts_idx in block_to_tsubs[b_remove]:
                            cover_count[ts_idx] -= 1
                        current.pop(i)
                        current_set.discard(b_remove)
                        improved = True
                        break
                    else:
                        # Find a block that covers all exclusive tsubs and is not in current
                        # This is expensive, so do a limited search
                        # For each exclusive tsub, find blocks that cover it
                        # Intersect these sets
                        candidate_add = None
                        for ts_idx in exclusive_tsubs[:3]:  # Limit to first 3 for speed
                            if candidate_add is None:
                                candidate_add = set(tsub_to_blocks[ts_idx])
                            else:
                                candidate_add &= set(tsub_to_blocks[ts_idx])
                        
                        if candidate_add:
                            # Remove blocks already in current
                            candidate_add -= current_set
                            candidate_add -= {b_remove}
                            
                            if candidate_add:
                                # Pick one randomly
                                b_add = rng.choice(sorted(candidate_add))
                                
                                # Check if adding b_add covers all exclusive tsubs
                                all_covered = True
                                for ts_idx in exclusive_tsubs:
                                    if ts_idx not in block_to_tsubs[b_add]:
                                        all_covered = False
                                        break
                                
                                if all_covered:
                                    # Perform swap
                                    for ts_idx in block_to_tsubs[b_remove]:
                                        cover_count[ts_idx] -= 1
                                    current.pop(i)
                                    current_set.discard(b_remove)
                                    
                                    for ts_idx in block_to_tsubs[b_add]:
                                        cover_count[ts_idx] += 1
                                    current.append(b_add)
                                    current_set.add(b_add)
                                    improved = True
                                    break
        
        return current
    
    # Main: multiple restarts
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy construction
        block_indices = greedy_construction(restart)
        
        if block_indices is None:
            continue
        
        # Reduction
        reduced = reduce_blocks(block_indices)
        
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced[:]
        
        # Local search on the reduced set
        if time.time() < deadline - 10.0:
            ls_result = local_search(reduced, restart, time_limit=3.0)
            if len(ls_result) < best_len:
                best_len = len(ls_result)
                best_blocks = ls_result[:]
        
        # Early exit if we hit a very good solution
        if best_len <= 330:
            break
    
    if best_blocks is None:
        # Fallback
        block_indices = greedy_construction(0)
        if block_indices is not None:
            best_blocks = reduce_blocks(block_indices)
        else:
            # Should not happen, but return something
            return []
    
    # Convert block indices back to actual blocks
    return [list(all_ksubs[i]) for i in sorted(best_blocks)]

# EVOLVE-BLOCK-END
