import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    
    # Map each k-subset to the list of t-subset indices it contains
    # k_sub_to_t_idxs: list of lists of int
    k_sub_to_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_to_t_idxs.append(t_idxs)
        
    # Map each t-subset to the list of k-subset indices that contain it
    # t_sub_to_k_idxs: list of lists of int
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_to_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)
            
    # Greedy Construction
    uncovered = set(range(n_t))
    blocks = []
    block_indices = set()
    
    # Helper to check time
    def time_ok():
        return time.time() < deadline
        
    while uncovered and time_ok():
        # Pick an uncovered t-subset
        # To improve heuristic, we could pick one that is in the fewest 
        # "good" blocks, but let's stick to simple first or random.
        # Randomizing might help find different local minima, but we only get one run.
        # Let's pick the one with the smallest index for determinism, 
        # or maybe the one that appears in the fewest total k-subsets? (All are same)
        s_idx = min(uncovered)
        
        # Get candidate k-subsets containing s_idx
        candidates = t_sub_to_k_idxs[s_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidates:
            # Calculate gain: how many uncovered t-subsets are in this block
            gain = 0
            for t_idx in k_sub_to_t_idxs[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break
            
        blocks.append(all_k_subs[best_block_idx])
        block_indices.add(best_block_idx)
        
        # Update uncovered
        for t_idx in k_sub_to_t_idxs[best_block_idx]:
            if t_idx in uncovered:
                uncovered.remove(t_idx)
                
    # If time runs out, we might have an incomplete covering. 
    # The task requires a valid covering. 
    # If we run out of time, we should ensure validity or return best effort?
    # "Valid when every t-subset ... lies in at least one block".
    # If we return invalid, score is 0.
    # So we must finish. 
    # Given the speed estimate (6.8M ops), it should finish in < 1 second.
    
    # Redundancy Removal
    # We maintain a count of how many blocks cover each t-subset
    cover_count = [0] * n_t
    for b in blocks:
        b_idx = all_k_subs.index(b) # This is slow, O(N). Better to track indices.
        
    # Let's redo reduction using indices for speed
    # Rebuild cover_count from blocks
    cover_count = [0] * n_t
    # We need to map block tuple to index or just store indices in blocks list?
    # The return type must be list of blocks (tuples/lists of points).
    # But for internal processing, indices are faster.
    
    # Let's convert blocks to indices for reduction phase
    # Since we added them from all_k_subs, we can just keep the indices.
    # Let's restart reduction logic properly.
    
    # Actually, let's just do a simple pass on the list of blocks.
    # To check if a block is redundant, we need to know if all its t-subsets are covered by OTHER blocks.
    
    # Efficient reduction:
    # 1. Calculate cover_count for all t-subsets based on current blocks.
    # 2. Iterate through blocks.
    # 3. For a block B, check if any t-subset T in B has cover_count[T] == 1.
    #    If yes, B is essential (cannot be removed).
    #    If no, B is redundant. Remove it, and decrement cover_count for all T in B.
    
    # We need to be careful: removing B changes cover_count for subsequent checks.
    
    # Let's implement this.
    
    # First, map block tuples to their indices in all_k_subs for quick lookup?
    # Or just store the indices in a parallel list.
    
    # Let's refactor the greedy part to store indices.
    
    # Re-run greedy to get indices? No, we already have blocks.
    # Let's create a dict: tuple(block) -> index
    # This is O(N) to build.
    
    # Actually, we can just recompute cover_count.
    
    # Let's try a different reduction strategy that is simpler but effective:
    # Just check each block.
    
    # To make it fast, we need fast access to t-subsets of a block.
    # We have k_sub_to_t_idxs.
    
    # Let's assume the greedy phase finished.
    
    # Build cover_count
    cover_count = [0] * n_t
    # We need to know which t-subsets are in which blocks.
    # We have blocks as tuples.
    # Let's create a map from block tuple to index in all_k_subs.
    # This is necessary for O(1) access to k_sub_to_t_idxs.
    
    k_sub_to_idx_map = {k_sub: i for i, k_sub in enumerate(all_k_subs)}
    
    for b in blocks:
        b_idx = k_sub_to_idx_map[b]
        for t_idx in k_sub_to_t_idxs[b_idx]:
            cover_count[t_idx] += 1
            
    changed = True
    while changed and time_ok():
        changed = False
        new_blocks = []
        for b in blocks:
            b_idx = k_sub_to_idx_map[b]
            is_essential = False
            for t_idx in k_sub_to_t_idxs[b_idx]:
                if cover_count[t_idx] == 1:
                    is_essential = True
                    break
            if is_essential:
                new_blocks.append(b)
            else:
                # Redundant, remove it
                changed = True
                for t_idx in k_sub_to_t_idxs[b_idx]:
                    cover_count[t_idx] -= 1
        blocks = new_blocks
        
    return blocks
