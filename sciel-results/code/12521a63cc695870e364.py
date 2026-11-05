import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Precomputation
    # All t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # All k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute incidence matrix: M[i, j] = 1 if block i covers subset j
    # M shape: (n_blocks, n_subsets)
    # Using sparse-like logic or direct numpy array since 50k * 3876 is ~200M bits = 25MB, which is fine.
    # However, creating it directly might be slow. Let's create a list of lists of indices first, then maybe convert if needed.
    # Actually, for vectorized gain, we need to know which subsets are covered by a block.
    # Let's store block_covers as a list of arrays of indices for fast boolean masking on counts.
    
    block_covers = []
    for block in all_blocks:
        covered_indices = np.array([subset_to_idx[sub] for sub in itertools.combinations(block, t)], dtype=np.int32)
        block_covers.append(covered_indices)
        
    # 2. Greedy Initialization
    # cover_counts[j] is the number of blocks currently covering subset j
    cover_counts = np.zeros(n_subsets, dtype=np.int32)
    current_blocks = set()
    
    # Greedy: pick block covering most uncovered subsets
    # We maintain a boolean array 'uncovered' for speed? 
    # Gain for block i = sum(cover_counts[block_covers[i]] == 0)
    
    # To speed up greedy, we can just scan all blocks. 
    # 50k blocks * (avg 15 subsets per block) checks. 
    # If we do this 150 times, it's 50k * 15 * 150 = 112.5M ops. 
    # In Python this might be tight. Let's try to optimize the gain calculation.
    
    # Alternative: Precompute a matrix? 
    # Let's stick to the list of indices but use numpy sum for the gain.
    
    for _ in range(300): # Max iterations
        if time.time() > deadline:
            break
        
        # Check if done
        if np.all(cover_counts > 0):
            break
            
        best_block = -1
        best_gain = 0
        
        # Iterate through blocks not in current set
        # Optimization: We can skip blocks if we know they can't beat best_gain? Hard to do easily.
        # Let's just iterate.
        
        # Vectorized approach for gain calculation is hard without a full matrix because 
        # we need to sum over variable length arrays.
        # However, we can update a 'potential' score? No, counts change.
        
        # Let's try a simpler optimization: 
        # Randomly sample a subset of blocks if n_blocks is large? No, 50k is manageable if inner loop is C-speed (numpy).
        
        for i in range(n_blocks):
            if i in current_blocks:
                continue
            
            # Get indices of subsets covered by block i
            idxs = block_covers[i]
            # Count how many of these are currently uncovered (count == 0)
            gain = np.sum(cover_counts[idxs] == 0)
            
            if gain > best_gain:
                best_gain = gain
                best_block = i
                
        if best_gain == 0:
            break # All covered or no block helps (should be all covered)
            
        current_blocks.add(best_block)
        # Update counts
        idxs = block_covers[best_block]
        cover_counts[idxs] += 1

    best_size = len(current_blocks)
    best_blocks = set(current_blocks)
    
    # 3. Local Search
    random.seed(42)
    
    # Precompute a reverse index: subset_idx -> list of blocks covering it
    # This helps in repair/swap to find candidates quickly? 
    # Actually, for "repair", we need to find a block that covers a specific set of subsets.
    # Let's stick to scanning all blocks for the best repair candidate, but make it fast.
    
    while time.time() < deadline:
        # Try removal
        removed = False
        # Shuffle blocks to try different ones
        blocks_list = list(current_blocks)
        random.shuffle(blocks_list)
        
        for b in blocks_list:
            if time.time() > deadline:
                break
                
            # Check if b is redundant
            # Redundant if all subsets covered by b have count > 1
            idxs_b = block_covers[b]
            if np.all(cover_counts[idxs_b] > 1):
                # Remove b
                cover_counts[idxs_b] -= 1
                current_blocks.remove(b)
                removed = True
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = set(current_blocks)
                break # Restart removal search after a change
        
        if not removed:
            # Try swap/repair
            # Pick a random block to remove
            if not current_blocks:
                break
                
            b1 = random.choice(list(current_blocks))
            idxs_b1 = block_covers[b1]
            
            # Identify subsets that will become uncovered if b1 is removed
            # These are subsets where cover_counts[j] == 1 and j is in idxs_b1
            # We need to find a block b2 that covers ALL of these.
            
            # Tentatively remove b1 to see which ones become 0? 
            # No, just look at current counts.
            # Uncovered_if_removed = {j in idxs_b1 | cover_counts[j] == 1}
            
            # If no subsets become uncovered, b1 is redundant (should have been caught above, but maybe race condition? No, we break on removal).
            # So there is at least one such subset.
            
            # To find b2, we need a block that covers all these specific subsets.
            # Let's get the list of critical subsets
            critical_mask = (cover_counts == 1)
            # Intersect with idxs_b1
            # critical_subsets = idxs_b1[cover_counts[idxs_b1] == 1]
            
            critical_counts = cover_counts[idxs_b1]
            critical_idxs = idxs_b1[critical_counts == 1]
            
            if len(critical_idxs) == 0:
                # This shouldn't happen if removal failed, but just in case
                continue
                
            # Find best b2 that covers all critical_idxs
            # We can iterate through all blocks not in current_blocks
            # and check if critical_idxs is a subset of block_covers[b2]
            
            best_b2 = -1
            # Optimization: If critical_idxs is large, few blocks will cover it.
            # We can check the first critical subset to filter candidates?
            # Let's just scan. 50k blocks.
            # For each block, check if it covers all critical_idxs.
            
            # To speed this up, we can precompute for each subset which blocks cover it.
            # But building that is O(n_blocks * t_subsets_per_block).
            # Let's try a heuristic: 
            # Pick one random critical subset, say c_idx.
            # Only consider blocks that cover c_idx.
            
            # Let's build a quick index for this specific critical subset to narrow down candidates.
            # Actually, we can just check `np.all(np.isin(critical_idxs, block_covers[b2]))`? 
            # That's slow.
            
            # Better: 
            # For the first critical subset c0, get all blocks covering c0.
            # How to get that fast? 
            # We didn't precompute reverse index. 
            # Let's just scan all blocks, but break early if a block doesn't cover the first critical subset.
            
            c0 = critical_idxs[0]
            
            for b2 in range(n_blocks):
                if b2 in current_blocks:
                    continue
                    
                # Check if b2 covers c0
                # We can do this by checking if c0 is in block_covers[b2]
                # Using np.isin or binary search? block_covers[b2] is sorted?
                # itertools.combinations generates sorted tuples, and we mapped them to indices.
                # The indices in block_covers are not necessarily sorted by index value, 
                # but the subsets themselves are sorted. The mapping subset_to_idx is arbitrary.
                # So block_covers[b2] is an array of arbitrary indices.
                
                # Fast check: 
                # We can precompute a set for each block? No, memory.
                # Let's use numpy searchsorted if we sort the indices? 
                # Or just use `c0 in block_covers[b2]`? For small arrays (size ~15), this is fast in C.
                
                if c0 not in block_covers[b2]:
                    continue
                    
                # If it covers c0, check if it covers ALL critical_idxs
                # critical_idxs is small (usually 1-3).
                # Check if all elements of critical_idxs are in block_covers[b2]
                # We can do: len(set(critical_idxs).intersection(block_covers[b2])) == len(critical_idxs)
                # Or: np.all(np.isin(critical_idxs, block_covers[b2]))
                
                if np.all(np.isin(critical_idxs, block_covers[b2])):
                    best_b2 = b2
                    break # Found a valid swap, take the first one? 
                    # Or find the best one? 
                    # "Best" could mean covers the most *other* subsets to increase robustness?
                    # For now, any valid swap reduces size or maintains it. 
                    # Since we are in "not removed" phase, we assume b1 is not redundant.
                    # So swapping b1 with b2 keeps size same. 
                    # Wait, the goal is to MINIMIZE size. 
                    # If we can't remove, we can't reduce size with a 1-for-1 swap.
                    # So why do swaps? 
                    # Swaps can help escape local minima for future removals.
                    # So we should accept a swap if it increases "potential" for removal.
                    # Heuristic: Accept if b2 covers more subsets than b1? Or just random valid swap.
                    # Let's just accept the first valid swap to change the landscape.
                    
            if best_b2 != -1:
                # Perform swap
                cover_counts[idxs_b1] -= 1
                current_blocks.remove(b1)
                
                idxs_b2 = block_covers[best_b2]
                cover_counts[idxs_b2] += 1
                current_blocks.add(best_b2)
                
                # Update best if size decreased? No, size is same.
                # But maybe this state allows removals later.
                
                # We don't update best_size here because it didn't change.
                # But we continue the loop to try removals on the new state.

    # Convert best_blocks to list of lists
    return [list(all_blocks[i]) for i in sorted(best_blocks)]
