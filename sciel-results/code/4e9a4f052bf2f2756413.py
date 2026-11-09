import time
import sys
from itertools import combinations

# Set recursion limit just in case, though depth is small (max 132)
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 12
    # All 5-subsets of 12 points
    all_5_subsets = list(combinations(range(n), 5))
    num_5_subsets = len(all_5_subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # All 6-subsets of 12 points
    all_6_subsets = list(combinations(range(n), 6))
    
    # For each 6-subset, find which 5-subsets it covers
    # block_covered_bits[i] will be a bitmask representing the 5-subsets covered by block i
    block_covered_bits = []
    # For each 5-subset index, list of block indices that cover it
    # We will build this as a list of lists of block indices
    covers = [[] for _ in range(num_5_subsets)]
    
    for b_idx, block in enumerate(all_6_subsets):
        mask = 0
        # A 6-subset contains 6 five-subsets
        for i in range(6):
            # Remove the i-th element from the 6-tuple to form a 5-subset
            # block is a tuple of 6 sorted elements
            sub = block[:i] + block[i+1:]
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
            covers[idx].append(b_idx)
        block_covered_bits.append(mask)
        
    # We want to find a set of blocks that covers all 5-subsets exactly once.
    # This is an Exact Cover problem.
    # State: 
    #   - uncovered_mask: bitmask of 5-subsets not yet covered
    #   - available_blocks: set/list of block indices still available
    #   - solution: list of chosen blocks
    
    # Optimization: Pre-sort blocks or use a heuristic.
    # Algorithm X:
    # 1. Pick a 5-subset (column) with the fewest available blocks covering it.
    # 2. Try each such block.
    # 3. If a block is chosen, remove all 5-subsets it covers from 'uncovered' 
    #    and remove all blocks that conflict with it (i.e., share any 5-subset) from 'available'.
    
    # To make "fewest available blocks" fast, we can just iterate through uncovered bits
    # and check the length of the intersection of covers[idx] with available_blocks.
    
    # Initial state
    initial_uncovered = (1 << num_5_subsets) - 1
    initial_available = set(range(len(all_6_subsets)))
    
    solution = []
    
    def backtrack(uncovered, available, current_sol):
        if time.time() > deadline:
            return False
        
        if uncovered == 0:
            current_sol.extend(available) # Wait, no. If uncovered is 0, we are done.
            # Actually, we just need to return True. The solution is in current_sol.
            return True
        
        # Find the 5-subset with the minimum number of covering blocks in 'available'
        # We iterate through the set bits in 'uncovered'
        # To speed up, we can just check the first few or find the min properly.
        # Given the structure, the min count is usually small (around 6 or so initially, decreasing).
        
        min_count = float('inf')
        best_col = -1
        best_covers = []
        
        # Iterate through uncovered bits
        temp = uncovered
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            temp ^= lsb
            
            # Count how many blocks in 'available' cover this idx
            # covers[idx] is a list of block indices
            count = 0
            # Since available is a set, checking membership is O(1)
            # But iterating the list and checking membership might be slow if list is large.
            # Initial covers[idx] size is 6 (each 5-subset is in 6 blocks).
            # So this loop is very fast.
            for b_idx in covers[idx]:
                if b_idx in available:
                    count += 1
                    if count > min_count:
                        break
            
            if count < min_count:
                min_count = count
                best_col = idx
                best_covers = [b_idx for b_idx in covers[idx] if b_idx in available]
                
                if min_count == 1:
                    break
                if min_count == 0:
                    return False # Dead end
            
        if best_col == -1 or min_count == 0:
            return False
            
        # Try each block that covers best_col
        for b_idx in best_covers:
            # If we choose block b_idx:
            # 1. It must be in available (it is, by construction of best_covers)
            # 2. Remove b_idx from available
            # 3. Remove all 5-subsets covered by b_idx from uncovered
            # 4. Remove all blocks that conflict with b_idx from available
            #    A block conflicts if it shares any 5-subset with b_idx.
            
            new_uncovered = uncovered & ~block_covered_bits[b_idx]
            
            # Calculate new available set
            # A block j is available if:
            # 1. j is in current available
            # 2. j does not conflict with b_idx
            # Conflict means (block_covered_bits[j] & block_covered_bits[b_idx]) != 0
            
            # To be efficient, we can iterate through the 5-subsets covered by b_idx
            # and remove all blocks covering those 5-subsets.
            # blocks_to_remove = set()
            # blocks_to_remove.add(b_idx)
            # mask_b = block_covered_bits[b_idx]
            # temp_mask = mask_b
            # while temp_mask:
            #    lsb = temp_mask & -temp_mask
            #    idx = lsb.bit_length() - 1
            #    temp_mask ^= lsb
            #    for j in covers[idx]:
            #        if j in available:
            #            blocks_to_remove.add(j)
            
            # This is O(6 * 6) = 36 operations per step, which is very fast.
            
            # Let's implement the removal
            new_available = set()
            # We can just filter the current available set
            # But creating a new set is O(N). N is up to 924.
            # 924 * 132 steps * 132 depth? No, branching factor drops.
            # Let's just filter.
            
            mask_b = block_covered_bits[b_idx]
            
            # We need to keep block j if (block_covered_bits[j] & mask_b) == 0
            # And j != b_idx
            
            # Optimization: 
            # Instead of checking all 924 blocks, we can precompute conflicts?
            # No, dynamic availability changes.
            # But we can just iterate available.
            
            for j in available:
                if j == b_idx:
                    continue
                if (block_covered_bits[j] & mask_b) == 0:
                    new_available.add(j)
            
            # Recurse
            current_sol.append(b_idx)
            if backtrack(new_uncovered, new_available, current_sol):
                return True
            current_sol.pop()
            
        return False

    # Start search
    if backtrack(initial_uncovered, initial_available, solution):
        # Convert block indices to the actual subsets
        result_blocks = [list(all_6_subsets[i]) for i in solution]
        return result_blocks
    else:
        # Fallback if exact cover fails or times out (should not happen for this size)
        # Return a greedy packing as fallback to ensure valid output
        from itertools import combinations as comb
        covered, blocks = set(), []
        for b in comb(range(12), 6):
            subsets = list(comb(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks

# Keep the interface exactly as requested
# The function name is construct
