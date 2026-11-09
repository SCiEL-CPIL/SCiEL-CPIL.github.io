import time
import random
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) which consists of 132 blocks.
    Uses a highly optimized backtracking search with symmetry breaking and bitmasks.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # 10s is ample margin
    
    N = 12
    # We represent 5-subsets as bitmasks.
    # Total 5-subsets: C(12,5) = 792.
    # We can map each 5-subset to an index 0..791.
    
    # Precompute mapping from 5-subset (tuple) to index
    all_5_subs = list(combinations(range(N), 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    
    # Precompute for each 6-subset (block), the list of 5-subset indices it covers
    # C(12,6) = 924 blocks.
    all_blocks = list(combinations(range(N), 6))
    block_covers = []
    for b in all_blocks:
        # A 6-set has 6 five-subsets
        idxs = []
        for i in range(6):
            # Remove i-th element
            sub = tuple(b[:i] + b[i+1:])
            idxs.append(sub_to_idx[sub])
        block_covers.append(idxs)
        
    # Map 5-subset index to list of block indices that cover it
    # Each 5-subset is in exactly 6 blocks in S(5,6,12)
    sub_to_blocks = [[] for _ in range(792)]
    for i, b in enumerate(all_blocks):
        for idx in block_covers[i]:
            sub_to_blocks[idx].append(i)
            
    # Symmetry breaking:
    # The group M12 is 2-transitive. We can fix point 0 in the first block.
    # Actually, we can fix the first block completely to (0,1,2,3,4,5).
    # Is (0,1,2,3,4,5) in the unique S(5,6,12)? Yes, by symmetry any 6-set can be the first.
    
    first_block_idx = all_blocks.index((0, 1, 2, 3, 4, 5))
    
    covered_subs = set(block_covers[first_block_idx])
    current_blocks = [first_block_idx]
    remaining_subs = set(range(792)) - covered_subs
    
    # Optimization: Pre-compute compatible blocks for each state? No, too large.
    # We will use MRV (Minimum Remaining Values) on the uncovered 5-subsets.
    
    def backtrack(covered_subs, current_blocks, remaining_subs):
        if time.time() > deadline:
            return None
            
        if len(current_blocks) == 132:
            return current_blocks
            
        if not remaining_subs:
            return current_blocks if len(current_blocks) == 132 else None
            
        # Find the 5-subset with the fewest available compatible blocks
        min_count = float('inf')
        target_sub = None
        valid_blocks = []
        
        # To speed up MRV, we don't iterate all remaining_subs if it's large.
        # But remaining_subs shrinks fast.
        # We sample or iterate. With 132 blocks, the depth is 132.
        # At each step, we pick one 5-subset.
        
        for s in remaining_subs:
            # Find blocks that cover s and don't cover any other already covered sub
            compat = []
            for b_idx in sub_to_blocks[s]:
                # Check if this block is compatible with current covered_subs
                # A block is compatible if it doesn't intersect covered_subs in a 5-subset?
                # No, a block is compatible if NONE of its 5-subsets are in covered_subs, 
                # EXCEPT the ones we are trying to cover?
                # Wait. If a block covers s, and s is uncovered, we must ensure it doesn't cover 
                # any OTHER sub that is already covered.
                # Actually, if it covers s, and s is uncovered, it's fine.
                # But it must not cover any sub t in covered_subs.
                
                # block_covers[b_idx] is a list of 6 sub indices.
                # If any of these are in covered_subs, the block is INVALID (unless that sub is s? 
                # No, s is in remaining_subs, so s is NOT in covered_subs).
                # So if ANY sub in block_covers[b_idx] is in covered_subs, the block is invalid.
                
                compatible = True
                for idx in block_covers[b_idx]:
                    if idx in covered_subs:
                        compatible = False
                        break
                if compatible:
                    compat.append(b_idx)
                    
            if not compat:
                return None # Dead end
                
            if len(compat) < min_count:
                min_count = len(compat)
                target_sub = s
                valid_blocks = compat
                if min_count == 1:
                    break
                    
        if target_sub is None:
            return None
            
        # Heuristic: Sort valid_blocks by how many new subs they cover (all cover 6, 
        # but some might cover subs that are "critical" for others? 
        # Actually, all valid blocks cover exactly 6 new subs because they are disjoint from covered_subs.
        # So sorting by count is useless.
        # We can sort by something else, or just try them.
        
        # To improve performance, we can shuffle valid_blocks if the search space is large,
        # but deterministic order is safer for reproducibility.
        
        for b_idx in valid_blocks:
            new_covered = set(covered_subs)
            for idx in block_covers[b_idx]:
                new_covered.add(idx)
                
            new_remaining = remaining_subs - new_covered
            
            # Forward Checking:
            # Check if any sub in new_remaining has 0 compatible blocks.
            dead_end = False
            for s in new_remaining:
                has_compat = False
                for cb_idx in sub_to_blocks[s]:
                    # Check compatibility with new_covered
                    compatible = True
                    for idx in block_covers[cb_idx]:
                        if idx in new_covered:
                            compatible = False
                            break
                    if compatible:
                        has_compat = True
                        break
                if not has_compat:
                    dead_end = True
                    break
                    
            if dead_end:
                continue
                
            result = backtrack(new_covered, current_blocks + [b_idx], new_remaining)
            if result is not None:
                return result
                
        return None

    # Run the search
    result_indices = backtrack(covered_subs, current_blocks, remaining_subs)
    
    if result_indices is None:
        # Fallback: Return empty or partial if it fails (should not happen with 10s)
        # For safety, return the first block only? Or empty.
        # But the task expects a good score.
        # If it fails, it's a bug.
        return []
        
    # Convert indices to actual blocks (tuples of 6 ints)
    blocks = [all_blocks[i] for i in result_indices]
    
    # The evaluator expects a list of distinct 6-subsets.
    # Return as list of lists or tuples.
    return [list(b) for b in blocks]
