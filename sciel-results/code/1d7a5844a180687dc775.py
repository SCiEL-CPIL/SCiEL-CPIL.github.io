import time
import sys
from itertools import combinations

# Increase recursion limit for backtracking
sys.setrecursionlimit(2000)

# Precomputation
# Points 0..11
# All 6-subsets
all_blocks = []
block_to_5sets = []
# Map 5-subset to ID
# 5-subsets of 0..11
five_sets = list(combinations(range(12), 5))
five_set_to_id = {s: i for i, s in enumerate(five_sets)}

# For each 5-set, list of block indices that contain it
# This helps in MRV: for a 5-set S, we only need to check blocks containing S.
# But we also need to check if the OTHER 5-sets of the block are covered.
# So for each block, we store its 5-set IDs.

for b in combinations(range(12), 6):
    ids = [five_set_to_id[s] for s in combinations(b, 5)]
    all_blocks.append(list(b))
    block_to_5sets.append(ids)

# Invert: for each 5-set ID, which blocks contain it?
blocks_containing_5set = [[] for _ in range(len(five_sets))]
for i, ids in enumerate(block_to_5sets):
    for id in ids:
        blocks_containing_5set[id].append(i)

# Global state for backtracking
covered = set()
blocks_found = []

def backtrack(deadline):
    # Check deadline
    if time.time() > deadline:
        return False
        
    # Find all uncovered 5-sets
    uncovered = [i for i in range(len(five_sets)) if i not in covered]
    if not uncovered:
        return True
        
    # MRV: Find the uncovered 5-set with the fewest valid candidate blocks
    min_count = float('inf')
    best_5set = -1
    best_candidates = []
    
    for s_id in uncovered:
        # Get blocks containing this 5-set
        candidates = []
        for b_idx in blocks_containing_5set[s_id]:
            # Check if all 5-sets of this block are uncovered
            # The 5-set s_id is uncovered by definition.
            # Check the other 5.
            valid = True
            for other_id in block_to_5sets[b_idx]:
                if other_id != s_id and other_id in covered:
                    valid = False
                    break
            if valid:
                candidates.append(b_idx)
                
        if len(candidates) < min_count:
            min_count = len(candidates)
            best_5set = s_id
            best_candidates = candidates
            
            if min_count == 0:
                return False # Dead end
            
            if min_count == 1:
                # Forced move, no need to check others? 
                # Actually, we should still check if other 5-sets are valid?
                # No, if we pick this block, it covers s_id.
                # The constraint is that we MUST cover all 5-sets.
                # If we pick a block that doesn't contain some other 5-set, that's fine, it will be covered later.
                # But if some 5-set has 0 candidates, we fail.
                # So we must check ALL uncovered 5-sets to see if any have 0 candidates.
                # Wait, if we pick a block, it might cover other 5-sets, reducing their candidate count.
                # But we are checking the CURRENT state.
                # If in the current state, some 5-set has 0 candidates, we are stuck.
                # So we must scan all.
                pass

    # If we found a 5-set with 0 candidates, backtrack
    if min_count == 0:
        return False

    # Pick a block from best_candidates
    # Heuristic: pick the first one? Or random?
    # Deterministic is better for reproducibility and speed if it works.
    b_idx = best_candidates[0]
    
    # Add block
    blocks_found.append(all_blocks[b_idx])
    for id in block_to_5sets[b_idx]:
        covered.add(id)
        
    # Recurse
    if backtrack(deadline):
        return True
        
    # Backtrack
    blocks_found.pop()
    for id in block_to_5sets[b_idx]:
        covered.discard(id)
        
    # Try other candidates?
    # If the first choice failed, try the next.
    for b_idx in best_candidates[1:]:
        blocks_found.append(all_blocks[b_idx])
        for id in block_to_5sets[b_idx]:
            covered.add(id)
            
        if backtrack(deadline):
            return True
            
        blocks_found.pop()
        for id in block_to_5sets[b_idx]:
            covered.discard(id)
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Reset state
    global covered, blocks_found
    covered = set()
    blocks_found = []
    
    # To improve speed, we can fix the first block due to symmetry?
    # The group acts transitively on blocks? No, on flags?
    # We can just let it search.
    
    if backtrack(deadline):
        return blocks_found
    
    # If it fails (shouldn't happen for S(5,6,12) if search space is explored correctly),
    # return what we have? Or try a different strategy?
    # Given the uniqueness, it should find it.
    return blocks_found

# To ensure the function is self-contained and doesn't rely on global state pollution between calls
# (though construct is called once usually), we wrap the logic.

def construct():
    import time
    import sys
    from itertools import combinations
    
    sys.setrecursionlimit(5000)
    
    # Precomputation
    five_sets = list(combinations(range(12), 5))
    five_set_to_id = {s: i for i, s in enumerate(five_sets)}
    n_5sets = len(five_sets)
    
    all_blocks = []
    block_to_5sets = []
    blocks_containing_5set = [[] for _ in range(n_5sets)]
    
    for b in combinations(range(12), 6):
        ids = [five_set_to_id[s] for s in combinations(b, 5)]
        all_blocks.append(list(b))
        block_to_5sets.append(ids)
        for id in ids:
            blocks_containing_5set[id].append(len(all_blocks) - 1)
            
    covered = set()
    blocks_found = []
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        # Optimization: If number of uncovered 5-sets is small, we can be faster
        uncovered = [i for i in range(n_5sets) if i not in covered]
        if not uncovered:
            return True
            
        min_count = float('inf')
        best_candidates = []
        
        for s_id in uncovered:
            candidates = []
            for b_idx in blocks_containing_5set[s_id]:
                valid = True
                for other_id in block_to_5sets[b_idx]:
                    if other_id != s_id and other_id in covered:
                        valid = False
                        break
                if valid:
                    candidates.append(b_idx)
                    
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_candidates = candidates
                
                if min_count == 0:
                    return False
                    
        if min_count == 0:
            return False
            
        # Try candidates
        for b_idx in best_candidates:
            blocks_found.append(all_blocks[b_idx])
            for id in block_to_5sets[b_idx]:
                covered.add(id)
                
            if backtrack():
                return True
                
            blocks_found.pop()
            for id in block_to_5sets[b_idx]:
                covered.discard(id)
                
        return False
        
    if backtrack():
        return blocks_found
        
    # Fallback: Return whatever was found (should be empty or partial)
    # But for this task, we expect 132.
    return blocks_found
