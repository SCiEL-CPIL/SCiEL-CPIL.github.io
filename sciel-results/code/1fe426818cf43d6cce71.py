import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = range(12)
    all_5_subs = [tuple(s) for s in combinations(points, 5)]
    all_6_blocks = [tuple(b) for b in combinations(points, 6)]
    
    # Map 5-subset to list of blocks that contain it
    # And block to list of 5-subsets
    sub_to_blocks = {s: [] for s in all_5_subs}
    block_to_subs = {}
    
    for b in all_6_blocks:
        subs = [tuple(s) for s in combinations(b, 5)]
        block_to_subs[b] = subs
        for s in subs:
            sub_to_blocks[s].append(b)
            
    # Symmetry breaking: First block is (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    covered_subs = set(block_to_subs[first_block])
    selected_blocks = [first_block]
    
    # Initial uncovered
    uncovered = set(all_5_subs) - covered_subs
    
    # Candidates: All blocks except first_block
    # We will maintain a list of active candidates
    # To be fast, we don't pre-filter all, but check validity on the fly
    
    def backtrack(uncovered, selected):
        if time.time() > deadline:
            return selected  # Timeout, return best effort (should be 132 if found)
            
        if not uncovered:
            return selected
            
        # Find the 5-subset in uncovered that is covered by the fewest candidates
        # We need to know which candidates are still "valid" (i.e., all their subsets are in uncovered)
        # This is expensive to check for all.
        # Alternative: Pick the subset with the smallest global degree?
        # Or just pick the first one?
        # Given the symmetry, any choice might work, but MRV is safer.
        
        # Let's iterate through uncovered subsets to find the one with fewest *potential* blocks.
        # A block is potential if all its 5-subsets are in uncovered.
        
        min_count = float('inf')
        best_subset = None
        best_candidates = []
        
        # Optimization: Only check a subset of uncovered?
        # With 792 subsets, checking all is fast (792 * 6 checks).
        
        for s in uncovered:
            count = 0
            cands = []
            for b in sub_to_blocks[s]:
                # Check if b is valid: all subsets of b must be in uncovered
                valid = True
                for sub in block_to_subs[b]:
                    if sub not in uncovered:
                        valid = False
                        break
                if valid:
                    count += 1
                    cands.append(b)
                    
            if count < min_count:
                min_count = count
                best_subset = s
                best_candidates = cands
                if count == 0:
                    break # Dead end
                    
        if min_count == 0:
            return None # Fail
            
        # Try each candidate for best_subset
        for b in best_candidates:
            # Select b
            new_uncovered = set(uncovered)
            for sub in block_to_subs[b]:
                new_uncovered.remove(sub)
                
            new_selected = selected + [b]
            
            result = backtrack(new_uncovered, new_selected)
            if result is not None:
                return result
                
        return None

    result = backtrack(uncovered, selected_blocks)
    
    if result:
        return result
    else:
        # Fallback: Return what we have or empty?
        # Should not happen.
        return selected_blocks
