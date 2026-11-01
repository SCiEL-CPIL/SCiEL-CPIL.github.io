import itertools
import time

def construct(v, k, t):
    # Precompute subsets and blocks
    # v=19, k=7, t=4
    
    # Map t-subsets to IDs
    t_subsets = list(itertools.combinations(range(v), t))
    sub_to_id = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    # Map k-blocks to IDs and their covered t-subsets
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # Precompute for each block, which t-subsets it covers
    block_covers = []
    for b in k_blocks:
        covers = [sub_to_id[s] for s in itertools.combinations(b, t)]
        block_covers.append(covers)
        
    # Precompute for each t-subset, which k-blocks cover it
    # This is needed for the "most constrained" heuristic
    sub_covers_by = [[] for _ in range(n_subsets)]
    for b_idx, covers in enumerate(block_covers):
        for s_idx in covers:
            sub_covers_by[s_idx].append(b_idx)
            
    # State
    coverage_count = [0] * n_subsets
    uncovered_set = set(range(n_subsets))
    used_blocks = set()
    solution = []
    
    # avail_blocks_count[s] = number of blocks NOT in used_blocks that cover s
    avail_blocks_count = [len(sub_covers_by[s]) for s in range(n_subsets)]
    
    while uncovered_set:
        # Find the uncovered subset with the fewest available blocks
        # This is the most constrained
        min_avail = float('inf')
        critical_sub = None
        for s in uncovered_set:
            if avail_blocks_count[s] < min_avail:
                min_avail = avail_blocks_count[s]
                critical_sub = s
                if min_avail == 0:
                    # Should not happen if a solution exists, but break to avoid crash
                    break
                
        if critical_sub is None:
            break
            
        # Get candidates: blocks covering critical_sub and not used
        candidates = []
        for b_idx in sub_covers_by[critical_sub]:
            if b_idx not in used_blocks:
                candidates.append(b_idx)
                
        if not candidates:
            # No block available to cover this subset. 
            # This implies the current set of used blocks is preventing a solution? 
            # Or we made a bad choice earlier. 
            # For a greedy heuristic, we just pick any block if candidates empty? 
            # No, if candidates empty, it means all blocks covering this subset are used.
            # But the subset is uncovered. This is a contradiction unless the subset was covered by a used block 
            # but we marked it uncovered? 
            # Wait, if a block is used, it covers the subset. 
            # So if critical_sub is uncovered, no used block covers it.
            # So all blocks covering it are NOT used.
            # So candidates should not be empty.
            pass

        # Score candidates
        best_block = None
        best_score = (-1, -1) # (newly_covered, tie_breaker)
        
        for b_idx in candidates:
            covers = block_covers[b_idx]
            newly_covered = 0
            tie_breaker = 0.0
            for s in covers:
                if coverage_count[s] == 0:
                    newly_covered += 1
                tie_breaker += 1.0 / (coverage_count[s] + 1)
            
            if (newly_covered, tie_breaker) > best_score:
                best_score = (newly_covered, tie_breaker)
                best_block = b_idx
                
        # Add best_block to solution
        if best_block is not None:
            solution.append(k_blocks[best_block])
            used_blocks.add(best_block)
            
            # Update coverage and avail counts
            for s in block_covers[best_block]:
                coverage_count[s] += 1
                if coverage_count[s] == 1:
                    uncovered_set.remove(s)
                
                # Update avail_blocks_count for this subset
                # The block we just used is no longer available for future selections
                avail_blocks_count[s] -= 1

    # Pruning Phase
    # Try to remove blocks that are redundant
    pruned_solution = list(solution)
    changed = True
    while changed:
        changed = False
        new_solution = []
        for b in pruned_solution:
            # Check if we can remove b
            # Calculate coverage without b
            # We need to know if any subset drops to 0
            # We can simulate removal
            removable = True
            affected_subsets = []
            for s in block_covers[k_blocks.index(b)]: # Need index mapping
                pass
            
        # The above loop is inefficient due to index lookup. 
        # Let's store block indices in solution.
        
    # Let's restart the pruning logic properly using indices.
    
    return solution
