import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return set()
    
    # Calculate coverage counts
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
    
    # Try to remove blocks iteratively
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            # Check if block i is redundant
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    return selected

def _local_search(selected, blocks, block_covers, sub_to_blocks, n_t, deadline):
    """Local search: try swap moves (remove one block, add another) to reduce size."""
    if not selected:
        return selected
    
    # Calculate coverage counts
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(selected)
    improved = True
    
    while improved and _deadline_ok(deadline):
        improved = False
        
        # First pass: remove redundant blocks
        for b_remove in list(best_selected):
            if not _deadline_ok(deadline):
                break
            redundant = True
            for s in block_covers[b_remove]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                best_selected.discard(b_remove)
                for s in block_covers[b_remove]:
                    final_counts[s] -= 1
                best_size = len(best_selected)
                improved = True
        
        if not _deadline_ok(deadline):
            break
        
        # Second pass: try swap moves (remove b_remove, add b_add)
        # A swap is valid if after the swap, every t-subset is still covered.
        # Condition: for every s in block_covers[b_remove], if final_counts[s]==1 then s must be in block_covers[b_add]
        # And b_add must not already be in best_selected.
        
        # Build a set for fast membership
        sel_set = best_selected
        
        for b_remove in list(best_selected):
            if not _deadline_ok(deadline):
                break
            
            # Find subsets only covered by b_remove
            unique_subs = []
            for s in block_covers[b_remove]:
                if final_counts[s] == 1:
                    unique_subs.append(s)
            
            if not unique_subs:
                # Already handled in first pass, but double-check
                continue
            
            # Find candidate blocks that cover all unique_subs
            candidate_blocks = None
            for s in unique_subs:
                if candidate_blocks is None:
                    candidate_blocks = set(sub_to_blocks[s])
                else:
                    candidate_blocks &= set(sub_to_blocks[s])
            
            if not candidate_blocks:
                continue
            
            # Remove b_remove from candidates
            candidate_blocks.discard(b_remove)
            
            # Try each candidate
            for b_add in candidate_blocks:
                if b_add in sel_set:
                    continue
                
                # Check validity: for all s in block_covers[b_remove], 
                # if final_counts[s]==1, then s must be in block_covers[b_add]
                valid = True
                for s in block_covers[b_remove]:
                    if final_counts[s] == 1:
                        # s is only covered by b_remove, so b_add must cover it
                        if s not in block_covers[b_add]:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Valid swap: remove b_remove, add b_add
                # Net change: size stays same, but we might enable further removals
                # Apply the swap
                sel_set.discard(b_remove)
                sel_set.add(b_add)
                for s in block_covers[b_remove]:
                    final_counts[s] -= 1
                for s in block_covers[b_add]:
                    final_counts[s] += 1
                improved = True
                break  # Restart the outer loop after a swap
            
            if improved:
                break
    
    return best_selected

def _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline, rng=None):
    """Greedy set cover with randomized tie-breaking."""
    if rng is None:
        rng = random.Random()
    
    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
            
        # Find the uncovered subset with fewest candidate blocks
        min_count = float('inf')
        target_sub = -1
        for s in uncovered:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            break
            
        # Get candidates for target_sub
        candidates = sub_to_blocks[target_sub]
        
        # Pick the candidate that maximizes coverage of other uncovered subsets
        # Use a small random perturbation to diversify
        best_block = -1
        max_score = -1
        
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            # Add small random perturbation
            score = gain + rng.random() * 0.5
            if score > max_score:
                max_score = score
                best_block = b
        
        # Select best_block
        selected.add(best_block)
        
        # Update uncovered
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []
    
    # 3. Multiple randomized greedy restarts with local search
    best_result = None
    best_size = float('inf')
    
    # Number of restarts depends on available time
    elapsed = time.time() - start
    remaining = deadline - time.time()
    # Estimate time per restart: greedy is O(n_blocks * n_t) roughly, local search adds more
    # Let's try at least 3 restarts if time permits
    n_restarts = max(4, int(remaining / 15))  # ~15s per restart estimate
    n_restarts = min(n_restarts, 12)  # Cap at 12
    
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline, rng)
        
        if not selected:
            continue
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search (swap-based)
        if _deadline_ok(deadline):
            selected = _local_search(selected, blocks, block_covers, sub_to_blocks, n_t, deadline)
        
        # Final prune after local search
        if _deadline_ok(deadline):
            selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Check if this is better
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = set(selected)
    
    # If no result found, fall back to a single greedy
    if best_result is None:
        selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        best_result = selected
    
    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity (just in case)
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
            # Invalid, try again with a fresh greedy
            selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline)
            selected = _prune_redundant(selected, block_covers, n_t, deadline)
            result = [list(blocks[i]) for i in selected]
            
    return result
