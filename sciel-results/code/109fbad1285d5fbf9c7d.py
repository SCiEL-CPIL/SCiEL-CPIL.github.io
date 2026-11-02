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
        return selected
    
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
            
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
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

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=200):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    # Build coverage counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        # Try to remove a random block
        block_to_remove = random.choice(list(best_selected))
        
        # Find uncovered subsets if we remove this block
        uncovered_if_removed = []
        for s in block_covers[block_to_remove]:
            if counts[s] <= 1:
                uncovered_if_removed.append(s)
        
        if not uncovered_if_removed:
            # Block is redundant, remove it
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_size -= 1
            continue
        
        # Try to find a replacement block that covers all uncovered_if_removed
        best_replacement = None
        
        # Get candidate blocks from the most constrained uncovered subset
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_if_removed:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            continue
        
        candidates = sub_to_blocks[target_sub]
        
        # Try candidates, prefer those that cover more uncovered subsets
        best_gain = -1
        for b in candidates:
            if b in best_selected:
                continue
            
            # Check if this block covers all uncovered_if_removed
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            # Count how many currently covered subsets this block would add
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_replacement = b
        
        if best_replacement is not None:
            # Perform the swap
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_replacement)
            for s in block_covers[best_replacement]:
                counts[s] += 1
    
    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
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
        
        candidates = sub_to_blocks[target_sub]
        
        # Calculate gains for all candidates
        gains = []
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            gains.append((gain, b))
        
        # Sort by gain descending
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

def _branch_and_bound(uncovered, block_covers, sub_to_blocks, best_size, deadline):
    """
    Branch and bound solver for Set Cover.
    Returns a set of block indices if a better solution is found, else None.
    """
    if not uncovered:
        return set()
    
    if not _deadline_ok(deadline):
        return None
    
    # Lower Bound: Max number of disjoint subsets in uncovered
    # Greedy disjoint packing
    remaining = set(uncovered)
    lb = 0
    while remaining:
        # Pick a random subset from remaining to start a "group"
        s = next(iter(remaining))
        lb += 1
        # Remove all subsets that intersect with s? No, disjoint means no common elements?
        # Wait, t-subsets are disjoint if they share no points.
        # But here 'uncovered' is a set of t-subset indices.
        # Two t-subsets are disjoint if their underlying point sets are disjoint.
        # We need to precompute disjointness or check on the fly.
        # For speed, we can use a weaker LB: ceil(len(uncovered) / max_cover)
        # max_cover is the max number of uncovered subsets a single block can cover.
        max_cov = 0
        for b in sub_to_blocks[s]: # Just check candidates of s? No, any block.
            # This is expensive.
            pass
        
        # Let's use a simpler, faster LB:
        # LB = ceil( |uncovered| / max_{b} |b \cap uncovered| )
        # We can estimate max_cov by checking a few candidates.
        break # Skip complex LB for now, use simple one below
        
    # Simple LB
    # Find max coverage among blocks covering the most constrained subset
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
        return None
        
    max_cov = 0
    for b in sub_to_blocks[target_sub]:
        cov = 0
        for s in block_covers[b]:
            if s in uncovered:
                cov += 1
        if cov > max_cov:
            max_cov = cov
            
    if max_cov == 0:
        return None
        
    lb = (len(uncovered) + max_cov - 1) // max_cov
    
    if lb >= best_size:
        return None
        
    # Branch on target_sub
    # We must pick one block from sub_to_blocks[target_sub]
    candidates = sub_to_blocks[target_sub]
    
    # Sort candidates by coverage descending to find good solutions earlier
    sorted_candidates = []
    for b in candidates:
        cov = 0
        for s in block_covers[b]:
            if s in uncovered:
                cov += 1
        sorted_candidates.append((cov, b))
    sorted_candidates.sort(reverse=True)
    
    for cov, b in sorted_candidates:
        if not _deadline_ok(deadline):
            return None
            
        # Try adding b
        new_uncovered = set(uncovered)
        for s in block_covers[b]:
            new_uncovered.discard(s)
            
        # Recurse
        # We pass best_size - 1 because we used 1 block
        result = _branch_and_bound(new_uncovered, block_covers, sub_to_blocks, best_size - 1, deadline)
        
        if result is not None:
            result.add(b)
            return result
            
    return None

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

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                  randomize=(restart > 0), top_k=10)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
                
    # 4. Branch and Bound attempt
    if best_selected is not None and _deadline_ok(deadline):
        # Use the current best as the initial upper bound
        # B&B will try to find a solution with < best_size blocks
        uncovered = set(range(n_t))
        # Verify best_selected covers everything
        # (Assuming local search/prune maintained validity)
        
        # To make B&B effective, we should start from the uncovered set
        # and try to prove that best_size is not optimal or find something smaller.
        # However, B&B as implemented above starts from scratch.
        # We can use the best_solution to guide the search?
        # For now, just run B&B with bound = best_size.
        
        # Optimization: Limit the search depth or nodes?
        # The recursive B&B might be too slow.
        # We'll add a node counter to the B&B to stop if it gets too deep/wide.
        
        node_count = [0]
        max_nodes = 10000 # Safety limit
        
        def _bndb(uncovered, best_limit):
            if not _deadline_ok(deadline) or node_count[0] > max_nodes:
                return None
            node_count[0] += 1
            
            if not uncovered:
                return set()
                
            # LB
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
                return None
                
            max_cov = 0
            for b in sub_to_blocks[target_sub]:
                cov = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    
            if max_cov == 0:
                return None
                
            lb = (len(uncovered) + max_cov - 1) // max_cov
            
            if lb >= best_limit:
                return None
                
            candidates = sub_to_blocks[target_sub]
            
            sorted_candidates = []
            for b in candidates:
                cov = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        cov += 1
                sorted_candidates.append((cov, b))
            sorted_candidates.sort(reverse=True)
            
            for cov, b in sorted_candidates:
                if not _deadline_ok(deadline) or node_count[0] > max_nodes:
                    return None
                    
                new_uncovered = set(uncovered)
                for s in block_covers[b]:
                    new_uncovered.discard(s)
                
                result = _bndb(new_uncovered, best_limit - 1)
                
                if result is not None:
                    result.add(b)
                    return result
                    
            return None
            
        bnb_result = _bndb(set(range(n_t)), best_size)
        
        if bnb_result is not None and len(bnb_result) < best_size:
            best_selected = bnb_result
            best_size = len(best_selected)
            
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
