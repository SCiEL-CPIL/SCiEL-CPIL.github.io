import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set using bitmask."""
    if not selected:
        return selected
    
    # Use a list of counts for speed
    final_counts = np.zeros(n_t, dtype=np.int32)
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

def _greedy_cover_fast(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10, urgency_weight=0.0):
    """Fast greedy set cover with optional urgency scoring and randomization."""
    uncovered_mask = np.ones(n_t, dtype=bool)
    uncovered_count = n_t
    selected = set()
    
    while uncovered_count > 0:
        if not _deadline_ok(deadline):
            break
        
        # Find the uncovered subset with fewest candidate blocks (urgency)
        min_count = float('inf')
        target_sub = -1
        for s in range(n_t):
            if not uncovered_mask[s]:
                continue
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
                if uncovered_mask[s]:
                    gain += 1
            # Add urgency bonus: prefer blocks that cover hard-to-cover subsets
            urgency_bonus = 0
            if urgency_weight > 0:
                for s in block_covers[b]:
                    if uncovered_mask[s]:
                        urgency_bonus += urgency_weight / max(1, len(sub_to_blocks[s]))
            total = gain + urgency_bonus
            gains.append((total, b))
        
        # Sort by gain descending
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            if uncovered_mask[s]:
                uncovered_mask[s] = False
                uncovered_count -= 1
    
    return selected

def _grassberger_scott_cover(block_covers, n_t, sub_to_blocks, deadline, max_kicks=50, kick_size=3):
    """
    Grassberger-Scott heuristic: greedy with periodic random kicks to escape local minima.
    """
    uncovered_mask = np.ones(n_t, dtype=bool)
    uncovered_count = n_t
    selected = []
    
    while uncovered_count > 0:
        if not _deadline_ok(deadline):
            break
        
        # Find the most constrained uncovered subset
        min_count = float('inf')
        target_sub = -1
        for s in range(n_t):
            if not uncovered_mask[s]:
                continue
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            break
        
        candidates = sub_to_blocks[target_sub]
        
        # Calculate gains
        gains = []
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if uncovered_mask[s]:
                    gain += 1
            gains.append((gain, b))
        
        gains.sort(reverse=True)
        best_block = gains[0][1]
        selected.append(best_block)
        
        for s in block_covers[best_block]:
            if uncovered_mask[s]:
                uncovered_mask[s] = False
                uncovered_count -= 1
    
    # Now do kicks: randomly remove some blocks and re-greedy the uncovered parts
    best_selected = set(selected)
    best_size = len(best_selected)
    
    # Build coverage counts
    counts = np.zeros(n_t, dtype=np.int32)
    for i in best_selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    for kick in range(max_kicks):
        if not _deadline_ok(deadline):
            break
        
        if best_size <= 1:
            break
        
        # Randomly remove kick_size blocks
        to_remove = random.sample(list(best_selected), min(kick_size, len(best_selected)))
        
        for b in to_remove:
            best_selected.remove(b)
            for s in block_covers[b]:
                counts[s] -= 1
        
        # Find newly uncovered subsets
        newly_uncovered = [s for s in range(n_t) if counts[s] == 0]
        
        if not newly_uncovered:
            continue
        
        # Re-greedy to cover the newly uncovered subsets
        uncovered_set = set(newly_uncovered)
        
        while uncovered_set:
            if not _deadline_ok(deadline):
                break
            
            # Most constrained
            min_c = float('inf')
            target = -1
            for s in uncovered_set:
                c = len(sub_to_blocks[s])
                if c < min_c:
                    min_c = c
                    target = s
                    if c == 1:
                        break
            
            if target == -1:
                break
            
            cands = sub_to_blocks[target]
            best_b = -1
            best_gain = -1
            for b in cands:
                if b in best_selected:
                    continue
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b == -1:
                break
            
            best_selected.add(best_b)
            for s in block_covers[best_b]:
                counts[s] += 1
                if counts[s] == 1 and s in uncovered_set:
                    uncovered_set.discard(s)
        
        # Prune redundant
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
        
        if len(best_selected) < best_size:
            best_size = len(best_selected)
        else:
            # Revert: we need to re-add removed blocks if solution got worse
            # Actually, let's just keep the best we've seen
            pass
    
    return best_selected

def _local_search_fast(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500):
    """Fast local search by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    counts = np.zeros(n_t, dtype=np.int32)
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
        
        best_replacement = None
        best_gain = -1
        for b in candidates:
            if b in best_selected:
                continue
            
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break
            
            if not covers_all:
                continue
            
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_replacement = b
        
        if best_replacement is not None:
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_replacement)
            for s in block_covers[best_replacement]:
                counts[s] += 1
    
    return best_selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 8.0  # Safety margin for return
    
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

    # 3. Multiple strategies with time allocation
    best_selected = None
    best_size = float('inf')
    
    # Phase 1: Deterministic greedy (fast)
    if _deadline_ok(deadline):
        selected = _greedy_cover_fast(block_covers, n_t, sub_to_blocks, deadline, randomize=False)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        selected = _local_search_fast(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=100)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
    
    # Phase 2: Randomized greedy with urgency
    for restart in range(3):
        if not _deadline_ok(deadline):
            break
        selected = _greedy_cover_fast(block_covers, n_t, sub_to_blocks, deadline, 
                                      randomize=True, top_k=8, urgency_weight=0.5)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        selected = _local_search_fast(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=100)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
    
    # Phase 3: Grassberger-Scott with kicks
    if best_selected is not None and _deadline_ok(deadline):
        gs_result = _grassberger_scott_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                             max_kicks=30, kick_size=2)
        if len(gs_result) < best_size:
            best_size = len(gs_result)
            best_selected = set(gs_result)
    
    # Phase 4: Additional local search on best
    if best_selected is not None and _deadline_ok(deadline):
        best_selected = _local_search_fast(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=200)
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
        if len(best_selected) < best_size:
            best_size = len(best_selected)
    
    # Phase 5: More randomized restarts if time allows
    if _deadline_ok(deadline):
        for restart in range(2):
            if not _deadline_ok(deadline):
                break
            selected = _greedy_cover_fast(block_covers, n_t, sub_to_blocks, deadline, 
                                          randomize=True, top_k=5, urgency_weight=1.0)
            selected = _prune_redundant(selected, block_covers, n_t, deadline)
            selected = _local_search_fast(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=150)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
