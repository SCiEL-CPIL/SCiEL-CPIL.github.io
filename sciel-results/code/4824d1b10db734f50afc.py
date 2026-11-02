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

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500):
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
    no_improve_count = 0
    max_no_improve = 50
    
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
            no_improve_count = 0
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
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
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
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
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

def _walksat_cover(block_covers, n_t, sub_to_blocks, deadline, initial_selected, max_steps=5000, p=0.5):
    """
    WalkSAT-style local search for set cover.
    Objective: minimize number of blocks while covering all t-subsets.
    We use a penalty-based approach: cost = num_blocks + penalty * num_uncovered
    """
    if not initial_selected:
        return set()
    
    selected = set(initial_selected)
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    # Count uncovered
    uncovered_count = sum(1 for c in counts if c == 0)
    
    # Penalty weight: make covering more important than minimizing blocks
    # But we want to find minimal cover, so we balance
    # Use a dynamic penalty: start high, decrease over time
    penalty = 10.0
    
    best_selected = set(selected)
    best_cost = len(selected) + penalty * uncovered_count
    
    for step in range(max_steps):
        if not _deadline_ok(deadline):
            break
        
        # Decrease penalty over time to focus on size reduction
        penalty = max(1.0, 10.0 * (1.0 - step / max_steps))
        
        current_cost = len(selected) + penalty * uncovered_count
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_selected = set(selected)
        
        # If fully covered, try to reduce size
        if uncovered_count == 0:
            # Try to remove a random block if it's redundant
            if selected:
                block_to_try = random.choice(list(selected))
                can_remove = True
                for s in block_covers[block_to_try]:
                    if counts[s] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    selected.remove(block_to_try)
                    for s in block_covers[block_to_try]:
                        counts[s] -= 1
                    continue
        
        # Choose a move: flip a block (add or remove)
        # With probability p, pick a block that covers an uncovered subset
        # Otherwise, pick a random block
        if random.random() < p and uncovered_count > 0:
            # Pick a random uncovered subset
            uncovered_subs = [i for i, c in enumerate(counts) if c == 0]
            if uncovered_subs:
                target_sub = random.choice(uncovered_subs)
                candidates = sub_to_blocks[target_sub]
                if candidates:
                    b = random.choice(candidates)
                    if b in selected:
                        # Try removing it? No, it's needed for coverage
                        # Instead, try swapping with another block covering target_sub
                        continue
                    else:
                        # Add this block
                        selected.add(b)
                        for s in block_covers[b]:
                            counts[s] += 1
                        # Update uncovered count
                        for s in block_covers[b]:
                            if counts[s] == 1:
                                uncovered_count -= 1
                continue
            else:
                # No uncovered, try to remove a block
                if selected:
                    b = random.choice(list(selected))
                    can_remove = True
                    for s in block_covers[b]:
                        if counts[s] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        selected.remove(b)
                        for s in block_covers[b]:
                            counts[s] -= 1
                        for s in block_covers[b]:
                            if counts[s] == 0:
                                uncovered_count += 1
                    continue
        
        # Random move: pick a random block and flip it
        all_blocks = list(range(len(block_covers)))
        b = random.choice(all_blocks)
        
        if b in selected:
            # Try removing
            can_remove = True
            for s in block_covers[b]:
                if counts[s] <= 1:
                    can_remove = False
                    break
            if can_remove:
                selected.remove(b)
                for s in block_covers[b]:
                    counts[s] -= 1
                for s in block_covers[b]:
                    if counts[s] == 0:
                        uncovered_count += 1
            else:
                # Can't remove without breaking coverage, so add a different block?
                # Or just skip
                pass
        else:
            # Try adding
            # Check if this helps
            gain = 0
            for s in block_covers[b]:
                if counts[s] == 0:
                    gain += 1
            
            # Only add if it covers something new or if we have too many blocks
            if gain > 0 or len(selected) > n_t // 2:
                selected.add(b)
                for s in block_covers[b]:
                    counts[s] += 1
                for s in block_covers[b]:
                    if counts[s] == 1:
                        uncovered_count -= 1
    
    # Final: ensure validity
    # If best_selected doesn't cover everything, use the last valid state
    # Actually, we track best_cost which includes penalty, so best_selected should be good
    # But let's verify
    best_counts = [0] * n_t
    for i in best_selected:
        for s in block_covers[i]:
            best_counts[s] += 1
    best_uncovered = sum(1 for c in best_counts if c == 0)
    
    if best_uncovered > 0:
        # best_selected is invalid, fall back to initial or last valid
        # For now, return initial if it was valid
        init_counts = [0] * n_t
        for i in initial_selected:
            for s in block_covers[i]:
                init_counts[s] += 1
        init_uncovered = sum(1 for c in init_counts if c == 0)
        if init_uncovered == 0:
            return set(initial_selected)
        else:
            return best_selected  # Even if invalid, caller should check
    
    return best_selected

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
    
    num_restarts = 12
    
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
                
    # 4. WalkSAT-style local search
    if best_selected is not None and _deadline_ok(deadline):
        walksat_result = _walksat_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                         best_selected, max_steps=10000, p=0.6)
        if walksat_result is not None:
            # Verify validity
            ws_counts = [0] * n_t
            for i in walksat_result:
                for s in block_covers[i]:
                    ws_counts[s] += 1
            ws_uncovered = sum(1 for c in ws_counts if c == 0)
            if ws_uncovered == 0:
                # Prune
                walksat_result = _prune_redundant(walksat_result, block_covers, n_t, deadline)
                if len(walksat_result) < best_size:
                    best_size = len(walksat_result)
                    best_selected = walksat_result
    
    # 5. Final local search pass
    if best_selected is not None and _deadline_ok(deadline):
        best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300)
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
        best_size = len(best_selected)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
