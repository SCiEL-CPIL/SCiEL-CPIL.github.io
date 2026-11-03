import itertools
import time
import random
import sys
from collections import defaultdict

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
    """Try to improve the solution by removing blocks and minimal repair."""
    if not selected or len(selected) < 2:
        return selected
    
    # Build coverage counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(best_selected)
    
    # Precompute: for each block, which t-subsets does it uniquely cover (count==1)?
    # We'll recompute as needed since counts change.
    
    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not best_selected:
            break
        
        # Pick a random block to try removing
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
        
        # Try to find a minimal set of replacement blocks that covers all uncovered_if_removed
        # Greedy: repeatedly pick the block that covers the most uncovered subsets
        remaining_uncovered = set(uncovered_if_removed)
        replacements = []
        
        while remaining_uncovered:
            if not _deadline_ok(deadline):
                break
            
            # Find the block that covers the most remaining uncovered subsets
            best_block = -1
            best_gain = 0
            
            # Get candidate blocks from the most constrained uncovered subset
            min_count = float('inf')
            target_sub = -1
            for s in remaining_uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
            
            candidates = sub_to_blocks[target_sub]
            
            for b in candidates:
                if b in best_selected or b == block_to_remove:
                    continue
                
                gain = 0
                for s in block_covers[b]:
                    if s in remaining_uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block == -1 or best_gain == 0:
                break
            
            replacements.append(best_block)
            for s in block_covers[best_block]:
                remaining_uncovered.discard(s)
        
        # Check if replacements are valid and beneficial
        if replacements and len(replacements) < len(uncovered_if_removed):
            # Net change: -1 (removed) + len(replacements) (added)
            # Only accept if it reduces total size
            if len(replacements) < 1:
                # Shouldn't happen since uncovered_if_removed was non-empty
                pass
            else:
                # Accept the change: remove block_to_remove, add replacements
                # This is only beneficial if len(replacements) < 1, which means we removed without adding
                # But since uncovered_if_removed was non-empty, we need at least 1 replacement
                # So this path only helps if we can find redundant blocks later
                pass
        
        # Alternative: try a swap (remove one, add one) that might enable further removals
        if not _deadline_ok(deadline):
            break
        
        # Try 1-for-1 swap: remove block_to_remove, add one block that covers all uncovered
        best_swap = -1
        best_swap_gain = 0  # We want to minimize "damage" - maximize overlap with existing coverage
        
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_if_removed:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub != -1:
            candidates = sub_to_blocks[target_sub]
            for b in candidates:
                if b in best_selected or b == block_to_remove:
                    continue
                
                # Check if this block covers all uncovered_if_removed
                covers_all = True
                for s in uncovered_if_removed:
                    if s not in block_covers[b]:
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Count how many subsets this block already covers (redundant coverage)
                redundant_coverage = 0
                for s in block_covers[b]:
                    if counts[s] >= 1 and s != -1:
                        redundant_coverage += 1
                
                # We prefer blocks with more redundant coverage (less "waste")
                if redundant_coverage > best_swap_gain:
                    best_swap_gain = redundant_coverage
                    best_swap = b
        
        if best_swap != -1:
            # Perform the swap
            best_selected.remove(block_to_remove)
            for s in block_covers[block_to_remove]:
                counts[s] -= 1
            best_selected.add(best_swap)
            for s in block_covers[best_swap]:
                counts[s] += 1
    
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
    
    # Use more restarts if we have time; adaptive based on elapsed time
    num_restarts = 8
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy: at each step, pick randomly among top-k candidates
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
            
            # Pick randomly among top candidates
            top_k = min(15, len(gains))
            if restart == 0:
                # First restart: deterministic greedy
                best_block = gains[0][1]
            else:
                # Randomized: pick from top-k with weighting toward higher gain
                # Use softmax-like selection
                r = random.random()
                cumulative = 0
                best_block = gains[0][1]
                for idx in range(min(top_k, len(gains))):
                    # Weight by rank: higher gain = higher weight
                    w = 1.0 / (idx + 1)
                    cumulative += w
                    if r < cumulative:
                        best_block = gains[idx][1]
                        break
                else:
                    best_block = gains[min(top_k, len(gains)) - 1][1]
            
            selected.add(best_block)
            
            for s in block_covers[best_block]:
                uncovered.discard(s)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search - give it more iterations
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
        
        # Additional local search pass on the final best solution
        if _deadline_ok(deadline):
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500)
            best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
