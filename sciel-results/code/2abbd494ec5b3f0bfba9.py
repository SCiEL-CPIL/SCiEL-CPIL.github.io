import itertools
import time
import random
import sys
import numpy as np
from functools import lru_cache

def _deadline_ok(deadline):
    return time.time() < deadline

def _bitmask_from_list(lst):
    m = 0
    for i in lst:
        m |= (1 << i)
    return m

def _popcount(x):
    return bin(x).count('1')

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

def _local_search_2exchange(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=100):
    """Try to improve by removing 2 blocks and adding 1-2 blocks."""
    if not selected or len(selected) < 3:
        return selected
    
    selected_list = list(selected)
    counts = [0] * n_t
    for i in selected_list:
        for s in block_covers[i]:
            counts[s] += 1
    
    improved = True
    iter_count = 0
    while improved and _deadline_ok(deadline) and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # Try removing pairs of blocks
        n_sel = len(selected_list)
        for i in range(n_sel):
            if not _deadline_ok(deadline):
                break
            for j in range(i+1, n_sel):
                if not _deadline_ok(deadline):
                    break
                
                bi = selected_list[i]
                bj = selected_list[j]
                
                # Check what becomes uncovered
                uncovered_if_removed = []
                for s in block_covers[bi]:
                    if counts[s] <= 1:
                        uncovered_if_removed.append(s)
                for s in block_covers[bj]:
                    if counts[s] <= 2:
                        if counts[s] == 2:
                            # Only uncovered if both bi and bj cover it
                            uncovered_if_removed.append(s)
                        else:
                            uncovered_if_removed.append(s)
                
                if not uncovered_if_removed:
                    # Both blocks are redundant
                    selected_list.remove(bj)
                    selected_list.remove(bi)
                    for s in block_covers[bi]:
                        counts[s] -= 1
                    for s in block_covers[bj]:
                        counts[s] -= 1
                    improved = True
                    break
                
                # Try to find replacement blocks
                # Get candidates from most constrained uncovered subset
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
                
                # Try single replacement
                best_single = None
                best_gain = -1
                for b in candidates:
                    if b in selected_list:
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
                        best_single = b
                
                if best_single is not None:
                    # Remove both, add one
                    selected_list.remove(bj)
                    selected_list.remove(bi)
                    for s in block_covers[bi]:
                        counts[s] -= 1
                    for s in block_covers[bj]:
                        counts[s] -= 1
                    selected_list.append(best_single)
                    for s in block_covers[best_single]:
                        counts[s] += 1
                    improved = True
                    break
            if improved:
                break
        
        # Try removing one block and adding two
        if not improved and _deadline_ok(deadline):
            n_sel = len(selected_list)
            for i in range(n_sel):
                if not _deadline_ok(deadline):
                    break
                
                bi = selected_list[i]
                
                uncovered_if_removed = []
                for s in block_covers[bi]:
                    if counts[s] <= 1:
                        uncovered_if_removed.append(s)
                
                if not uncovered_if_removed:
                    # Redundant
                    selected_list.remove(bi)
                    for s in block_covers[bi]:
                        counts[s] -= 1
                    improved = True
                    break
                
                # Try to find two replacements
                # Get candidates from most constrained uncovered subset
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
                
                # Try pairs of candidates
                best_pair = None
                best_gain = -1
                cand_list = [b for b in candidates if b not in selected_list]
                
                for idx1 in range(len(cand_list)):
                    if not _deadline_ok(deadline):
                        break
                    b1 = cand_list[idx1]
                    
                    # Check if b1 covers some uncovered
                    covers_b1 = set()
                    for s in block_covers[b1]:
                        if s in uncovered_if_removed:
                            covers_b1.add(s)
                    
                    remaining_uncovered = [s for s in uncovered_if_removed if s not in covers_b1]
                    
                    if not remaining_uncovered:
                        # b1 alone covers all
                        gain = 0
                        for s in block_covers[b1]:
                            if counts[s] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (b1, None)
                        continue
                    
                    # Find b2 that covers remaining
                    for idx2 in range(idx1+1, len(cand_list)):
                        b2 = cand_list[idx2]
                        
                        covers_all = True
                        for s in remaining_uncovered:
                            if s not in block_covers[b2]:
                                covers_all = False
                                break
                        
                        if not covers_all:
                            continue
                        
                        gain = 0
                        for s in block_covers[b1]:
                            if counts[s] == 0:
                                gain += 1
                        for s in block_covers[b2]:
                            if counts[s] == 0:
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (b1, b2)
                    
                    if best_pair and best_pair[1] is not None:
                        break
                
                if best_pair is not None:
                    selected_list.remove(bi)
                    for s in block_covers[bi]:
                        counts[s] -= 1
                    if best_pair[0]:
                        selected_list.append(best_pair[0])
                        for s in block_covers[best_pair[0]]:
                            counts[s] += 1
                    if best_pair[1]:
                        selected_list.append(best_pair[1])
                        for s in block_covers[best_pair[1]]:
                            counts[s] += 1
                    improved = True
                    break
    
    return set(selected_list)

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
    
    # Apply 2-exchange local search
    best_selected = _local_search_2exchange(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=50)
    
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

def _greedy_cover_bitmask(block_covers_bitmasks, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover using bitmasks for speed."""
    uncovered_mask = (1 << n_t) - 1
    selected = set()
    
    while uncovered_mask:
        if not _deadline_ok(deadline):
            break
        
        # Find the uncovered subset with fewest candidate blocks
        min_count = float('inf')
        target_sub = -1
        for s in range(n_t):
            if (uncovered_mask >> s) & 1:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
        
        if target_sub == -1:
            break
        
        candidates = sub_to_blocks[target_sub]
        
        # Calculate gains for all candidates using bitmask
        gains = []
        for b in candidates:
            gain = _popcount(block_covers_bitmasks[b] & uncovered_mask)
            gains.append((gain, b))
        
        # Sort by gain descending
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        uncovered_mask &= ~block_covers_bitmasks[best_block]
    
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
    block_covers_bitmasks = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            mask |= (1 << idx)
        blocks.append(block)
        block_covers.append(covers)
        block_covers_bitmasks.append(mask)
        
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
        
        # Use bitmask greedy for speed
        selected = _greedy_cover_bitmask(block_covers_bitmasks, n_t, sub_to_blocks, deadline, 
                                          randomize=(restart > 0), top_k=10)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search (includes 2-exchange)
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
        node_count = [0]
        max_nodes = 5000 # Safety limit
        
        def _bndb(uncovered_mask, best_limit):
            if not _deadline_ok(deadline) or node_count[0] > max_nodes:
                return None
            node_count[0] += 1
            
            if uncovered_mask == 0:
                return set()
                
            # Find the uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in range(n_t):
                if (uncovered_mask >> s) & 1:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                return None
                
            # Calculate max coverage among candidates
            max_cov = 0
            for b in sub_to_blocks[target_sub]:
                cov = _popcount(block_covers_bitmasks[b] & uncovered_mask)
                if cov > max_cov:
                    max_cov = cov
                    
            if max_cov == 0:
                return None
                
            lb = (_popcount(uncovered_mask) + max_cov - 1) // max_cov
            
            if lb >= best_limit:
                return None
                
            candidates = sub_to_blocks[target_sub]
            
            sorted_candidates = []
            for b in candidates:
                cov = _popcount(block_covers_bitmasks[b] & uncovered_mask)
                sorted_candidates.append((cov, b))
            sorted_candidates.sort(reverse=True)
            
            for cov, b in sorted_candidates:
                if not _deadline_ok(deadline) or node_count[0] > max_nodes:
                    return None
                    
                new_uncovered_mask = uncovered_mask & ~block_covers_bitmasks[b]
                
                result = _bndb(new_uncovered_mask, best_limit - 1)
                
                if result is not None:
                    result.add(b)
                    return result
                    
            return None
            
        bnb_result = _bndb((1 << n_t) - 1, best_size)
        
        if bnb_result is not None and len(bnb_result) < best_size:
            best_selected = bnb_result
            best_size = len(best_selected)
            
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
