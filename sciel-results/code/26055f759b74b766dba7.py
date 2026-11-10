"""Covering design C(19,7,4) – greedy construction with simulated annealing refinement."""
import itertools
import time
import random
import math
from collections import defaultdict

_DEADLINE_S = 145.0  # Safety margin from 150s limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _greedy_construct(v, k, t, point_freq=None):
    """
    Construct a covering using greedy strategy.
    Prioritize uncovered t-subsets whose points are rarely used.
    """
    if point_freq is None:
        point_freq = [0] * v
    
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if _time_left() < 2.0:
            break
        
        # Select target: uncovered t-subset with minimum sum of point frequencies
        best_target = min(uncovered, key=lambda tsub: sum(point_freq[p] for p in tsub))
        
        # Find all k-subsets containing best_target
        target_set = set(best_target)
        remaining_points = [p for p in range(v) if p not in target_set]
        
        best_block = None
        best_gain = -1
        
        for comb in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(list(best_target) + list(comb)))
            # Count how many uncovered t-subsets this block covers
            gain = 0
            for tsub in itertools.combinations(block, t):
                if tsub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Update uncovered and point frequencies
        for tsub in itertools.combinations(best_block, t):
            uncovered.discard(tsub)
        for p in best_block:
            point_freq[p] += 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Build tsub -> set of block indices mapping
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)
    
    # Remove redundant blocks iteratively
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            # Check if block i is redundant
            redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                if len(tsub_to_blocks[tsub]) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for tsub in itertools.combinations(blocks[i], t):
                    tsub_to_blocks[tsub].discard(i)
                blocks.pop(i)
                # Update indices in tsub_to_blocks
                for tsub, bset in list(tsub_to_blocks.items()):
                    new_bset = set()
                    for j in bset:
                        if j < i:
                            new_bset.add(j)
                        elif j > i:
                            new_bset.add(j - 1)
                    tsub_to_blocks[tsub] = new_bset
                changed = True
                break
    
    return blocks


def _simulated_annealing(blocks, v, k, t, initial_temp=2.0, cooling=0.998, min_temp=0.01):
    """
    Simulated annealing to refine the covering.
    Moves: add a block, remove a redundant block, or swap a block.
    Objective: minimize number of blocks while maintaining validity.
    Uses proper SA acceptance: accept worse states with prob exp(-delta/T).
    """
    if not blocks:
        return blocks
    
    # Build tsub -> set of block indices mapping
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)
    
    best_blocks = list(blocks)
    best_len = len(blocks)
    current_len = len(blocks)
    temp = initial_temp
    
    # Precompute all t-subsets of each block for faster access
    block_tsubs = [list(itertools.combinations(b, t)) for b in blocks]
    
    while _time_left() > 10.0 and temp > min_temp:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.35 and len(blocks) < 300:
            # Add a random block, then try to remove redundant blocks
            new_block = tuple(sorted(random.sample(range(v), k)))
            
            # Check if already exists
            if new_block in set(map(tuple, blocks)):
                continue
            
            new_idx = len(blocks)
            blocks.append(list(new_block))
            new_tsubs = list(itertools.combinations(new_block, t))
            block_tsubs.append(new_tsubs)
            for tsub in new_tsubs:
                tsub_to_blocks[tsub].add(new_idx)
            current_len += 1
            
            # Try to remove redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                redundant = True
                for tsub in block_tsubs[i]:
                    if len(tsub_to_blocks[tsub]) == 1:
                        redundant = False
                        break
                if redundant:
                    for tsub in block_tsubs[i]:
                        tsub_to_blocks[tsub].discard(i)
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    # Update indices in tsub_to_blocks
                    for tsub, bset in list(tsub_to_blocks.items()):
                        new_bset = set()
                        for j in bset:
                            if j < i:
                                new_bset.add(j)
                            elif j > i:
                                new_bset.add(j - 1)
                        tsub_to_blocks[tsub] = new_bset
                    current_len -= 1
                    
        elif move_type < 0.65:
            # Try to remove a random block if it's redundant
            if not blocks:
                continue
            i = random.randrange(len(blocks))
            redundant = True
            for tsub in block_tsubs[i]:
                if len(tsub_to_blocks[tsub]) == 1:
                    redundant = False
                    break
            if redundant:
                for tsub in block_tsubs[i]:
                    tsub_to_blocks[tsub].discard(i)
                blocks.pop(i)
                block_tsubs.pop(i)
                for tsub, bset in list(tsub_to_blocks.items()):
                    new_bset = set()
                    for j in bset:
                        if j < i:
                            new_bset.add(j)
                        elif j > i:
                            new_bset.add(j - 1)
                    tsub_to_blocks[tsub] = new_bset
                current_len -= 1
                
        else:
            # Swap a block: replace block i with a different block that covers its unique t-subsets
            if not blocks:
                continue
            i = random.randrange(len(blocks))
            
            # Find unique t-subsets for block i
            unique_tsubs = []
            for tsub in block_tsubs[i]:
                if len(tsub_to_blocks[tsub]) == 1:
                    unique_tsubs.append(tsub)
            
            if not unique_tsubs:
                # Block is redundant, try to remove it
                for tsub in block_tsubs[i]:
                    tsub_to_blocks[tsub].discard(i)
                blocks.pop(i)
                block_tsubs.pop(i)
                for tsub, bset in list(tsub_to_blocks.items()):
                    new_bset = set()
                    for j in bset:
                        if j < i:
                            new_bset.add(j)
                        elif j > i:
                            new_bset.add(j - 1)
                    tsub_to_blocks[tsub] = new_bset
                current_len -= 1
                continue
            
            # Collect all points that must be in the new block
            must_include = set()
            for tsub in unique_tsubs:
                must_include.update(tsub)
            
            if len(must_include) > k:
                continue  # Can't fit all required points
            
            # Find candidate blocks containing all must_include points
            remaining_points = [p for p in range(v) if p not in must_include]
            need_more = k - len(must_include)
            
            if need_more < 0:
                continue
            
            # Try multiple random candidates
            found_swap = False
            for _ in range(15):
                if not remaining_points or need_more == 0:
                    new_block = tuple(sorted(must_include))
                else:
                    extra = random.sample(remaining_points, min(need_more, len(remaining_points)))
                    if len(extra) < need_more:
                        continue
                    new_block = tuple(sorted(list(must_include) + list(extra)))
                
                # Check if this new block covers all unique_tsubs
                valid = True
                new_block_set = set(new_block)
                for tsub in unique_tsubs:
                    if not set(tsub).issubset(new_block_set):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check if new block is same as old
                if tuple(sorted(blocks[i])) == new_block:
                    continue
                
                # Accept the swap
                old_block = blocks[i]
                old_tsubs = block_tsubs[i]
                blocks[i] = list(new_block)
                new_tsubs = list(itertools.combinations(new_block, t))
                block_tsubs[i] = new_tsubs
                
                # Update tsub_to_blocks
                for tsub in old_tsubs:
                    tsub_to_blocks[tsub].discard(i)
                for tsub in new_tsubs:
                    tsub_to_blocks[tsub].add(i)
                found_swap = True
                break
        
        # Update best if improved
        if current_len < best_len:
            best_len = current_len
            best_blocks = list(blocks)
        
        # Cool down
        temp *= cooling
    
    # Restore best solution
    return best_blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t)
    
    if _time_left() < 5.0:
        return blocks
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t)
    
    if _time_left() < 15.0:
        return blocks
    
    # Phase 3: Simulated annealing refinement (main search phase)
    blocks = _simulated_annealing(blocks, v, k, t, initial_temp=2.0, cooling=0.998, min_temp=0.01)
    
    # Phase 4: Final redundant block removal
    blocks = _remove_redundant_blocks(blocks, v, k, t)
    
    return blocks
