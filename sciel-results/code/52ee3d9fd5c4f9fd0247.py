"""Covering design C(19,7,4) - bitset-based greedy + local search + simulated annealing."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def _build_index(v, k, t):
    """Precompute t-subset index and bitsets for fast coverage tracking."""
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    return all_t_subsets, n_t, t_subset_index

def _block_t_subsets(block, t):
    """Yield indices of all t-subsets within a block."""
    return [t_subset_index[s] for s in itertools.combinations(block, t)]

def _greedy_construct(v, k, t, deadline):
    """Fast greedy construction using bitsets for uncovered tracking."""
    all_t_subsets, n_t, t_subset_index = _build_index(v, k, t)
    
    # Bitset of uncovered t-subsets (1 = uncovered)
    uncovered = np.ones(n_t, dtype=np.uint8)
    uncovered_count = n_t
    
    blocks = []
    block_set = set()
    
    while uncovered_count > 0:
        if time.time() > deadline - 10.0:
            break
        
        # Find an uncovered t-subset to center the next block
        unc_idx = np.where(uncovered == 1)[0]
        if len(unc_idx) == 0:
            break
        first_idx = unc_idx[0]
        base = list(all_t_subsets[first_idx])
        
        # Greedily build a block of size k around the base
        block = set(base)
        while len(block) < k:
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in block:
                    continue
                # Count newly covered t-subsets
                new_block = list(block) + [x]
                new_block.sort()
                gain = 0
                for s in itertools.combinations(new_block, t):
                    idx = t_subset_index.get(s)
                    if idx is not None and uncovered[idx] == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.add(best_point)
        
        block = sorted(block)
        key = tuple(block)
        if key not in block_set:
            block_set.add(key)
            blocks.append(block)
            # Update uncovered
            for s in itertools.combinations(block, t):
                idx = t_subset_index[s]
                if uncovered[idx] == 1:
                    uncovered[idx] = 0
                    uncovered_count -= 1
    
    return blocks, uncovered, uncovered_count, all_t_subsets, n_t, t_subset_index

def _improve_blocks(blocks, v, k, t, all_t_subsets, n_t, t_subset_index, deadline):
    """Local search: remove redundant blocks, try swaps."""
    # Build coverage count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for block in blocks:
        for s in itertools.combinations(block, t):
            idx = t_subset_index[s]
            cover_count[idx] += 1
    
    uncovered = np.where(cover_count == 0)[0]
    uncovered_set = set(uncovered.tolist())
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Pass 1: Remove redundant blocks
        for i in range(len(blocks)):
            if time.time() > deadline - 5.0:
                break
            block = blocks[i]
            essential = False
            for s in itertools.combinations(block, t):
                idx = t_subset_index[s]
                if cover_count[idx] == 1:
                    essential = True
                    break
            if not essential:
                for s in itertools.combinations(block, t):
                    idx = t_subset_index[s]
                    cover_count[idx] -= 1
                blocks.pop(i)
                improved = True
                break
        
        if time.time() > deadline - 5.0:
            break
        
        # Pass 2: Try to replace a block with a better one
        if uncovered_set:
            # Find a block to potentially improve
            rng = random.Random(int(time.time() * 1000))
            unc_list = list(uncovered_set)
            rng.shuffle(unc_list)
            
            for idx in unc_list[:30]:
                if time.time() > deadline - 5.0:
                    break
                base = list(all_t_subsets[idx])
                # Build a candidate block
                block = set(base)
                while len(block) < k:
                    best_gain = -1
                    best_point = -1
                    for x in range(v):
                        if x in block:
                            continue
                        new_block = list(block) + [x]
                        new_block.sort()
                        gain = 0
                        for s in itertools.combinations(new_block, t):
                            si = t_subset_index.get(s)
                            if si is not None and cover_count[si] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    block.add(best_point)
                
                candidate = sorted(block)
                # Check if this candidate covers more uncovered than any existing block
                cand_gain = 0
                for s in itertools.combinations(candidate, t):
                    si = t_subset_index[s]
                    if cover_count[si] == 0:
                        cand_gain += 1
                
                if cand_gain > 0:
                    # Find the least useful existing block to replace
                    min_use = float('inf')
                    min_idx = -1
                    for i, b in enumerate(blocks):
                        use = 0
                        for s in itertools.combinations(b, t):
                            si = t_subset_index[s]
                            if cover_count[si] == 1:
                                use += 1
                        if use < min_use:
                            min_use = use
                            min_idx = i
                    
                    if min_idx >= 0 and min_use < cand_gain:
                        # Replace
                        old_block = blocks[min_idx]
                        for s in itertools.combinations(old_block, t):
                            si = t_subset_index[s]
                            cover_count[si] -= 1
                        blocks[min_idx] = candidate
                        for s in itertools.combinations(candidate, t):
                            si = t_subset_index[s]
                            cover_count[si] += 1
                            if cover_count[si] == 1:
                                uncovered_set.discard(si)
                        improved = True
                        break
    
    return blocks, cover_count, uncovered_set

def _simulated_annealing(blocks, v, k, t, all_t_subsets, n_t, t_subset_index, cover_count, uncovered_set, deadline):
    """Simulated annealing to escape local optima."""
    rng = random.Random(42)
    current_blocks = list(blocks)
    current_uncovered = set(uncovered_set)
    best_blocks = list(current_blocks)
    best_uncovered = set(current_uncovered)
    best_count = len(best_blocks)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min and time.time() < deadline - 2.0:
        # Generate a neighbor
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try adding a random block
            if current_uncovered:
                unc_idx = rng.choice(list(current_uncovered))
                base = list(all_t_subsets[unc_idx])
                block = set(base)
                while len(block) < k:
                    x = rng.randint(0, v - 1)
                    if x not in block:
                        block.add(x)
                new_block = sorted(block)
                if tuple(new_block) not in [tuple(b) for b in current_blocks]:
                    new_blocks = current_blocks + [new_block]
                    new_uncovered = set(current_uncovered)
                    for s in itertools.combinations(new_block, t):
                        si = t_subset_index[s]
                        if cover_count[si] == 0:
                            new_uncovered.discard(si)
                    delta = len(new_uncovered) - len(current_uncovered)
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        current_blocks = new_blocks
                        current_uncovered = new_uncovered
                        if len(current_uncovered) < len(best_uncovered):
                            best_uncovered = set(current_uncovered)
                            best_blocks = list(current_blocks)
                            best_count = len(best_blocks)
        elif move_type < 0.7:
            # Try removing a block
            if len(current_blocks) > 1:
                i = rng.randint(0, len(current_blocks) - 1)
                block = current_blocks[i]
                # Check if removable
                removable = True
                for s in itertools.combinations(block, t):
                    si = t_subset_index[s]
                    if cover_count[si] == 1:
                        removable = False
                        break
                if removable:
                    new_blocks = current_blocks[:i] + current_blocks[i+1:]
                    current_blocks = new_blocks
                    if len(current_uncovered) < len(best_uncovered):
                        best_uncovered = set(current_uncovered)
                        best_blocks = list(current_blocks)
                        best_count = len(best_blocks)
        else:
            # Try swapping a point in a block
            if current_blocks:
                i = rng.randint(0, len(current_blocks) - 1)
                block = list(current_blocks[i])
                j = rng.randint(0, k - 1)
                old_point = block[j]
                new_point = rng.randint(0, v - 1)
                if new_point != old_point and new_point not in block:
                    block[j] = new_point
                    block.sort()
                    new_block = block
                    # Compute delta
                    old_covered = set()
                    for s in itertools.combinations(current_blocks[i], t):
                        old_covered.add(t_subset_index[s])
                    new_covered = set()
                    for s in itertools.combinations(new_block, t):
                        new_covered.add(t_subset_index[s])
                    lost = old_covered - new_covered
                    gained = new_covered - old_covered
                    delta = 0
                    for si in lost:
                        if cover_count[si] == 1:
                            delta += 1
                    for si in gained:
                        if cover_count[si] == 0:
                            delta -= 1
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        # Apply move
                        current_blocks[i] = new_block
                        for si in lost:
                            cover_count[si] -= 1
                            if cover_count[si] == 0:
                                current_uncovered.add(si)
                        for si in gained:
                            cover_count[si] += 1
                            if cover_count[si] == 1:
                                current_uncovered.discard(si)
                        if len(current_uncovered) < len(best_uncovered):
                            best_uncovered = set(current_uncovered)
                            best_blocks = list(current_blocks)
                            best_count = len(best_blocks)
        
        T *= cooling
    
    return best_blocks, best_uncovered

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Phase 1: Greedy construction
    blocks, uncovered, uncovered_count, all_t_subsets, n_t, t_subset_index = _greedy_construct(v, k, t, deadline)
    
    if time.time() > deadline - 15.0:
        return blocks
    
    # Phase 2: Local search improvement
    blocks, cover_count, uncovered_set = _improve_blocks(blocks, v, k, t, all_t_subsets, n_t, t_subset_index, deadline)
    
    if time.time() > deadline - 10.0:
        return blocks
    
    # Phase 3: Simulated annealing
    blocks, uncovered_set = _simulated_annealing(blocks, v, k, t, all_t_subsets, n_t, t_subset_index, cover_count, uncovered_set, deadline)
    
    # Final cleanup: remove redundant blocks
    cover_count = np.zeros(n_t, dtype=np.int32)
    for block in blocks:
        for s in itertools.combinations(block, t):
            idx = t_subset_index[s]
            cover_count[idx] += 1
    
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                break
            block = blocks[i]
            essential = False
            for s in itertools.combinations(block, t):
                idx = t_subset_index[s]
                if cover_count[idx] == 1:
                    essential = True
                    break
            if not essential:
                for s in itertools.combinations(block, t):
                    idx = t_subset_index[s]
                    cover_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break
    
    # Deduplicate
    seen = set()
    unique_blocks = []
    for block in blocks:
        key = tuple(block)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)
    
    return unique_blocks
