"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _subsets_to_ints(v, k, t):
    """Precompute all t-subsets as frozensets and map them to indices."""
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    return all_subsets, subset_to_idx


def _block_mask(block, subset_to_idx, v, k, t):
    """Return the set of t-subset indices covered by a block."""
    covered = set()
    for c in itertools.combinations(block, t):
        idx = subset_to_idx[frozenset(c)]
        covered.add(idx)
    return covered


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(all_subsets)))
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        block_set_local = set(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set_local.add(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                uncovered.discard(idx)
                
    return blocks


def _compute_block_gain(block, counts, subset_to_idx, v, k, t, weight_func=None):
    """Compute weighted gain of adding a block to current covering."""
    if weight_func is None:
        weight_func = lambda idx: 1.0
    gain = 0.0
    for c in itertools.combinations(block, t):
        fs = frozenset(c)
        if fs in subset_to_idx:
            idx = subset_to_idx[fs]
            if counts[idx] == 0:
                gain += 10.0 * weight_func(idx)
            elif counts[idx] == 1:
                gain += 1.0 * weight_func(idx)
    return gain


def _primal_dual_weights(counts, n_subsets):
    """Compute primal-dual-inspired weights: higher for less-covered subsets."""
    weights = [1.0] * n_subsets
    for i in range(n_subsets):
        if counts[i] == 0:
            weights[i] = 10.0
        elif counts[i] == 1:
            weights[i] = 2.0
        elif counts[i] == 2:
            weights[i] = 1.0
        else:
            weights[i] = 0.5
    return weights


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute
    all_subsets, subset_to_idx = _subsets_to_ints(v, k, t)
    n_subsets = len(all_subsets)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx)
    
    # Precompute covered subsets for each block
    block_covered = [_block_mask(b, subset_to_idx, v, k, t) for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Simulated Annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    iteration = 0
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block (try multiple times per iteration)
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            # Check if all subsets in this block have count > 1
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block with primal-dual-guided gain
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]
            
            # Identify critical subsets (count == 1) in old_block
            critical = [idx for idx in old_covered if counts[idx] == 1]
            
            # Collect points needed for critical subsets
            points_needed = set()
            for idx in critical:
                points_needed.update(all_subsets[idx])
            
            # If we can't fit all critical points in one block, skip
            if len(points_needed) > k:
                continue
            
            # Build new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Compute primal-dual weights for guidance
            pd_weights = _primal_dual_weights(counts, n_subsets)
            
            # Greedy fill to maximize primal-dual weighted gain
            for _ in range(k - len(new_block_set)):
                if not cands:
                    break
                best_x = None
                best_gain = -1.0
                for x in cands:
                    # Compute gain of adding x to new_block_set
                    gain = 0.0
                    current = list(new_block_set)
                    for rest in itertools.combinations(current, t-1):
                        fs = frozenset(list(rest) + [x])
                        if fs in subset_to_idx:
                            idx = subset_to_idx[fs]
                            if counts[idx] == 0:
                                gain += 10.0 * pd_weights[idx]
                            elif counts[idx] == 1:
                                gain += 1.0 * pd_weights[idx]
                    # Small random perturbation to avoid ties
                    gain += rng.random() * 0.01
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_block_set.add(best_x)
                cands.remove(best_x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                
                if new_block == old_block:
                    continue
                
                new_covered = _block_mask(new_block, subset_to_idx, v, k, t)
                
                # Check if swap is valid: no subset loses its last cover
                valid = True
                for idx in old_covered:
                    if idx not in new_covered and counts[idx] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Compute net improvement (primal-dual style)
                    net_gain = 0.0
                    for idx in new_covered - old_covered:
                        if counts[idx] == 0:
                            net_gain += 10.0
                        elif counts[idx] == 1:
                            net_gain += 1.0
                    for idx in old_covered - new_covered:
                        if counts[idx] == 1:
                            net_gain -= 5.0  # penalty for potentially uncovering
                    
                    # Accept with simulated annealing probability
                    if net_gain >= 0 or rng.random() < math.exp(net_gain / T):
                        # Apply swap
                        for idx in old_covered:
                            counts[idx] -= 1
                        for idx in new_covered:
                            counts[idx] += 1
                        
                        blocks[i] = new_block
                        block_covered[i] = new_covered
        
        # 3. Try to merge two blocks into one (replace 2 blocks with 1) - increased probability
        if len(blocks) >= 2 and rng.random() < 0.25:
            i, j = rng.sample(range(len(blocks)), 2)
            # Try to find a block that covers all critical subsets from both
            crit1 = [idx for idx in block_covered[i] if counts[idx] == 1]
            crit2 = [idx for idx in block_covered[j] if counts[idx] == 1]
            
            points_needed = set()
            for idx in crit1 + crit2:
                points_needed.update(all_subsets[idx])
            
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                pd_weights = _primal_dual_weights(counts, n_subsets)
                
                for _ in range(k - len(new_block_set)):
                    if not cands:
                        break
                    best_x = None
                    best_gain = -1.0
                    for x in cands:
                        gain = 0.0
                        current = list(new_block_set)
                        for rest in itertools.combinations(current, t-1):
                            fs = frozenset(list(rest) + [x])
                            if fs in subset_to_idx:
                                idx = subset_to_idx[fs]
                                if counts[idx] == 0:
                                    gain += 10.0 * pd_weights[idx]
                                elif counts[idx] == 1:
                                    gain += 1.0 * pd_weights[idx]
                        gain += rng.random() * 0.01
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    new_block_set.add(best_x)
                    cands.remove(best_x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    new_covered = _block_mask(new_block, subset_to_idx, v, k, t)
                    
                    # Check if this single block covers all critical subsets from both
                    valid = True
                    for idx in crit1 + crit2:
                        if idx not in new_covered:
                            valid = False
                            break
                    
                    if valid:
                        # Remove both blocks and add the new one
                        for idx in block_covered[i]:
                            counts[idx] -= 1
                        for idx in block_covered[j]:
                            counts[idx] -= 1
                        for idx in new_covered:
                            counts[idx] += 1
                        
                        # Remove blocks i and j (careful with indices)
                        if i < j:
                            blocks.pop(j)
                            block_covered.pop(j)
                            blocks.pop(i)
                            block_covered.pop(i)
                        else:
                            blocks.pop(i)
                            block_covered.pop(i)
                            blocks.pop(j)
                            block_covered.pop(j)
                        
                        blocks.append(new_block)
                        block_covered.append(new_covered)
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                            no_improve = 0
        
        # 4. Random perturbation if stuck
        if no_improve > max_no_improve:
            # Try a new greedy cover with different seed
            new_rng = random.Random(rng.randint(0, 10**9))
            new_cover = _greedy_cover(v, k, t, new_rng, all_subsets, subset_to_idx)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                block_covered = [_block_mask(b, subset_to_idx, v, k, t) for b in blocks]
                counts = [0] * n_subsets
                for covered in block_covered:
                    for idx in covered:
                        counts[idx] += 1
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
            no_improve = 0
            T = max(T_min, T * 0.5)
        
        if len(blocks) < best_count:
            no_improve = 0
        else:
            no_improve += 1
        
        T = max(T_min, T * cooling)
    
    # Final polish: exhaustive removal pass
    for _ in range(10):
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                changed = True
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        if not changed:
            break

    return best_blocks
# EVOLVE-BLOCK-END
