"""
Hybrid covering design solver: efficient greedy + simulated annealing refinement.
Combines fast NumPy-based greedy construction with SA-based block swaps/removals.
"""
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    """Construct a covering design C(v,k,t) minimizing the number of blocks."""
    deadline = time.time() + 110  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point, precompute which t-subsets contain it (as a set of indices)
    point_tsubs = [set() for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for pt in ts:
            point_tsubs[pt].add(i)
    
    # For a block (k-subset), compute the set of t-subset indices it covers
    def block_cover(block):
        """Return a frozenset of t-subset indices covered by this block."""
        cov = set()
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[c]
            cov.add(idx)
        return frozenset(cov)
    
    # Greedy construction
    def greedy_construct():
        """Fast greedy set cover with randomization."""
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks = []
        block_covs = []
        
        while uncovered.any() and time.time() < deadline - 15:
            # Pick a random uncovered t-subset as seed
            unc_idx = np.where(uncovered)[0]
            seed_idx = rng.choice(unc_idx)
            seed = all_tsubsets[seed_idx]
            block = list(seed)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage of uncovered subsets
            while len(block) < k:
                best_x = None
                best_gain = 0
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    # Compute gain: number of newly covered t-subsets
                    # A t-subset (a1,...,a_{t-1},x) is covered if {a1,...,a_{t-1}} subset of block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            b_tuple = tuple(block)
            cov = block_cover(block)
            
            # Only add if it covers at least one new subset
            if any(uncovered[i] for i in cov):
                blocks.append(block)
                block_covs.append(cov)
                for idx in cov:
                    uncovered[idx] = False
        
        return blocks, block_covs
    
    # Build initial covering
    blocks, block_covs = greedy_construct()
    
    # If we didn't finish, try a deterministic fallback to ensure validity
    if time.time() < deadline - 5:
        uncovered = np.ones(n_tsubsets, dtype=bool)
        for cov in block_covs:
            for idx in cov:
                uncovered[idx] = False
        while uncovered.any():
            unc_idx = np.where(uncovered)[0]
            seed = all_tsubsets[unc_idx[0]]
            block = list(seed)
            # Complete block greedily
            while len(block) < k:
                best_x = None
                best_gain = 0
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            cov = block_cover(block)
            blocks.append(block)
            block_covs.append(cov)
            for idx in cov:
                uncovered[idx] = False
    
    # Precompute coverage counts
    counts = np.zeros(n_tsubsets, dtype=np.int32)
    for cov in block_covs:
        for idx in cov:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Simulated Annealing refinement
    T = 30.0
    T_min = 0.05
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 5 and T > T_min:
        T *= cooling
        no_improve += 1
        
        # Phase 1: Try to remove redundant blocks
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covs[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covs[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covs.pop(i)
                removed = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                break
        
        if removed:
            continue
        
        # Phase 2: Try to swap a block for a better one
        if len(blocks) == 0:
            break
        
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        old_cov = block_covs[i]
        
        # Identify critical subsets (count == 1) in old block
        critical = [idx for idx in old_cov if counts[idx] == 1]
        points_needed = set()
        for idx in critical:
            points_needed.update(all_tsubsets[idx])
        
        if len(points_needed) > k:
            continue
        
        # Build new block containing required points
        new_block_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        # Greedy fill
        for x in cands:
            if len(new_block_set) >= k:
                break
            # Compute gain: prefer covering low-count subsets
            gain = 0
            for rest in itertools.combinations(new_block_set, t - 1):
                ts = tuple(sorted(rest + (x,)))
                if ts in tsub_to_idx:
                    idx = tsub_to_idx[ts]
                    if counts[idx] == 0:
                        gain += 100
                    elif counts[idx] == 1:
                        gain += 10
                    else:
                        gain += 1
            if gain > 0:
                new_block_set.add(x)
            elif len(new_block_set) < k and rng.random() < 0.3:
                new_block_set.add(x)
        
        if len(new_block_set) < k:
            # Fill remaining randomly
            remaining = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(remaining)
            for x in remaining:
                if len(new_block_set) >= k:
                    break
                new_block_set.add(x)
        
        new_block = tuple(sorted(new_block_set))
        if new_block == tuple(old_block):
            continue
        
        new_cov = block_cover(new_block)
        
        # Check validity: no subset loses its last cover
        valid = True
        for idx in old_cov:
            if idx not in new_cov and counts[idx] <= 1:
                valid = False
                break
        
        if valid:
            # Compute energy change
            delta_e = 0
            for idx in old_cov:
                if idx not in new_cov:
                    if counts[idx] == 1:
                        delta_e += 1000
                    else:
                        delta_e -= 1
            for idx in new_cov:
                if idx not in old_cov:
                    if counts[idx] == 0:
                        delta_e -= 1000
                    else:
                        delta_e += 1
            
            # Accept with probability based on temperature
            if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                # Apply swap
                for idx in old_cov:
                    counts[idx] -= 1
                for idx in new_cov:
                    counts[idx] += 1
                blocks[i] = list(new_block)
                block_covs[i] = new_cov
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
        
        # Phase 3: Occasionally try to merge/replace two blocks with one
        if len(blocks) >= 2 and rng.random() < 0.1:
            j = rng.randrange(len(blocks))
            if j == i:
                j = (j + 1) % len(blocks)
            
            # Try to find a single block that covers all critical subsets of both
            crit1 = [idx for idx in block_covs[i] if counts[idx] <= 1]
            crit2 = [idx for idx in block_covs[j] if counts[idx] <= 1]
            pts = set()
            for idx in crit1 + crit2:
                pts.update(all_tsubsets[idx])
            
            if len(pts) <= k:
                new_block_set = set(pts)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    new_block_set.add(x)
                
                if len(new_block_set) == k:
                    nb = tuple(sorted(new_block_set))
                    ncov = block_cover(nb)
                    
                    # Check if this single block covers all critical subsets of both
                    valid_merge = True
                    for idx in crit1:
                        if idx not in ncov:
                            valid_merge = False
                            break
                    if valid_merge:
                        for idx in crit2:
                            if idx not in ncov:
                                valid_merge = False
                                break
                    
                    if valid_merge:
                        # Remove both blocks and add new one
                        for idx in block_covs[i]:
                            counts[idx] -= 1
                        for idx in block_covs[j]:
                            counts[idx] -= 1
                        for idx in ncov:
                            counts[idx] += 1
                        
                        # Rebuild blocks list
                        new_blocks = []
                        new_covs = []
                        for idx_b, (b, c) in enumerate(zip(blocks, block_covs)):
                            if idx_b != i and idx_b != j:
                                new_blocks.append(b)
                                new_covs.append(c)
                        new_blocks.append(list(nb))
                        new_covs.append(ncov)
                        
                        blocks = new_blocks
                        block_covs = new_covs
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                            no_improve = 0
                        break
    
    # Final redundancy removal pass
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_blocks = []
            new_covs = []
            for i in range(len(blocks)):
                redundant = True
                for idx in block_covs[i]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_covs[i]:
                        counts[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(blocks[i])
                    new_covs.append(block_covs[i])
            blocks = new_blocks
            block_covs = new_covs
    
    # Return the best solution found
    return best_blocks
