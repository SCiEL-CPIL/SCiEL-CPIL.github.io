"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 138.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each point, which t-subsets contain it
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        for p in ts:
            point_tsubs[p].append(tsub_to_idx[ts])
    
    # Precompute t-subsets for each k-block candidate? Too many.
    # Instead, compute on the fly but cache.
    
    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    def greedy_cover():
        """Deterministic greedy: pick block covering most uncovered t-subsets."""
        coverage = np.zeros(n_tsubs, dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() > deadline:
                return None
            
            # Find first uncovered t-subset
            uncovered_mask = coverage == 0
            if not uncovered_mask.any():
                break
            
            # Pick the uncovered t-subset that appears in the most potential blocks
            # Actually, just pick the first one for speed
            uncovered_idx = np.argmax(uncovered_mask)
            seed_tsub = all_tsubs[uncovered_idx]
            block_set = set(seed_tsub)
            
            # Greedily add points to maximize new coverage
            for _ in range(k - t):
                if time.time() > deadline:
                    return None
                
                best_point = None
                best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    # Count new t-subsets covered
                    gain = 0
                    for tsub_idx in point_tsubs[p]:
                        if coverage[tsub_idx] == 0:
                            ts = all_tsubs[tsub_idx]
                            others = [x for x in ts if x != p]
                            if all(o in block_set for o in others):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point is not None:
                    block_set.add(best_point)
                else:
                    # Add any unused point
                    for p in range(v):
                        if p not in block_set:
                            block_set.add(p)
                            break
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                coverage[tsub_to_idx[ts]] += 1
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if not blocks:
            return blocks
        
        n_blocks = len(blocks)
        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs_list = []
        
        for block in blocks:
            tsubs = block_tsub_indices(block)
            block_tsubs_list.append(tsubs)
            for idx in tsubs:
                coverage_count[idx] += 1
        
        keep = [True] * n_blocks
        for i in range(n_blocks):
            if not keep[i]:
                continue
            can_remove = True
            for idx in block_tsubs_list[i]:
                if coverage_count[idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                keep[i] = False
                for idx in block_tsubs_list[i]:
                    coverage_count[idx] -= 1
        
        return [blocks[i] for i in range(n_blocks) if keep[i]]
    
    def walk_sat_polish(blocks, max_time):
        """WalkSAT-style local search to minimize block count while maintaining coverage.
        
        State: set of active blocks (indices into a pool).
        Objective: minimize (num_uncovered_tsubs * W + num_active_blocks)
        where W is large enough that covering all t-subsets is prioritized.
        
        Moves:
        - Remove an active block (if it doesn't uncover any t-subset, or with penalty)
        - Add a new block from the pool
        - Swap: remove one active, add one from pool
        """
        if not blocks:
            return blocks
        
        # Build a pool of candidate blocks: current blocks + some random ones
        pool = list(blocks)
        pool_set = set(tuple(b) for b in pool)
        
        # Add random candidate blocks to the pool
        n_random_candidates = min(50, max(0, 200 - len(pool)))
        for _ in range(n_random_candidates):
            if time.time() > deadline:
                break
            candidate = sorted(rng.sample(range(v), k))
            ct = tuple(candidate)
            if ct not in pool_set:
                pool.append(candidate)
                pool_set.add(ct)
        
        n_pool = len(pool)
        
        # Precompute t-subset indices for each pool block
        pool_tsubs = [block_tsub_indices(b) for b in pool]
        
        # Initial state: all original blocks active
        active = set(range(len(blocks)))
        
        # Compute coverage
        coverage = np.zeros(n_tsubs, dtype=np.int32)
        for i in active:
            for idx in pool_tsubs[i]:
                coverage[idx] += 1
        
        def objective(coverage, active_set):
            n_uncovered = int((coverage == 0).sum())
            return n_uncovered * 10000 + len(active_set)
        
        best_obj = objective(coverage, active)
        best_active = set(active)
        best_blocks = [pool[i] for i in sorted(best_active)]
        
        # WalkSAT loop
        p_remove = 0.4  # probability of trying to remove a block
        temperature = 1.0
        
        steps_since_improve = 0
        max_steps_no_improve = 500
        
        while time.time() < deadline and steps_since_improve < max_steps_no_improve:
            # Choose a move
            if len(active) == 0:
                # Must add something
                i = rng.randint(0, n_pool - 1)
                active.add(i)
                for idx in pool_tsubs[i]:
                    coverage[idx] += 1
            elif rng.random() < p_remove:
                # Try to remove a random active block
                candidates = list(active)
                i = rng.choice(candidates)
                # Check what would happen
                new_uncovered = 0
                for idx in pool_tsubs[i]:
                    if coverage[idx] == 1:
                        new_uncovered += 1
                
                # Simulate removal
                old_obj = objective(coverage, active)
                for idx in pool_tsubs[i]:
                    coverage[idx] -= 1
                active.discard(i)
                new_obj = objective(coverage, active)
                
                # Accept if better or with some probability
                delta = new_obj - old_obj
                if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 0.1)):
                    best_obj = min(best_obj, new_obj)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_active = set(active)
                        best_blocks = [pool[i] for i in sorted(best_active)]
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    # Reject: undo
                    active.add(i)
                    for idx in pool_tsubs[i]:
                        coverage[idx] += 1
                    steps_since_improve += 1
            else:
                # Try to add a random pool block
                i = rng.randint(0, n_pool - 1)
                if i in active:
                    steps_since_improve += 1
                    continue
                active.add(i)
                for idx in pool_tsubs[i]:
                    coverage[idx] += 1
                new_obj = objective(coverage, active)
                
                delta = new_obj - best_obj
                if delta <= 0:
                    best_obj = new_obj
                    best_active = set(active)
                    best_blocks = [pool[i] for i in sorted(best_active)]
                    steps_since_improve = 0
                else:
                    # This move made things worse (added a block); undo
                    active.discard(i)
                    for idx in pool_tsubs[i]:
                        coverage[idx] -= 1
                    steps_since_improve += 1
            
            # Anneal temperature
            temperature *= 0.999
            if temperature < 0.01:
                temperature = 0.01
        
        return best_blocks
    
    def local_search_swap(blocks):
        """Try to replace a block with another that covers its unique t-subsets better."""
        if not blocks or len(blocks) < 2:
            return blocks
        
        n_blocks = len(blocks)
        block_tsubs_list = [block_tsub_indices(b) for b in blocks]
        
        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        for tsubs in block_tsubs_list:
            for idx in tsubs:
                coverage_count[idx] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            for i in range(n_blocks):
                if time.time() > deadline:
                    break
                
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsubs_list[i] if coverage_count[idx] == 1]
                
                if not unique_tsubs:
                    continue
                
                # Try to find a replacement block from the pool
                best_j = -1
                best_new_unique = float('inf')
                
                for j in range(n_blocks):
                    if j == i:
                        continue
                    
                    # Check if block j covers all unique_tsubs of block i
                    block_j_set = set(blocks[j])
                    covers_all = True
                    for idx in unique_tsubs:
                        ts = all_tsubs[idx]
                        if not all(p in block_j_set for p in ts):
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Count how many of block j's t-subsets would become unique
                    # after removing block i
                    new_unique = 0
                    for idx in block_tsubs_list[j]:
                        if coverage_count[idx] == 1 and idx not in block_tsubs_list[i]:
                            new_unique += 1
                    for idx in block_tsubs_list[i]:
                        if coverage_count[idx] == 1:
                            # This one is covered by j, so it won't be unique after swap
                            pass
                    
                    if new_unique < best_new_unique:
                        best_new_unique = new_unique
                        best_j = j
                
                if best_j >= 0:
                    # Perform swap
                    for idx in block_tsubs_list[i]:
                        coverage_count[idx] -= 1
                    for idx in block_tsubs_list[best_j]:
                        coverage_count[idx] += 1
                    
                    block_tsubs_list[i] = block_tsubs_list[best_j].copy()
                    blocks[i] = blocks[best_j].copy()
                    changed = True
        
        return blocks
    
    # Main: generate initial covering, then polish with WalkSAT
    best_blocks = None
    best_count = float('inf')
    
    # Generate initial covering
    blocks = greedy_cover()
    if blocks is None:
        # Fallback
        blocks = []
        uncovered = set(all_tsubs)
        while uncovered:
            seed = min(uncovered)
            block_set = set(seed)
            while len(block_set) < k:
                best_p = max(range(v), key=lambda p: (
                    sum(1 for ts in itertools.combinations(block_set, t-1) 
                        if tuple(sorted(ts + (p,))) in uncovered) if p not in block_set else -1
                ))
                block_set.add(best_p)
            block = sorted(block_set)
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
    
    # Remove redundant
    blocks = remove_redundant(blocks)
    
    # WalkSAT polish
    blocks = walk_sat_polish(blocks, max_time=60)
    
    # Remove redundant again
    blocks = remove_redundant(blocks)
    
    # Local search swap
    blocks = local_search_swap(blocks)
    
    # Final remove
    blocks = remove_redundant(blocks)
    
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
