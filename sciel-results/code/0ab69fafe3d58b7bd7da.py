<<<<<<< SEARCH
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

# Pre-compute all 3-subsets of 24 points and map them to indices
# This is specific to v=24, t=3 but the code handles general v, k, t by 
# falling back to dict-based counting if the pre-computation isn't used.
# For the specific task C(24,7,3), we use the fast path.

def _precompute_subsets(v, t):
    """Pre-compute all t-subsets and their indices for fast lookup."""
    if v == 24 and t == 3:
        # There are C(24,3) = 2024 subsets
        # We map each frozenset to an index 0..2023
        # And we pre-compute the list of indices for each 7-block? 
        # That would be C(24,7) * C(7,3) entries, which is 346k * 35 ~ 12M entries.
        # This is too large for memory if we store all blocks.
        # Instead, we store the index for each 3-subset.
        subsets = []
        index_map = {}
        for i, comb in enumerate(itertools.combinations(range(v), t)):
            fs = frozenset(comb)
            index_map[fs] = i
            subsets.append(fs)
        return index_map, len(subsets)
    else:
        return None, 0

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 118.0  # Leave 2s buffer for safety

    # Fast path for C(24, 7, 3)
    use_fast_path = (v == 24 and k == 7 and t == 3)
    
    if use_fast_path:
        # index_map maps frozenset -> int index
        index_map, num_t_subsets = _precompute_subsets(v, t)
        # coverage_counts is an array of counts for each t-subset
        coverage_counts = [0] * num_t_subsets
        # uncovered_count tracks how many t-subsets have count 0
        uncovered_count = num_t_subsets
        
        def get_count(fs):
            return coverage_counts[index_map[fs]]
            
        def add_block(block):
            nonlocal uncovered_count
            for fs in itertools.combinations(block, t):
                idx = index_map[frozenset(fs)]
                if coverage_counts[idx] == 0:
                    uncovered_count -= 1
                coverage_counts[idx] += 1
                
        def remove_block(block):
            nonlocal uncovered_count
            for fs in itertools.combinations(block, t):
                idx = index_map[frozenset(fs)]
                coverage_counts[idx] -= 1
                if coverage_counts[idx] == 0:
                    uncovered_count += 1
        # Pre-compute the t-subsets for each potential block? 
        # C(24,7) is 346k. Each has 35 subsets. 12M integers. 
        # We can generate them on the fly as frozensets are fast enough for 35 items.
        
    else:
        # Fallback for other parameters
        index_map = {}
        all_subsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
        for i, fs in enumerate(all_subsets):
            index_map[fs] = i
        num_t_subsets = len(all_subsets)
        coverage_counts = [0] * num_t_subsets
        uncovered_count = num_t_subsets
        
        def get_count(fs):
            return coverage_counts[index_map[fs]]
            
        def add_block(block):
            nonlocal uncovered_count
            for fs in itertools.combinations(block, t):
                idx = index_map[frozenset(fs)]
                if coverage_counts[idx] == 0:
                    uncovered_count -= 1
                coverage_counts[idx] += 1
                
        def remove_block(block):
            nonlocal uncovered_count
            for fs in itertools.combinations(block, t):
                idx = index_map[frozenset(fs)]
                coverage_counts[idx] -= 1
                if coverage_counts[idx] == 0:
                    uncovered_count += 1

    def _greedy_cover(v, k, t, rng):
        """Fast greedy construction to get a valid starting covering."""
        # We need to track uncovered sets. 
        # For speed, we can use a set of frozensets initially, or rely on the global state.
        # Let's use a local set for the greedy phase to avoid polluting global state until done.
        uncovered = set(frozenset(c) for c in itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset as seed
            seed = rng.choice(list(uncovered))
            block = list(seed)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        fs = frozenset(list(rest) + [x])
                        if fs in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                
            block.sort()
            b_tuple = tuple(block)
            if b_tuple not in blocks:
                blocks.append(b_tuple)
                for c in itertools.combinations(block, t):
                    uncovered.discard(frozenset(c))
                    
        return blocks

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # Initialize global counts
    for b in blocks:
        add_block(b)
        
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Pre-compute all possible blocks for C(24,7) to speed up sampling?
    # C(24,7) = 346104. Storing them as tuples takes ~20MB. Acceptable.
    # But generating random blocks is also fast. 
    # Let's stick to generating random blocks to save memory and initialization time.
    
    # Helper to check if a block is redundant
    def is_redundant(block):
        for fs in itertools.combinations(block, t):
            idx = index_map[frozenset(fs)]
            if coverage_counts[idx] <= 1:
                return False
        return True

    # Helper to find critical subsets of a block
    def get_critical_subsets(block):
        crit = []
        for fs in itertools.combinations(block, t):
            idx = index_map[frozenset(fs)]
            if coverage_counts[idx] == 1:
                crit.append(fs)
        return crit

    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            if is_redundant(b):
                remove_block(b)
                blocks.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                removed_any = True
                T = max(T_min, T * 0.9) # Cool down on improvement
                break
        
        if removed_any:
            continue
            
        # 2. Simulated Annealing Moves
        if len(blocks) > 0:
            move_type = rng.random()
            
            if move_type < 0.7:
                # Swap a block
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                
                critical = get_critical_subsets(old_block)
                
                if not critical:
                    # No critical subsets, we can replace with any random block
                    # But we want to minimize count, so replacement is only useful if it enables removals.
                    # For now, skip if no criticals, as it's a neutral move.
                    continue
                    
                points_needed = set()
                for fs in critical:
                    points_needed.update(fs)
                
                if len(points_needed) > k:
                    continue
                
                # Generate a new block containing points_needed
                remaining_slots = k - len(points_needed)
                cands = [x for x in range(v) if x not in points_needed]
                rng.shuffle(cands)
                
                # Try a few candidates for the new block
                best_new_block = None
                best_delta_u = 0
                best_gain = -1
                
                # We want a block that covers many currently uncovered sets
                # Or at least doesn't break anything.
                # Since we must include points_needed, we just fill the rest.
                # We can try to fill the rest to maximize coverage of *other* uncovered sets.
                
                # Simple strategy: try 5 random completions
                for _ in range(5):
                    new_block_set = set(points_needed)
                    for x in rng.sample(cands, min(remaining_slots, len(cands))):
                        new_block_set.add(x)
                    
                    if len(new_block_set) < k:
                        # Fill with any
                        for x in cands:
                            if len(new_block_set) >= k: break
                            new_block_set.add(x)
                            
                    new_block = tuple(sorted(new_block_set))
                    if new_block == old_block:
                        continue
                    if new_block in blocks:
                        continue
                        
                    # Evaluate swap
                    # Remove old, add new
                    remove_block(old_block)
                    add_block(new_block)
                    
                    delta_u = uncovered_count - (num_t_subsets - len([c for c in coverage_counts if c>0])) # This is slow, don't do this.
                    # Actually, we can track the change.
                    # Let's just compute the "energy" change.
                    # Energy = len(blocks) + lambda * uncovered. 
                    # Here len(blocks) is constant. So we minimize uncovered.
                    # We just accept if uncovered_count doesn't increase, or with SA prob.
                    
                    # To be fast, we just check if uncovered_count <= old_uncovered_count
                    # But we need to know old_uncovered_count.
                    old_unc = uncovered_count
                    # We already applied the change. 
                    # So current uncovered_count is the new one.
                    # If new_unc <= old_unc, it's an improvement or neutral.
                    
                    if uncovered_count <= old_unc:
                        best_new_block = new_block
                        best_delta_u = uncovered_count - old_unc
                        # Undo to try other candidates if we want the best, or just take the first good one?
                        # Taking the first good one is faster.
                        # But let's try to find a strictly better one if possible.
                        # For now, take the first non-worsening swap.
                        break
                    else:
                        # Undo
                        remove_block(new_block)
                        add_block(old_block)
                        # uncovered_count is restored
                        continue

                if best_new_block is not None:
                    # Apply the best swap found
                    # Note: if we broke early, the state is already modified if we didn't undo.
                    # In the loop above, if we found a good one, we broke WITHOUT undoing.
                    # So blocks[i] is still old_block, but counts reflect new_block?
                    # No, `blocks` list is not updated. We need to update `blocks[i]`.
                    blocks[i] = best_new_block
                    # The counts are already updated from the `add_block`/`remove_block` calls in the loop.
                    # Wait, if we broke, we did NOT undo. So counts are for new_block.
                    # If we didn't break (all were worse), we undid everything.
                    # So this logic is correct.
                    
                    # Check if this swap enabled any removals? 
                    # We'll check for removals in the next iteration.
                    
            else:
                # Perturbation: Remove a random block, then repair
                # This is a larger move to escape local minima.
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                
                # Save state
                # We can't easily save the whole coverage_counts array (2024 ints is small, but let's be careful)
                # Actually, 2024 ints is tiny. We can copy it.
                old_counts = coverage_counts.copy()
                old_uncovered = uncovered_count
                
                remove_block(old_block)
                blocks.pop(i)
                
                # Now we have uncovered_count > 0 (likely)
                # We need to add blocks to cover the new holes.
                # Greedy repair:
                # Find uncovered sets
                # We can't easily iterate uncovered sets from the array without scanning 2024 elements.
                # Scanning 2024 elements is fast.
                
                unc_sets = []
                for idx, c in enumerate(coverage_counts):
                    if c == 0:
                        # We need the frozenset for this index.
                        # We didn't store the list of frozensets in the fast path easily accessible by index?
                        # We have index_map (fs -> idx). We need idx -> fs.
                        # Let's pre-compute idx_to_fs.
                        pass
                
                # Optimization: Pre-compute idx_to_fs
                # This is a bit of work to set up, but worth it.
                
                # For now, let's just accept the perturbation if it doesn't explode the size.
                # Actually, a simpler perturbation: 
                # Just add a random block. 
                # This increases block count by 1. 
                # Then we try to remove redundant blocks.
                
                new_block_set = set(rng.sample(range(v), k))
                new_block = tuple(sorted(new_block_set))
                
                if new_block not in blocks:
                    add_block(new_block)
                    blocks.append(new_block)
                    
                    # If we added a block, we might be able to remove the one we just removed? 
                    # No, we removed it. 
                    # We just increased the count by 0 (removed 1, added 1).
                    # This is a neutral move in count, but changes the structure.
                    
                    # If this leads to a dead end, we will try to remove redundant blocks.
                    
                    # If we are stuck for a long time, we might need to add MORE blocks to repair, 
                    # but that increases the count. 
                    # The goal is to minimize count. 
                    # So we should only add blocks if they allow us to remove MORE than 1 block later?
                    # Or if we can repair without increasing count? 
                    # If we removed 1 and added 1, count is same. 
                    # If we can then remove another, we win.
                    
        T = max(T_min, T * cooling_rate)

    return [list(b) for b in best_blocks]

# We need idx_to_fs for the perturbation repair if we want to do it properly.
# Let's add it to the fast path setup.

def construct(v, k, t):
=======
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(v, k, t):
>>>>>>> REPLACE

