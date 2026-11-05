import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Helper to get mask of a block
    def get_block_mask(block):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Precompute masks for all blocks
    block_masks = [get_block_mask(b) for b in all_blocks]
    
    # Helper to enumerate blocks containing a set of points (for local search)
    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return []
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Global Greedy Construction
    def greedy_global(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            # Iterate through all precomputed blocks
            for i in range(n_blocks):
                mask = block_masks[i]
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if not best_indices:
                break  # Should not happen if valid covering exists
                
            # Pick randomly among best to allow diversity
            idx = random.choice(best_indices)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        return [all_blocks[i] for i in selected_indices]

    # Local Greedy Construction (Faster, restricted)
    def greedy_local(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet
            # Find a set bit
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]
            
            # Find best block containing start_triplet
            best_block = None
            best_gain = -1
            
            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_mask = b_mask
                    
            if best_block:
                blocks.append(best_block)
                uncovered &= ~best_mask
            else:
                # Fallback: shouldn't happen
                break
                
        return blocks

    # Local Search Optimizer
    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        max_iters = 5000
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                pairs_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(pairs_to_try)
                
                for i, j in pairs_to_try:
                    if time.time() > deadline - 10:
                        break
                        
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                    ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                    
                    needed_triplets = []
                    all_involved = ids_b1 | ids_b2
                    for tid in all_involved:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            
                    if not needed_triplets:
                        # Both redundant
                        b_remove = blocks.pop(j)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        b_remove = blocks.pop(i)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                            
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        covers = all(tid in new_ids for tid in needed_triplets)
                        
                        if covers:
                            # Swap
                            b_remove = blocks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                                
                            blocks.append(new_block)
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                                
                            improved = True
                            found_replacement = True
                            break
                            
                    if improved:
                        break
                        
            # 3. Try 1-for-1
            if not improved:
                i = random.randint(0, n-1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                
                needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                
                if needed_triplets:
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) <= k:
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1):
                                continue
                            new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                            if all(tid in new_ids for tid in needed_triplets):
                                b_remove = blocks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.append(new_block)
                                for triplet in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1
                                improved = True
                                break
                                
            if not improved:
                break
                
        return blocks

    # QUBO-inspired Simulated Annealing Optimizer
    def optimize_sa(blocks, max_time=30.0):
        """
        Simulated annealing on the block set.
        Energy = block_count + penalty * (redundancy)
        Moves: add block, remove block, swap block
        """
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        def is_valid(b_list):
            """Check if all triplets are covered"""
            for b in b_list:
                pass
            # Use counts to check
            return np.min(counts) > 0
        
        def energy(b_list):
            """
            Energy function: minimize block count + redundancy penalty
            """
            n_blocks = len(b_list)
            # Count redundancy: sum over triplets of max(0, count - 1)
            redundancy = np.sum(np.maximum(counts - 1, 0))
            return n_blocks + 0.1 * redundancy
        
        current_blocks = list(blocks)
        current_energy = energy(current_blocks)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        T = 10.0  # Initial temperature
        T_min = 0.01
        cooling = 0.995
        
        start = time.time()
        
        while T > T_min and (time.time() - start) < max_time:
            if time.time() > deadline - 5:
                break
            
            n = len(current_blocks)
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random block
                new_block = random.choice(all_blocks)
                if new_block in current_blocks:
                    continue
                trial_blocks = current_blocks + [new_block]
                trial_counts = counts.copy()
                for triplet in itertools.combinations(new_block, t):
                    trial_counts[triplet_to_id[triplet]] += 1
                
                # Check if valid (it will be, since we're adding)
                trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
                
            elif move_type < 0.7:
                # Try to remove a block
                if n == 0:
                    continue
                idx = random.randint(0, n - 1)
                trial_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                trial_counts = counts.copy()
                for triplet in itertools.combinations(current_blocks[idx], t):
                    trial_counts[triplet_to_id[triplet]] -= 1
                
                # Check validity
                if np.min(trial_counts) < 0:
                    continue
                if np.min(trial_counts) == 0:
                    # Invalid: some triplet uncovered
                    # Penalize heavily
                    uncovered = np.sum(trial_counts == 0)
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0)) + 100 * uncovered
                else:
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
                    
            else:
                # Try to swap a block
                if n == 0:
                    continue
                idx = random.randint(0, n - 1)
                old_block = current_blocks[idx]
                new_block = random.choice(all_blocks)
                if new_block == old_block:
                    continue
                if new_block in current_blocks:
                    continue
                
                trial_blocks = current_blocks[:]
                trial_blocks[idx] = new_block
                trial_counts = counts.copy()
                for triplet in itertools.combinations(old_block, t):
                    trial_counts[triplet_to_id[triplet]] -= 1
                for triplet in itertools.combinations(new_block, t):
                    trial_counts[triplet_to_id[triplet]] += 1
                
                if np.min(trial_counts) < 0:
                    continue
                if np.min(trial_counts) == 0:
                    uncovered = np.sum(trial_counts == 0)
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0)) + 100 * uncovered
                else:
                    trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
            
            delta_E = trial_energy - current_energy
            
            # Accept if better, or with probability exp(-delta_E / T) if worse
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Update
                current_blocks = trial_blocks
                current_energy = trial_energy
                
                # Update counts
                counts = trial_counts.copy()
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = list(current_blocks)
            
            T *= cooling
            
            # Occasionally try a more targeted move: remove a redundant block if exists
            if random.random() < 0.1:
                for i in range(len(current_blocks)):
                    b = current_blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        trial_blocks = current_blocks[:i] + current_blocks[i+1:]
                        trial_counts = counts.copy()
                        for triplet in itertools.combinations(b, t):
                            trial_counts[triplet_to_id[triplet]] -= 1
                        if np.min(trial_counts) > 0:
                            trial_energy = len(trial_blocks) + 0.1 * np.sum(np.maximum(trial_counts - 1, 0))
                            delta_E = trial_energy - current_energy
                            if delta_E < 0:
                                current_blocks = trial_blocks
                                current_energy = trial_energy
                                counts = trial_counts
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_blocks = list(current_blocks)
                        break
        
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Strategy: Mix of Global and Local Greedy
    candidates = []
    
    # Run Local Greedy 10 times
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        blocks = greedy_local(seed)
        candidates.append(blocks)
        
    # Run Global Greedy 5 times
    for seed in range(10, 15):
        if time.time() > deadline - 20:
            break
        blocks = greedy_global(seed)
        candidates.append(blocks)
        
    # Optimize all candidates with local search first
    optimized_candidates = []
    for blocks in candidates:
        if time.time() > deadline - 15:
            break
        opt_blocks = optimize(blocks)
        optimized_candidates.append(opt_blocks)
        
    # Now apply SA to the best few candidates
    # Sort by length and take top 3
    optimized_candidates.sort(key=lambda x: len(x))
    sa_candidates = optimized_candidates[:3]
    
    for blocks in sa_candidates:
        if time.time() > deadline - 10:
            break
        sa_blocks = optimize_sa(blocks, max_time=20.0)
        
        # Verify
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in sa_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(sa_blocks) < best_len:
                best_len = len(sa_blocks)
                best_blocks = sa_blocks
    
    # Also check the non-SA optimized candidates
    for blocks in optimized_candidates:
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_local(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
