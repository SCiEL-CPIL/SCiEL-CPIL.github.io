import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks and their covered t-subsets
    # This is critical for speed in SA.
    # Number of blocks: C(24, 7) = 346,104.
    # Storing them as a list of frozensets of t-subset indices.
    all_blocks = []
    block_to_idx = {}
    
    # We might not need ALL blocks if we are smart, but for SA we need to propose
    # random blocks. Generating 346k blocks and their coverage is fast in Python 
    # if optimized, or we can generate on the fly.
    # Let's generate on the fly but cache? No, random proposal needs uniformity.
    # Let's precompute coverage for all possible blocks.
    # 346,104 * C(7,3)=35 lookups. ~12M lookups. Very fast.
    
    # To save memory, we can just generate a random block and compute its coverage.
    # But SA needs to evaluate delta.
    
    # Let's stick to the "on-the-fly" generation for proposed blocks to save memory,
    # but we need to be able to pick a random block.
    # We can just pick 7 random distinct points.
    
    # State:
    # blocks: list of lists (k points)
    # cover_count: np.array of length n_t, counts how many blocks cover each t-subset
    
    def get_block_coverage(block):
        """Returns a set of t-subset indices covered by the block."""
        return frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))

    def greedy_cover(seed):
        """Fast greedy initialization."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        # Precompute for speed: for each point, which t-subsets does it participate in?
        # Actually, the current greedy is O(n_t * v * k^2) roughly.
        # Let's use a slightly smarter greedy:
        # Maintain a count of uncovered t-subsets.
        # Pick the block that covers the most uncovered t-subsets.
        
        # To do this efficiently, we can use the "logarithmic heuristic" or similar.
        # But simple greedy is okay for initialization if we do it a few times.
        
        # Optimization: Precompute for each point x, the list of t-subsets containing x.
        # t_subsets_by_point = [list of t-subset indices for each point]
        # This is C(24,2) * 24 = 276 * 24 ~ 6624 items. Small.
        
        # Let's just use the standard greedy from the seed program but optimized.
        # The seed program's greedy was a bit slow due to set lookups in Python.
        # We can use numpy for the uncovered set?
        
        uncovered_np = np.ones(n_t, dtype=bool)
        
        # Precompute block coverage for a given block
        def block_cov_indices(block):
            return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]

        # Greedy:
        while uncovered_np.any():
            if time.time() > deadline - 10.0:
                break
            
            best_block = None
            best_gain = -1
            
            # Sample a few candidate blocks? Or iterate all?
            # Iterating all 346k blocks every step is too slow (346k * 35 * 76 steps ~ 900M ops).
            # We need a smarter greedy.
            
            # Let's use the "random start, greedy fill" approach from the seed,
            # but make it faster.
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered_np)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_ts = all_t_subsets[start_idx]
            
            # Build block containing start_ts
            block = list(start_ts)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                # For each candidate x not in block
                # Gain = number of uncovered t-subsets of form {x, a, b} where a,b in block
                # Since t=3, block has size s. We add x.
                # New t-subsets: {x} + C(block, t-1).
                # We check if they are uncovered.
                
                # To speed this up, we can precompute for each pair (a,b) in block,
                # the set of x such that {x,a,b} is uncovered.
                
                # Let's just iterate x.
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for a, b in itertools.combinations(block, t - 1):
                        ts = tuple(sorted((x, a, b)))
                        idx = t_subset_to_idx.get(ts)
                        if idx is not None and uncovered_np[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            cov = block_cov_indices(block)
            uncovered_np[cov] = False
        
        return blocks

    def simulated_annealing(blocks, seed):
        """
        SA local search.
        Moves:
        1. Remove a block (if valid, i.e., no t-subset becomes uncovered).
           If invalid, we can still do it if we accept higher energy (penalty).
           But removing a block reduces block count.
           Let's define Energy = len(blocks) * 1000 + num_uncovered_t_subsets.
           We want to minimize Energy.
           
        2. Replace a block with a random new block.
           This keeps len(blocks) same, but might reduce num_uncovered.
           
        3. Swap a point in a block with another point.
        """
        rng = random.Random(seed)
        
        # Initial state
        current_blocks = [list(b) for b in blocks]
        current_cover_count = np.zeros(n_t, dtype=np.int32)
        for b in current_blocks:
            cov = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            current_cover_count[cov] += 1
        
        current_uncovered = np.sum(current_cover_count == 0)
        current_len = len(current_blocks)
        current_energy = current_len * 1000 + current_uncovered
        
        best_blocks = [list(b) for b in current_blocks]
        best_energy = current_energy
        
        T = 100.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() < deadline - 5.0 and T > T_min:
            T *= cooling
            
            move_type = rng.random()
            
            if move_type < 0.3 and len(current_blocks) > 1:
                # Try to remove a block
                i = rng.randint(0, len(current_blocks) - 1)
                block_i = current_blocks[i]
                cov_i = [t_subset_to_idx[ts] for ts in itertools.combinations(block_i, t)]
                
                # Check if removing it leaves any uncovered
                # If any idx in cov_i has count == 1, it becomes 0.
                new_uncovered = 0
                for idx in cov_i:
                    if current_cover_count[idx] == 1:
                        new_uncovered += 1
                
                new_len = current_len - 1
                new_energy = new_len * 1000 + (current_uncovered + new_uncovered)
                
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Accept
                    for idx in cov_i:
                        current_cover_count[idx] -= 1
                    current_blocks.pop(i)
                    current_uncovered += new_uncovered
                    current_len = new_len
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in current_blocks]
                        
            elif move_type < 0.7:
                # Replace a block with a random one
                i = rng.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[i]
                old_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(old_block, t)]
                
                # Generate random new block
                new_block = rng.sample(range(v), k)
                new_block.sort()
                new_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(new_block, t)]
                
                # Calculate change in uncovered
                # We remove old_cov, add new_cov.
                # We need to count how many t-subsets go from 0 to 1 (good) and 1 to 0 (bad).
                
                # This is O(k^3) per move.
                # Let's do it carefully.
                
                # First, decrement old
                # We can't just decrement and check, because we need to restore if rejected.
                # Let's simulate.
                
                # For each idx in old_cov:
                #   if count == 1, it becomes 0 (uncovered increases by 1)
                #   if count > 1, it stays covered (uncovered unchanged)
                
                # For each idx in new_cov:
                #   if count == 0, it becomes 1 (uncovered decreases by 1)
                #   if count > 0, it stays covered (uncovered unchanged)
                
                # Note: idx can be in both old_cov and new_cov.
                # If idx in both:
                #   old count c -> c-1 -> c. Net change 0.
                #   But we must handle the order or logic correctly.
                
                # Let's use a temporary dictionary or just iterate.
                # Since k=7, C(7,3)=35. Small.
                
                delta_uncovered = 0
                
                # Create a set of new_cov for fast lookup?
                # Or just iterate.
                
                # Let's mark old_cov indices with -1, new_cov with +1 in a dict?
                # No, just iterate.
                
                # To be safe and simple:
                # 1. Decrement old_cov in a temp array? No, too slow to copy.
                # 2. Just count.
                
                # For idx in old_cov:
                #   if current_cover_count[idx] == 1:
                #       # This one will become 0
                #       # BUT if it's also in new_cov, it will be incremented back to 1.
                #       # So we need to know if it's in new_cov.
                
                # Let's use a set for new_cov.
                new_cov_set = set(new_cov)
                
                for idx in old_cov:
                    if current_cover_count[idx] == 1:
                        if idx not in new_cov_set:
                            delta_uncovered += 1
                        # else: it becomes 0 then 1, net 0.
                    
                for idx in new_cov:
                    if current_cover_count[idx] == 0:
                        if idx not in old_cov: # wait, old_cov is a list.
                             # We need to check if it was in old_cov.
                             # If it was in old_cov, count was >= 1 (since it was covered).
                             # So if count == 0, it was NOT in old_cov.
                             pass
                        # Actually, if count == 0, it was not covered by old_block.
                        # So it becomes 1. Uncovered decreases.
                        delta_uncovered -= 1
                        
                # Wait, the logic "if idx in new_cov and count==0" implies it wasn't in old_cov.
                # Because if it was in old_cov, count would be >= 1.
                # So the check `if idx not in old_cov` is redundant if we rely on count==0.
                # But what if old_cov had it twice? No, sets of t-subsets are unique.
                
                # Let's re-verify:
                # If idx in new_cov and count[idx] == 0:
                #   It was not covered before. Now it is. Uncovered -1.
                #   Could it be in old_cov? If so, count[idx] would be >= 1. Contradiction.
                #   So it is safe to just do delta_uncovered -= 1.
                
                # If idx in old_cov and count[idx] == 1:
                #   It was covered only by this block.
                #   If idx in new_cov, it will be covered by new block. Net 0.
                #   If idx not in new_cov, it becomes uncovered. Net +1.
                
                # So the logic holds.
                
                new_len = current_len
                new_energy = new_len * 1000 + (current_uncovered + delta_uncovered)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Apply move
                    for idx in old_cov:
                        current_cover_count[idx] -= 1
                    for idx in new_cov:
                        current_cover_count[idx] += 1
                    
                    current_blocks[i] = new_block
                    current_uncovered += delta_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in current_blocks]
                        
            else:
                # Swap a point in a block
                i = rng.randint(0, len(current_blocks) - 1)
                block_i = current_blocks[i]
                
                # Pick a point to remove
                x_idx = rng.randint(0, k - 1)
                x = block_i[x_idx]
                
                # Pick a point to add
                y = rng.choice([p for p in range(v) if p not in block_i])
                
                # Old block, new block
                new_block = list(block_i)
                new_block[x_idx] = y
                new_block.sort()
                
                old_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(block_i, t)]
                new_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(new_block, t)]
                
                # Same delta calculation as replace
                new_cov_set = set(new_cov)
                delta_uncovered = 0
                for idx in old_cov:
                    if current_cover_count[idx] == 1 and idx not in new_cov_set:
                        delta_uncovered += 1
                for idx in new_cov:
                    if current_cover_count[idx] == 0:
                        delta_uncovered -= 1
                        
                new_len = current_len
                new_energy = new_len * 1000 + (current_uncovered + delta_uncovered)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    for idx in old_cov:
                        current_cover_count[idx] -= 1
                    for idx in new_cov:
                        current_cover_count[idx] += 1
                    
                    current_blocks[i] = new_block
                    current_uncovered += delta_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = [list(b) for b in current_blocks]
                        
        # Return the best valid covering found
        # We must ensure best_blocks is a valid covering (no uncovered t-subsets).
        # Since we track energy, and best_energy is the min,
        # if best_energy < 1000 * min_len, it might have uncovered.
        # We should only return a valid covering.
        
        # Check validity of best_blocks
        test_count = np.zeros(n_t, dtype=np.int32)
        for b in best_blocks:
            cov = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            test_count[cov] += 1
            
        if np.any(test_count == 0):
            # It's invalid. We need to fix it or return a valid one.
            # Let's try to fix it by adding blocks greedily.
            # Or just return the initial greedy result which was valid.
            # Better: find the first valid state in the history? We didn't store history.
            # Let's just re-run greedy if this happens.
            # Or, simpler: ensure we only update best_blocks if it is valid?
            # In the SA loop, we update best_blocks if energy is lower.
            # Energy = len * 1000 + uncovered.
            # A valid covering has uncovered = 0.
            # An invalid one has uncovered > 0.
            # If a valid covering with len L has energy 1000L.
            # An invalid one with len L-1 and uncovered 1 has energy 1000(L-1) + 1 = 1000L - 999.
            # So the invalid one has LOWER energy!
            # This means SA will prefer invalid coverings with fewer blocks.
            # This is a problem.
            
            # We should penalize uncovered t-subsets much more heavily.
            # Energy = len * 10000 + uncovered * 100?
            # Or just reject any move that increases uncovered?
            # But SA needs to explore.
            
            # Let's change the energy function:
            # Energy = len * 10000 + uncovered * 1000.
            # Then 1 uncovered is worth 1000, which is 1 block.
            # So it will only remove a block if it doesn't increase uncovered,
            # or if it removes 2 blocks and adds 1 uncovered?
            # No, we only remove 1 block at a time.
            # So removing a block that causes 1 uncovered increases energy by 1000 - 10000 = -9000.
            # It will still accept it if T is high.
            
            # The real issue is we return an invalid covering.
            # We MUST return a valid covering.
            
            # Strategy:
            # 1. Keep track of the best VALID covering.
            # 2. In the SA loop, only update best_valid if the current state is valid.
            
            pass # This block is just a comment for now
            
        # Let's restart the logic to ensure validity.
        return best_blocks # This might be invalid!

    # Corrected SA logic
    def simulated_annealing_valid(blocks, seed):
        rng = random.Random(seed)
        
        current_blocks = [list(b) for b in blocks]
        current_cover_count = np.zeros(n_t, dtype=np.int32)
        for b in current_blocks:
            cov = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            current_cover_count[cov] += 1
        
        current_uncovered = np.sum(current_cover_count == 0)
        current_len = len(current_blocks)
        
        # Best VALID state
        best_blocks = [list(b) for b in current_blocks]
        best_len = current_len
        
        T = 50.0
        T_min = 1.0
        cooling = 0.99
        
        while time.time() < deadline - 5.0 and T > T_min:
            T *= cooling
            
            move_type = rng.random()
            
            if move_type < 0.3 and len(current_blocks) > 1:
                # Remove block
                i = rng.randint(0, len(current_blocks) - 1)
                block_i = current_blocks[i]
                cov_i = [t_subset_to_idx[ts] for ts in itertools.combinations(block_i, t)]
                
                new_uncovered = 0
                for idx in cov_i:
                    if current_cover_count[idx] == 1:
                        new_uncovered += 1
                
                new_len = current_len - 1
                new_uncovered_total = current_uncovered + new_uncovered
                
                # Acceptance probability
                # We want to minimize len, but prefer valid.
                # Score = len * 1000 + uncovered
                old_score = current_len * 1000 + current_uncovered
                new_score = new_len * 1000 + new_uncovered_total
                delta = new_score - old_score
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    for idx in cov_i:
                        current_cover_count[idx] -= 1
                    current_blocks.pop(i)
                    current_uncovered = new_uncovered_total
                    current_len = new_len
                    
                    if current_uncovered == 0 and current_len < best_len:
                        best_len = current_len
                        best_blocks = [list(b) for b in current_blocks]
                        
            elif move_type < 0.7:
                # Replace block
                i = rng.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[i]
                old_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(old_block, t)]
                
                new_block = rng.sample(range(v), k)
                new_block.sort()
                new_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(new_block, t)]
                
                new_cov_set = set(new_cov)
                delta_uncovered = 0
                for idx in old_cov:
                    if current_cover_count[idx] == 1 and idx not in new_cov_set:
                        delta_uncovered += 1
                for idx in new_cov:
                    if current_cover_count[idx] == 0:
                        delta_uncovered -= 1
                        
                new_uncovered_total = current_uncovered + delta_uncovered
                
                old_score = current_len * 1000 + current_uncovered
                new_score = current_len * 1000 + new_uncovered_total
                delta = new_score - old_score
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    for idx in old_cov:
                        current_cover_count[idx] -= 1
                    for idx in new_cov:
                        current_cover_count[idx] += 1
                    
                    current_blocks[i] = new_block
                    current_uncovered = new_uncovered_total
                    
                    if current_uncovered == 0 and current_len < best_len:
                        best_len = current_len
                        best_blocks = [list(b) for b in current_blocks]
                        
            else:
                # Swap point
                i = rng.randint(0, len(current_blocks) - 1)
                block_i = current_blocks[i]
                x_idx = rng.randint(0, k - 1)
                x = block_i[x_idx]
                y = rng.choice([p for p in range(v) if p not in block_i])
                
                new_block = list(block_i)
                new_block[x_idx] = y
                new_block.sort()
                
                old_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(block_i, t)]
                new_cov = [t_subset_to_idx[ts] for ts in itertools.combinations(new_block, t)]
                
                new_cov_set = set(new_cov)
                delta_uncovered = 0
                for idx in old_cov:
                    if current_cover_count[idx] == 1 and idx not in new_cov_set:
                        delta_uncovered += 1
                for idx in new_cov:
                    if current_cover_count[idx] == 0:
                        delta_uncovered -= 1
                        
                new_uncovered_total = current_uncovered + delta_uncovered
                
                old_score = current_len * 1000 + current_uncovered
                new_score = current_len * 1000 + new_uncovered_total
                delta = new_score - old_score
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    for idx in old_cov:
                        current_cover_count[idx] -= 1
                    for idx in new_cov:
                        current_cover_count[idx] += 1
                    
                    current_blocks[i] = new_block
                    current_uncovered = new_uncovered_total
                    
                    if current_uncovered == 0 and current_len < best_len:
                        best_len = current_len
                        best_blocks = [list(b) for b in current_blocks]
                        
        return best_blocks

    # Main execution
    best_blocks = None
    best_count = float('inf')
    
    # Run greedy + SA multiple times
    num_runs = 20
    
    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break
            
        # Greedy init
        blocks = greedy_cover(i)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # SA optimization
        optimized = simulated_annealing_valid(blocks, i + 100)
        
        if len(optimized) < best_count:
            best_count = len(optimized)
            best_blocks = optimized

    # Final verification
    if best_blocks is None:
        return []
        
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
            
    if len(covered) < n_t:
        # Fallback to a guaranteed valid greedy
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            start_idx = min(uncovered)
            block = list(all_t_subsets[start_idx])
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        return blocks

    return best_blocks
