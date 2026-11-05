import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)

    # 2. Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # 3. Precompute for each block, the list of t-subset indices it covers
    # For v=24, k=7, t=3: n_blocks ~ 346k, each covers C(7,3)=35 tsubs.
    # Total entries ~ 12M. This is the main precomputation cost.
    block_tsubs = []
    for block in all_blocks:
        tsubs = [t_sub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs)

    # 4. Precompute inverse mapping: for each t-subset, list of block indices covering it
    tsub_to_blocks = defaultdict(list)
    for b_idx, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_to_blocks[ts_idx].append(b_idx)

    # Helper to get tsubs for a block index
    def get_tsubs(b_idx):
        return block_tsubs[b_idx]

    # 5. Greedy Initialization Strategies
    # We will run multiple greedy strategies with different random seeds and selection criteria.
    
    def greedy_init(seed, mode='max_gain'):
        """
        mode: 'max_gain' (standard greedy), 'ratio' (gain / size, though size is constant, 
              maybe gain / overlap? No, let's stick to max gain but with random tie-breaking 
              or weighted by inverse frequency of the t-subsets to prioritize rare ones).
        Actually, the prompt suggests "ratio of newly covered triplets to the number of points".
        Since all blocks have size k, this is just proportional to gain.
        However, a better heuristic is to weight by the rarity of the t-subsets.
        Let's implement 'weighted' where weight is 1/freq(tsub).
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        selected = []
        
        # Precompute weights for weighted mode
        if mode == 'weighted':
            pass
            # Count frequency of each t-subset in the entire universe of blocks
            # We already have tsub_to_blocks. freq = len(tsub_to_blocks[ts_idx])
            # inv_freq = 1.0 / freq
            # But we can just use the precomputed freq if we had it.
            # Let's compute freq once outside if needed, or just use max_gain for now 
            # and rely on multiple restarts.
            # Actually, let's stick to a fast max_gain with random tie-breaking for variety.
            
        while np.any(uncovered):
            if time.time() > deadline - 10:
                return None
            
            # Calculate gain for each block
            # gain[b] = sum(uncovered[block_tsubs[b]])
            # This is expensive if done naively in Python for 346k blocks.
            # But we can vectorize? 
            # block_tsubs is a list of lists. 
            # Creating a numpy array of shape (n_blocks, C(k,t)) is 346k * 35 ~ 12M ints.
            # 12M * 4 bytes = 48MB. Feasible.
            
            # Let's precompute the big numpy array once.
            # Wait, I didn't do that in the global scope. Let's do it here or globally.
            # Doing it globally is better.
            
        # Since I didn't precompute the big array globally, let's do it now if not done.
        # Actually, let's refactor to use a global precomputed array for speed.
        
        # For now, this function is a placeholder. I will implement the main loop below.
        return None

    # --- Refactored Plan ---
    # 1. Precompute `block_tsubs_array` as a numpy array of shape (n_blocks, C(k,t)) with dtype int32.
    # 2. Implement fast greedy using numpy.
    # 3. Implement local search (remove redundant, swap) using the numpy array for fast gain calculation.
    
    # Precompute big array
    t_per_block = int(math.comb(k, t))
    # Memory check: n_blocks * t_per_block * 4 bytes
    # 346104 * 35 * 4 = ~48 MB. OK.
    block_tsubs_arr = np.empty((n_blocks, t_per_block), dtype=np.int32)
    for i, tsubs in enumerate(block_tsubs):
        block_tsubs_arr[i] = tsubs

    # Release the list of lists to save memory? Not strictly necessary but good practice.
    # del block_tsubs
    
    def greedy_set_cover(seed, use_weighted=False):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        selected = []
        
        # Precompute weights if needed
        if use_weighted:
            # Weight of a t-subset is inverse of its frequency in all blocks
            # freq[ts_idx] = len(tsub_to_blocks[ts_idx])
            # We can compute this once.
            # Let's compute freq array
            if not hasattr(greedy_set_cover, 'freq_arr'):
                freq_arr = np.zeros(n_tsubs, dtype=np.float64)
                for ts_idx, b_list in tsub_to_blocks.items():
                    freq_arr[ts_idx] = len(b_list)
                greedy_set_cover.freq_arr = freq_arr
            weights = 1.0 / (greedy_set_cover.freq_arr + 1e-9)
        else:
            weights = None

        while np.any(uncovered):
            if time.time() > deadline - 20:
                return None

            # Calculate gains
            # uncovered is a boolean array of size n_tsubs.
            # block_tsubs_arr is (n_blocks, t_per_block).
            # We want for each block, the sum of weights of uncovered tsubs it covers.
            
            # If not weighted, gain = number of uncovered tsubs in block.
            # If weighted, gain = sum of weights of uncovered tsubs in block.
            
            # Efficient calculation:
            # For a specific block b, gain = sum(weights[uncovered_indices_in_b])
            # Vectorized: 
            # mask = uncovered[block_tsubs_arr] -> shape (n_blocks, t_per_block)
            # if weighted: gains = (mask * weights[block_tsubs_arr]).sum(axis=1)
            # if not: gains = mask.sum(axis=1)
            
            mask = uncovered[block_tsubs_arr]
            if use_weighted:
                # weights is size n_tsubs. We need to index it with block_tsubs_arr.
                # w_matrix = weights[block_tsubs_arr] -> (n_blocks, t_per_block)
                # gains = (mask * w_matrix).sum(axis=1)
                w_matrix = weights[block_tsubs_arr]
                gains = np.sum(mask * w_matrix, axis=1)
            else:
                gains = np.sum(mask, axis=1)
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if valid covering exists and uncovered > 0
            
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            if len(candidates) > 1:
                b_idx = candidates[rng.randrange(len(candidates))]
            else:
                b_idx = candidates[0]
                
            selected.append(b_idx)
            # Update uncovered
            uncovered[block_tsubs_arr[b_idx]] = False
            
        if np.any(uncovered):
            return None # Incomplete
            
        return selected

    def remove_redundant(selected_indices):
        """
        Remove blocks that do not cover any unique t-subset.
        Returns new list of indices.
        """
        if not selected_indices:
            return selected_indices
            
        # Count coverage for each t-subset by the current set of blocks
        # We can do this by iterating through selected blocks and incrementing counts.
        # n_selected ~ 80. n_tsubs ~ 2024.
        # For each block, update counts for its 35 tsubs.
        # Total ops ~ 80 * 35 = 2800. Very fast.
        
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in selected_indices:
            cover_count[block_tsubs_arr[b_idx]] += 1
            
        new_selected = []
        for b_idx in selected_indices:
            # Check if this block covers any t-subset with count 1
            # i.e., is it the only one covering some t-subset?
            # If yes, it's essential.
            # If no, it's redundant.
            tsubs_in_block = block_tsubs_arr[b_idx]
            if np.any(cover_count[tsubs_in_block] == 1):
                new_selected.append(b_idx)
            else:
                # Remove it and update counts
                cover_count[tsubs_in_block] -= 1
                
        return new_selected

    def local_search(selected_indices, time_limit):
        """
        Simulated Annealing / Hill Climbing on the set of blocks.
        Moves:
        1. Remove a block (if redundant or with penalty)
        2. Add a block (if it covers uncovered tsubs)
        3. Swap a block
        """
        current = list(selected_indices)
        # Precompute cover_count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in current:
            cover_count[block_tsubs_arr[b_idx]] += 1
            
        n_uncovered = np.sum(cover_count == 0)
        
        best = list(current)
        best_cost = len(current) + 1000 * n_uncovered
        
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        rng = random.Random(42)
        
        while T > min_T and time.time() < time_limit:
            move_type = rng.random()
            
            if move_type < 0.4:
                # Remove
                if len(current) == 0: continue
                b_idx = current[rng.randrange(len(current))]
                tsubs = block_tsubs_arr[b_idx]
                
                # Check if safe
                if np.any(cover_count[tsubs] == 1):
                    # Unsafe: some tsub becomes uncovered
                    delta_uncovered = np.sum(cover_count[tsubs] == 1)
                    delta_cost = -1 + 1000 * delta_uncovered
                    if rng.random() < np.exp(-delta_cost / T):
                        # Accept
                        cover_count[tsubs] -= 1
                        current.remove(b_idx)
                        n_uncovered += delta_uncovered
                        cost = len(current) + 1000 * n_uncovered
                        if cost < best_cost:
                            best_cost = cost
                            best = list(current)
                else:
                    # Safe
                    cover_count[tsubs] -= 1
                    current.remove(b_idx)
                    cost = len(current) + 1000 * n_uncovered
                    if cost < best_cost:
                        best_cost = cost
                        best = list(current)
                        
            elif move_type < 0.7:
                # Add
                if n_uncovered == 0:
                    continue
                # Pick a random uncovered tsub
                # Find indices where cover_count == 0
                # This is O(n_tsubs) which is ~2000. Acceptable.
                uncovered_indices = np.where(cover_count == 0)[0]
                if len(uncovered_indices) == 0:
                    continue
                target_ts = uncovered_indices[rng.randrange(len(uncovered_indices))]
                
                # Get blocks covering this tsub
                candidates = tsub_to_blocks[target_ts]
                if not candidates:
                    continue
                new_b_idx = candidates[rng.randrange(len(candidates))]
                
                if new_b_idx in current:
                    continue
                
                tsubs = block_tsubs_arr[new_b_idx]
                # Gain: number of uncovered tsubs this block covers
                gain = np.sum(cover_count[tsubs] == 0)
                delta_cost = 1 - 1000 * gain
                if rng.random() < np.exp(-delta_cost / T):
                    # Accept
                    cover_count[tsubs] += 1
                    current.append(new_b_idx)
                    n_uncovered -= gain
                    cost = len(current) + 1000 * n_uncovered
                    if cost < best_cost:
                        best_cost = cost
                        best = list(current)
                        
            else:
                # Swap
                if len(current) == 0: continue
                old_b_idx = current[rng.randrange(len(current))]
                
                # Pick a new block
                if n_uncovered > 0:
                    uncovered_indices = np.where(cover_count == 0)[0]
                    if len(uncovered_indices) > 0:
                        target_ts = uncovered_indices[rng.randrange(len(uncovered_indices))]
                        candidates = tsub_to_blocks[target_ts]
                        if candidates:
                            new_b_idx = candidates[rng.randrange(len(candidates))]
                        else:
                            new_b_idx = rng.randrange(n_blocks)
                    else:
                        new_b_idx = rng.randrange(n_blocks)
                else:
                    new_b_idx = rng.randrange(n_blocks)
                    
                if new_b_idx == old_b_idx or new_b_idx in current:
                    continue
                    
                old_tsubs = block_tsubs_arr[old_b_idx]
                new_tsubs = block_tsubs_arr[new_b_idx]
                
                # Calculate delta
                # Remove old, Add new
                # We need to know the state change.
                # Let's just apply and revert if rejected.
                # Apply remove
                cover_count[old_tsubs] -= 1
                # Apply add
                cover_count[new_tsubs] += 1
                
                # Calculate new n_uncovered
                # This is tricky to do incrementally without scanning.
                # But we can check the affected tsubs.
                # Affected = union(old_tsubs, new_tsubs)
                # We need to know how many went from 0->1 or 1->0.
                
                # Simpler: Just compute cost directly?
                # Cost = len(current) + 1000 * n_uncovered
                # len(current) is same.
                # So we just need change in n_uncovered.
                
                # Let's count changes in the affected set.
                affected = np.unique(np.concatenate([old_tsubs, new_tsubs]))
                
                # We need the OLD cover_count values for these.
                # We already modified cover_count.
                # This is getting messy.
                # Let's stick to Remove and Add for now, as Swap is complex to evaluate efficiently
                # without storing old state.
                # Revert
                cover_count[new_tsubs] -= 1
                cover_count[old_tsubs] += 1
                
                # Skip swap for simplicity in this version.
                pass

            T *= cooling

        return best

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy strategies
    seeds = range(10)
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        use_weighted = (seed % 2 == 0)
        selected = greedy_set_cover(seed, use_weighted)
        if selected is None:
            continue
            
        # Remove redundant
        selected = remove_redundant(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = list(selected)
            
        # Local search on this solution
        if time.time() < deadline - 5:
            improved = local_search(selected, deadline - 5)
            if len(improved) < best_len:
                best_len = len(improved)
                best_blocks = improved

    if best_blocks is None:
        # Fallback: simple greedy
        selected = greedy_set_cover(0, False)
        if selected:
            best_blocks = selected
        else:
            best_blocks = []

    return [list(all_blocks[b_idx]) for b_idx in best_blocks]

# Need to import math for comb
import math
