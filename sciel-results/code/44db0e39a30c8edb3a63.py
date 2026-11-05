import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    # For v=24, k=7, t=3: C(24,3)=2024 triplets. 
    # Python integers can handle bitmasks of size 2024 easily.
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            # Find block with max gain
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_gain == 0:
                break
                
            chosen_idx = rng.choice(candidates)
            
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_indices

    def get_counts(indices):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for idx in indices:
            mask = block_masks[idx]
            # Efficiently count bits set in mask? 
            # Since we need per-triplet counts, we iterate set bits.
            # Or simpler: since k is small, we can just iterate the block's triplets.
            # But we have the index. Let's use the precomputed block structure if available, 
            # or just iterate the mask bits.
            # Iterating mask bits is O(number of 1s). C(7,3)=35.
            m = mask
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                counts[tid] += 1
                m ^= lsb
        return counts

    def is_valid(indices):
        # Check if every triplet is covered at least once
        covered = 0
        for idx in indices:
            covered |= block_masks[idx]
        return (covered == all_mask)

    def remove_redundant(indices):
        if not indices:
            return indices
        counts = get_counts(indices)
        new_indices = []
        for idx in indices:
            # Check if any triplet in this block is uniquely covered by it
            unique = False
            m = block_masks[idx]
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                if counts[tid] == 1:
                    unique = True
                    break
                m ^= lsb
            if unique:
                new_indices.append(idx)
        return new_indices

    def simulated_annealing(initial_indices, seed):
        rng = random.Random(seed)
        current_indices = list(initial_indices)
        current_indices = remove_redundant(current_indices)
        
        if not is_valid(current_indices):
            # If initial is invalid, try to fix by adding random blocks
            # This shouldn't happen if greedy was good, but safety first
            while not is_valid(current_indices) and time.time() < deadline - 10:
                # Find uncovered
                covered = 0
                for idx in current_indices:
                    covered |= block_masks[idx]
                uncovered = all_mask & ~covered
                # Pick a random block that covers some uncovered
                # For speed, just pick random until we find one
                # Or greedy pick
                best_idx = -1
                best_gain = 0
                for idx in range(n_blocks):
                    gain = (block_masks[idx] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx == -1:
                    break
                current_indices.append(best_idx)
        
        current_cost = len(current_indices)
        best_indices = list(current_indices)
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.995
        
        step = 0
        while T > T_min and time.time() < deadline - 5:
            step += 1
            if step % 100 == 0:
                # Update best
                pass
            
            # Generate move
            move_type = rng.random()
            new_indices = list(current_indices)
            
            if move_type < 0.5 and len(current_indices) > 0:
                # Delete a random block
                idx_to_remove = rng.randrange(len(current_indices))
                new_indices.pop(idx_to_remove)
                # Cost change: +1 if valid, infinity if invalid? 
                # We want to minimize size. 
                # If we delete, size decreases by 1.
                # If invalid, we reject (or penalize).
                # Let's define energy = size + penalty * uncovered_count
                # But checking validity is O(N*K^t).
                # Let's just check validity.
                if is_valid(new_indices):
                    new_cost = len(new_indices)
                    delta_cost = new_cost - current_cost
                    if delta_cost < 0:
                        # Always accept improvement
                        current_indices = new_indices
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_indices = list(current_indices)
                    else:
                        # Should not happen for deletion if valid? 
                        # If valid, size is smaller. 
                        # Wait, if I delete a block, size is always smaller. 
                        # So if valid, delta_cost is always -1.
                        pass
                else:
                    # Invalid, reject
                    pass
            elif move_type < 0.9:
                # Swap: remove one, add one
                if len(current_indices) == 0:
                    continue
                idx_to_remove = rng.randrange(len(current_indices))
                idx_to_add = rng.randrange(n_blocks)
                
                # Avoid swapping with itself
                if current_indices[idx_to_remove] == idx_to_add:
                    continue
                
                new_indices.pop(idx_to_remove)
                new_indices.append(idx_to_add)
                
                # Check validity
                if is_valid(new_indices):
                    new_cost = len(new_indices)
                    delta_cost = new_cost - current_cost
                    # delta_cost is 0 here
                    # We accept if random() < exp(-delta/T)
                    # Since delta is 0, exp(0)=1, so always accept?
                    # This leads to random walk on valid solutions.
                    # To avoid getting stuck, we might want to bias towards "better" swaps?
                    # But size is same. 
                    # Maybe we should track "slack" or something?
                    # For now, accept to explore.
                    if rng.random() < 1.0: # Always accept same cost
                        current_indices = new_indices
                        # current_cost unchanged
                else:
                    pass
            else:
                # Insert a random block (might increase cost, but could enable future deletions)
                idx_to_add = rng.randrange(n_blocks)
                if idx_to_add not in current_indices:
                    new_indices.append(idx_to_add)
                    new_cost = len(new_indices)
                    delta_cost = new_cost - current_cost
                    if delta_cost < 0:
                        current_indices = new_indices
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_indices = list(current_indices)
                    else:
                        # Accept with probability exp(-delta/T)
                        if rng.random() < np.exp(-delta_cost / T):
                            current_indices = new_indices
                            current_cost = new_cost
            
            T *= cooling_rate
            
        return best_indices

    # 1. Get a good initial solution using greedy
    best_indices = None
    best_count = float('inf')
    
    # Run a few greedy seeds to find a strong starting point
    # Greedy is fast.
    for seed in range(5):
        if time.time() > deadline - 60:
            break
        inds = greedy_set_cover(seed)
        inds = remove_redundant(inds)
        if len(inds) < best_count:
            best_count = len(inds)
            best_indices = inds
            
    if best_indices is None:
        best_indices = remove_redundant(greedy_set_cover(0))
        best_count = len(best_indices)
        
    # 2. Run Simulated Annealing
    # We can run SA multiple times with different seeds, starting from the best greedy or random valid solutions
    # Let's run SA from the best greedy solution and maybe a couple of others
    
    sa_seeds = [100, 200, 300]
    for s in sa_seeds:
        if time.time() > deadline - 10:
            break
        
        # Start from best found so far
        start_inds = list(best_indices)
        
        # Optional: Perturb start
        # For now, just run SA
        
        result_inds = simulated_annealing(start_inds, s)
        
        if len(result_inds) < best_count:
            best_count = len(result_inds)
            best_indices = result_inds
            
    # Convert indices back to blocks
    final_blocks = [all_blocks[i] for i in best_indices]
    
    # Final sanity check
    if not is_valid(best_indices):
        # Fallback to a simple greedy if SA failed
        fallback = greedy_set_cover(0)
        final_blocks = [all_blocks[i] for i in fallback]
        
    return final_blocks

if __name__ == "__main__":
    pass
