import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)

    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = block_t_indices[b_idx]
                if not np.any(coverage[idxs] == 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def simulated_annealing(blocks_idx, coverage, max_time, seed):
        """
        SA on block selection with remove and swap moves.
        State: list of block indices + coverage array
        Energy: len(blocks) + penalty * num_uncovered
        """
        current_blocks = list(blocks_idx)
        current_coverage = coverage.copy()
        
        if not current_blocks:
            return current_blocks, current_coverage
        
        current_uncovered = np.sum(current_coverage == 0)
        current_energy = len(current_blocks) + 1000.0 * current_uncovered
        
        best_blocks = list(current_blocks)
        best_coverage = current_coverage.copy()
        best_energy = current_energy
        
        T = 20.0
        cooling_rate = 0.995
        min_T = 0.5
        penalty = 1000.0
        
        rng = random.Random(seed)
        
        while time.time() < max_time and T > min_T:
            move_type = rng.random()
            
            if move_type < 0.5 and len(current_blocks) > 0:
                # Try to remove a block
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                idxs = block_t_indices[b_idx]
                
                new_uncovered = np.sum(current_coverage[idxs] == 1)
                valid = (new_uncovered == 0)
                
                if valid:
                    # Always accept
                    current_coverage[idxs] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        current_coverage[idxs] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E
            
            elif move_type < 0.8 and len(current_blocks) > 0:
                # Try to swap: remove one, add another
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                old_idxs = block_t_indices[old_b]
                
                # Pick a new block to add
                new_b = rng.randint(0, num_k_blocks - 1)
                if new_b in current_blocks:
                    continue
                new_idxs = block_t_indices[new_b]
                
                # Compute energy change
                # After removing old_b: coverage[old_idxs] -= 1
                # After adding new_b: coverage[new_idxs] += 1
                # Uncovered count change:
                # Subsets in old_idxs \ new_idxs: if coverage was 1, become uncovered (+1 each)
                # Subsets in new_idxs \ old_idxs: if coverage was 0, become covered (-1 each)
                # Subsets in intersection: no net change to uncovered count
                
                # Efficient computation using sets
                old_set = set(old_idxs.tolist())
                new_set = set(new_idxs.tolist())
                
                only_old = old_set - new_set
                only_new = new_set - old_set
                
                # For only_old: count how many have coverage == 1 (will become uncovered)
                if only_old:
                    only_old_arr = np.array(list(only_old), dtype=np.int32)
                    will_uncover = np.sum(current_coverage[only_old_arr] == 1)
                else:
                    will_uncover = 0
                
                # For only_new: count how many have coverage == 0 (will become covered)
                if only_new:
                    only_new_arr = np.array(list(only_new), dtype=np.int32)
                    will_cover = np.sum(current_coverage[only_new_arr] == 0)
                else:
                    will_cover = 0
                
                delta_uncovered = will_uncover - will_cover
                delta_energy = 0.0 + penalty * delta_uncovered  # block count stays same
                
                if delta_energy <= 0:
                    # Always accept
                    current_coverage[old_idxs] -= 1
                    current_coverage[new_idxs] += 1
                    current_blocks[i] = new_b
                    current_energy += delta_energy
                else:
                    if rng.random() < np.exp(-delta_energy / T):
                        current_coverage[old_idxs] -= 1
                        current_coverage[new_idxs] += 1
                        current_blocks[i] = new_b
                        current_energy += delta_energy
            
            else:
                # Try to add a block (may help by enabling removals later)
                if len(current_blocks) < num_k_blocks:
                    new_b = rng.randint(0, num_k_blocks - 1)
                    if new_b in current_blocks:
                        continue
                    new_idxs = block_t_indices[new_b]
                    will_cover = np.sum(current_coverage[new_idxs] == 0)
                    delta_energy = 1.0 - penalty * will_cover
                    
                    if delta_energy <= 0:
                        current_coverage[new_idxs] += 1
                        current_blocks.append(new_b)
                        current_energy += delta_energy
                    else:
                        if rng.random() < np.exp(-delta_energy / T):
                            current_coverage[new_idxs] += 1
                            current_blocks.append(new_b)
                            current_energy += delta_energy
            
            # Track best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
                best_coverage = current_coverage.copy()
            
            T *= cooling_rate
        
        return best_blocks, best_coverage

    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered_indices = list(range(num_t_subsets))
        blocks_idx = []
        
        # Greedy Phase
        while uncovered_indices:
            if time.time() > deadline - 5.0:
                break
            
            pos = random.randint(0, len(uncovered_indices) - 1)
            u_idx = uncovered_indices[pos]
            
            candidates = t_subset_to_blocks[u_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            blocks_idx.append(best_block_idx)
            
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
            
            uncovered_indices = [x for x in uncovered_indices if coverage[x] == 0]
        
        if len(uncovered_indices) > 0:
            continue
        
        # Reduction Phase
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[i]) for i in blocks_idx]
            
            # SA Phase
            sa_budget = min(3.0, max(0.5, (deadline - 5.0) - time.time()))
            if sa_budget > 0.5:
                sa_blocks, sa_coverage = simulated_annealing(
                    blocks_idx, coverage, 
                    max_time=start_time + sa_budget + time.time() - time.time(), 
                    seed=restart * 100 + 42
                )
                
                # Verify SA result is valid
                if np.all(sa_coverage >= 1):
                    sa_score = len(sa_blocks)
                    if sa_score < best_score:
                        best_score = sa_score
                        best_blocks = [list(all_k_blocks[i]) for i in sa_blocks]

    return best_blocks
