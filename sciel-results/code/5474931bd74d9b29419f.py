import itertools
import time
import random
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Simulated Annealing 
    local search starting from a greedy solution.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # block_covers[i] is a list of t-subset indices covered by k-subset i
    block_covers = [None] * n_k
    for i, k_sub in enumerate(k_subsets):
        covers = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            covers.append(idx)
        block_covers[i] = covers

    # 2. Initialization: Greedy Set Cover
    # Precompute masks for fast greedy initialization
    block_masks = [0] * n_k
    for i in range(n_k):
        mask = 0
        for idx in block_covers[i]:
            mask |= (1 << idx)
        block_masks[i] = mask

    uncovered_mask = (1 << n_t) - 1
    initial_blocks = []
    while uncovered_mask:
        best_block_idx = -1
        max_covered = 0
        for i in range(n_k):
            covered = (uncovered_mask & block_masks[i]).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_block_idx = i
        if max_covered == 0:
            break
        initial_blocks.append(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    # 3. Simulated Annealing
    current_blocks = initial_blocks[:]
    current_cost = len(current_blocks)
    
    # Coverage counts
    cov_counts = [0] * n_t
    for b_idx in current_blocks:
        for t_idx in block_covers[b_idx]:
            cov_counts[t_idx] += 1
            
    best_blocks = current_blocks[:]
    best_cost = current_cost
    
    # SA Parameters
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.995
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if not current_blocks:
            break
            
        # Randomly choose between Removal and Swap
        if rng.random() < 0.4:
            # Try Removal
            b_idx_to_remove = current_blocks[rng.randint(0, len(current_blocks) - 1)]
            
            # Check validity: all t in block must have count > 1
            valid = True
            for t_idx in block_covers[b_idx_to_remove]:
                if cov_counts[t_idx] == 1:
                    valid = False
                    break
            
            if valid:
                for t_idx in block_covers[b_idx_to_remove]:
                    cov_counts[t_idx] -= 1
                current_blocks.remove(b_idx_to_remove)
                current_cost -= 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
            else:
                # Calculate penalty
                penalty = 0
                for t_idx in block_covers[b_idx_to_remove]:
                    if cov_counts[t_idx] == 1:
                        penalty += 1
                if penalty > 0:
                    prob = math.exp(-penalty / T)
                    if rng.random() < prob:
                        for t_idx in block_covers[b_idx_to_remove]:
                            cov_counts[t_idx] -= 1
                        current_blocks.remove(b_idx_to_remove)
                        current_cost -= 1
        else:
            # Try Swap
            if len(current_blocks) == 0:
                continue
            b_idx_to_swap = current_blocks[rng.randint(0, len(current_blocks) - 1)]
            new_b_idx = rng.randint(0, n_k - 1)
            
            if new_b_idx == b_idx_to_swap:
                continue
            if new_b_idx in current_blocks:
                continue
                
            # Check validity
            valid = True
            new_block_set = set(block_covers[new_b_idx])
            for t_idx in block_covers[b_idx_to_swap]:
                if cov_counts[t_idx] == 1 and t_idx not in new_block_set:
                    valid = False
                    break
            
            if valid:
                for t_idx in block_covers[b_idx_to_swap]:
                    cov_counts[t_idx] -= 1
                for t_idx in block_covers[new_b_idx]:
                    cov_counts[t_idx] += 1
                pos = current_blocks.index(b_idx_to_swap)
                current_blocks[pos] = new_b_idx
                # Cost same
            else:
                penalty = 0
                for t_idx in block_covers[b_idx_to_swap]:
                    if cov_counts[t_idx] == 1 and t_idx not in new_block_set:
                        penalty += 1
                if penalty > 0:
                    prob = math.exp(-penalty / T)
                    if rng.random() < prob:
                        for t_idx in block_covers[b_idx_to_swap]:
                            cov_counts[t_idx] -= 1
                        for t_idx in block_covers[new_b_idx]:
                            cov_counts[t_idx] += 1
                        pos = current_blocks.index(b_idx_to_swap)
                        current_blocks[pos] = new_b_idx
                        # Cost same, state invalid
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return [list(k_subsets[idx]) for idx in best_blocks]
