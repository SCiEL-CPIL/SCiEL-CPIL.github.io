import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    def run_greedy(rng=None):
        """
        Runs the greedy set cover algorithm.
        If rng is provided, it adds small random noise to the gains to perturb selection.
        Returns a list of block indices.
        """
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        selected_indices = []
        used_blocks = set()
        
        while remaining > 0:
            if time.time() > deadline:
                break

            best_j = -1
            best_gain = 0
            
            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                
                if rng is not None:
                    # Add small noise to break ties and perturb
                    gain += rng.random() * 0.5
                
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j == -1 or best_gain == 0:
                break

            selected_indices.append(best_j)
            used_blocks.add(best_j)
            
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain

        return selected_indices

    def remove_redundancies(selected_indices):
        """
        Removes redundant blocks from the selected indices.
        """
        if not selected_indices:
            return []
            
        active_indices = list(selected_indices)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_active = []
            for i_idx, bi in enumerate(active_indices):
                other_mask = 0
                for j_idx, bj in enumerate(active_indices):
                    if i_idx != j_idx:
                        other_mask |= ksub_masks[bj]
                
                if (ksub_masks[bi] & ~other_mask) == 0:
                    changed = True
                else:
                    new_active.append(bi)
            
            active_indices = new_active
            if not changed:
                break
                
        return active_indices

    # 1. Deterministic Greedy
    best_indices = run_greedy(None)
    best_indices = remove_redundancies(best_indices)
    best_count = len(best_indices)
    
    # 2. Randomized Greedy Restarts
    # Try to find a better solution by perturbing the greedy choice
    while time.time() < deadline - 5.0:
        rng = random.Random()
        # Use a slightly higher noise to explore different branches
        # 0.5 is arbitrary, could be tuned. 
        # If noise is too high, it picks random blocks. 
        # If too low, it's nearly deterministic.
        candidate_indices = run_greedy(rng)
        candidate_indices = remove_redundancies(candidate_indices)
        candidate_count = len(candidate_indices)
        
        if candidate_count < best_count:
            best_count = candidate_count
            best_indices = candidate_indices
            
    # Final conversion to list of blocks
    if best_indices:
        final_blocks = [list(all_ksubs[idx]) for idx in best_indices]
        # Verify validity
        final_mask = 0
        for idx in best_indices:
            final_mask |= ksub_masks[idx]
        if final_mask == (1 << n_tsubs) - 1:
            return final_blocks
    
    return []
