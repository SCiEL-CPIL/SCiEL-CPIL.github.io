import itertools
import time
import random
import numpy as np
from collections import defaultdict

DEADLINE_SECONDS = 115.0

def _generic_greedy(v, k, t, start_time):
    """Generic randomized greedy for small instances."""
    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    n_t = len(all_t)
    idx_map = {s: i for i, s in enumerate(all_t)}
    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)

    # coverage[b][s] = 1 if block b covers t-subset s
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    for b, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            cov[b, idx_map[sub]] = 1

    uncovered = np.ones(n_t, dtype=bool)
    chosen = []
    deadline = start_time + DEADLINE_SECONDS

    while uncovered.any() and time.time() < deadline:
        gains = cov[:, uncovered].sum(axis=1)
        if gains.max() <= 0:
            break
        b = int(np.argmax(gains))
        chosen.append(b)
        uncovered[cov[b] == 1] = False

    return [list(all_blocks[i]) for i in chosen]

def construct(v, k, t):
    start_time = time.time()

    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precompute for v=19, k=8, t=5
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)  # 11628
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)  # 75582

    # Build coverage matrix: cov[b, s] = 1 if block b covers t-subset s
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    block_subsets = []
    for b, blk in enumerate(all_blocks):
        subs = []
        for sub in itertools.combinations(blk, t):
            idx = idx_map[sub]
            cov[b, idx] = 1
            subs.append(idx)
        block_subsets.append(subs)
    
    block_subsets_np = [np.array(s, dtype=np.int32) for s in block_subsets]

    # For each t-subset, list of blocks that cover it
    blocks_for_subset = [np.where(cov[:, s] == 1)[0] for s in range(n_t)]
    
    # Precompute "scarcity" of each subset: number of blocks covering it
    subset_scarcity = np.array([len(blocks_for_subset[s]) for s in range(n_t)], dtype=np.float32)

    best_blocks = None
    best_count = float('inf')
    deadline = start_time + DEADLINE_SECONDS

    rng = random.Random(42)

    def prune(chosen_list, current_counts=None):
        if not chosen_list:
            return [], np.zeros(n_t, dtype=np.int16)
        
        if current_counts is None:
            counts = np.zeros(n_t, dtype=np.int16)
            for b in chosen_list:
                counts += cov[b]
        else:
            counts = current_counts.copy()
            
        new_chosen = []
        indices = list(range(len(chosen_list)))
        rng.shuffle(indices)
        
        for i in indices:
            b = chosen_list[i]
            subs = block_subsets_np[b]
            if len(subs) == 0:
                new_chosen.append(b)
                continue
                
            if np.all(counts[subs] > 1):
                counts -= cov[b]
            else:
                new_chosen.append(b)
                
        return new_chosen, counts

    def local_search(chosen, counts):
        improved = True
        local_iterations = 0
        max_local_iterations = 300
        deadline_local = deadline - 5.0
        
        while improved and local_iterations < max_local_iterations and time.time() < deadline_local:
            improved = False
            local_iterations += 1
            
            indices = list(range(len(chosen)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline_local:
                    break
                    
                b_old = chosen[i]
                subs_old = block_subsets_np[b_old]
                
                essential_mask = counts[subs_old] == 1
                if not np.any(essential_mask):
                    chosen.pop(i)
                    counts -= cov[b_old]
                    improved = True
                    break
                    
                essential_subs = subs_old[essential_mask]
                
                if len(essential_subs) == 0:
                    continue
                    
                cand_set = set(blocks_for_subset[essential_subs[0]].tolist())
                cand_set.discard(b_old)
                
                for s in essential_subs[1:]:
                    if not cand_set:
                        break
                    cand_set &= set(blocks_for_subset[s].tolist())
                
                if not cand_set:
                    continue
                
                cands_list = list(cand_set)
                b_new = rng.choice(cands_list)
                
                chosen[i] = b_new
                counts -= cov[b_old]
                counts += cov[b_new]
                improved = True
                break

        return chosen, counts

    # Main loop
    while time.time() < deadline - 10.0:
        # Strategy 1: Most Constrained Greedy
        # Pick the uncovered subset with the fewest covering blocks
        uncovered = np.ones(n_t, dtype=bool)
        chosen = []
        
        # Compute initial scarcity for uncovered subsets
        # scarcity[s] = number of blocks covering s that are still "available" (not chosen yet, though in greedy we don't remove blocks from pool, we just mark subsets covered)
        # Actually, in standard greedy, we don't remove blocks. We just pick the best block.
        # "Most Constrained" means: among uncovered subsets, pick one with min blocks_for_subset[s].
        # Then pick the best block covering it.
        
        while uncovered.any() and time.time() < deadline - 15.0:
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            # Find the uncovered subset with the minimum number of covering blocks
            # We can precompute the scarcity and just find the min among uncovered
            # But scarcity is static. 
            # Let's pick a random subset from the top 10 most constrained uncovered subsets to add randomness
            # Compute scarcity for all uncovered
            # This is O(n_t) each step, which is 11628. 300 steps * 11628 is ~3.5M ops. Fast.
            unc_scarcity = np.array([len(blocks_for_subset[s]) for s in unc_idx])
            
            # Find indices of the 10 most constrained
            k_top = min(10, len(unc_idx))
            # argsort to get indices of smallest scarcity
            top_indices = unc_idx[np.argsort(unc_scarcity)[:k_top]]
            
            # Pick one randomly from these
            s = int(rng.choice(top_indices))
            
            cands = blocks_for_subset[s]
            if len(cands) == 0:
                break

            # Compute gain for each candidate
            gains = np.einsum('ij,j->i', cov[cands], uncovered)
            gains_perturbed = gains + rng.random() * 0.5
            best_local = int(np.argmax(gains_perturbed))
            b = int(cands[best_local])
            
            chosen.append(b)
            uncovered[cov[b] == 1] = False
            
        if not uncovered.any():
            pruned, counts = prune(chosen)
            chosen = pruned
            
            chosen, counts = local_search(chosen, counts)
            
            pruned, _ = prune(chosen)
            chosen = pruned
            
            if len(chosen) < best_count:
                best_count = len(chosen)
                best_blocks = chosen[:]

    if best_blocks is None:
        # Fallback
        uncovered = np.ones(n_t, dtype=bool)
        chosen = []
        while uncovered.any() and time.time() < deadline:
            gains = np.einsum('ij,j->i', cov, uncovered)
            b = int(np.argmax(gains))
            if gains[b] <= 0:
                break
            chosen.append(b)
            uncovered[cov[b] == 1] = False
        best_blocks = chosen
        pruned, _ = prune(best_blocks)
        best_blocks = pruned

    return [list(all_blocks[i]) for i in best_blocks]
