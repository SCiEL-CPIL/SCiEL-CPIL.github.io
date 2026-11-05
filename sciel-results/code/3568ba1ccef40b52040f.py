import itertools
import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = []
for block in itertools.combinations(range(V), K):
    all_blocks.append(block)

N_BLOCKS = len(all_blocks)

# Precompute coverage matrix: coverage[i, j] = 1 if block i covers subset j
coverage = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.uint8)
for i, block in enumerate(all_blocks):
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        coverage[i, idx] = 1

# Precompute for each subset, which blocks cover it (as a set for fast lookup)
blocks_containing_subset = [set() for _ in range(N_SUBSETS)]
for i in range(N_BLOCKS):
    cols = np.where(coverage[i] == 1)[0]
    for j in cols:
        blocks_containing_subset[j].add(i)

# Subset frequency: how many blocks cover each subset
subset_freq = coverage.sum(axis=0)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    def dls_search(seed, max_iter=500000):
        """Distributed Local Search on the dual hypergraph."""
        rng = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        
        # Initialize with greedy set cover
        uncovered = np.ones(N_SUBSETS, dtype=bool)
        selected = np.zeros(N_BLOCKS, dtype=bool)
        selected_list = []
        
        # Greedy initialization
        remaining = N_SUBSETS
        while remaining > 0:
            # Find block covering most uncovered subsets
            gains = (coverage & uncovered[None, :]).sum(axis=1)
            # Among max gain, pick randomly for diversity
            max_gain = gains.max()
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) == 0:
                break
            idx = rng.choice(candidates.tolist())
            if not selected[idx]:
                selected[idx] = True
                selected_list.append(idx)
                uncovered &= ~coverage[idx].astype(bool)
                remaining = uncovered.sum()
        
        best_len = len(selected_list)
        best_selected = selected.copy()
        best_list = selected_list.copy()
        
        # DLS: maintain counts of how many selected blocks cover each subset
        counts = coverage[selected].sum(axis=0)  # shape (N_SUBSETS,)
        
        def score_block(i):
            """Score for adding block i: reward for uncovered, penalty for well-covered."""
            cov = coverage[i]
            # Reward: covers uncovered subsets
            reward = (cov & (counts == 0)).sum()
            # Penalty: covers well-covered subsets (count >= 2)
            penalty = (cov & (counts >= 2)).sum()
            # Slight penalty for covering singly-covered subsets
            mild_penalty = (cov & (counts == 1)).sum() * 0.1
            return reward - penalty - mild_penalty
        
        def score_remove(i):
            """Score for removing block i: cost = number of subsets that would become uncovered."""
            cov = coverage[i]
            # Subsets that would become uncovered if we remove i
            cost = (cov & (counts == 1)).sum()
            # Benefit: reduces over-coverage
            benefit = (cov & (counts >= 3)).sum() * 0.5
            return cost - benefit
        
        no_improve = 0
        max_no_improve = 2000
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Randomly decide: add or remove
            if rng.random() < 0.5:
                # Try to add a block
                # Compute scores for all unselected blocks
                unselected = ~selected
                if unselected.sum() == 0:
                    continue
                
                # Vectorized scoring
                cov_unsel = coverage[unselected]  # shape (n_unsel, N_SUBSETS)
                # Reward: covers uncovered
                reward = (cov_unsel & (counts == 0)[None, :]).sum(axis=1)
                # Penalty: covers well-covered
                penalty = (cov_unsel & (counts >= 2)[None, :]).sum(axis=1)
                mild_penalty = (cov_unsel & (counts == 1)[None, :]).sum(axis=1) * 0.1
                scores = reward - penalty - mild_penalty
                
                # Only consider blocks with positive score
                positive = scores > 0
                if positive.sum() == 0:
                    # Try removing instead
                    continue
                
                # Randomized selection among top scorers
                max_score = scores[positive].max()
                top_candidates = np.where(positive & (scores >= max_score - 2))[0]
                idx_unsel = np.where(unselected)[0]
                idx = idx_unsel[top_candidates[rng.randint(0, len(top_candidates) - 1)]]
                
                # Add block
                selected[idx] = True
                selected_list.append(idx)
                counts += coverage[idx]
                
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_selected = selected.copy()
                    best_list = selected_list.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
            else:
                # Try to remove a block
                if selected.sum() == 0:
                    continue
                
                selected_indices = np.where(selected)[0]
                cov_sel = coverage[selected_indices]
                # Cost: subsets that would become uncovered
                cost = (cov_sel & (counts == 1)[None, :]).sum(axis=1)
                # Benefit: reduces over-coverage
                benefit = (cov_sel & (counts >= 3)[None, :]).sum(axis=1) * 0.5
                scores = cost - benefit
                
                # Only consider blocks with low cost (prefer removable ones)
                low_cost = scores < 0
                if low_cost.sum() == 0:
                    # Try adding instead
                    continue
                
                # Randomized selection
                min_score = scores[low_cost].min()
                bottom_candidates = np.where(low_cost & (scores <= min_score + 2))[0]
                idx = selected_indices[bottom_candidates[rng.randint(0, len(bottom_candidates) - 1)]]
                
                # Remove block
                selected[idx] = False
                selected_list.remove(idx)
                counts -= coverage[idx]
                
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_selected = selected.copy()
                    best_list = selected_list.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            
            if no_improve > max_no_improve:
                # Perturbation: randomly flip a few blocks
                for _ in range(rng.randint(2, 5)):
                    if rng.random() < 0.5 and selected.sum() > 0:
                        sel_idx = np.where(selected)[0]
                        i = sel_idx[rng.randint(0, len(sel_idx) - 1)]
                        selected[i] = False
                        selected_list.remove(i)
                        counts -= coverage[i]
                    elif ~selected.any() == False:
                        unsel_idx = np.where(~selected)[0]
                        if len(unsel_idx) > 0:
                            i = unsel_idx[rng.randint(0, len(unsel_idx) - 1)]
                            selected[i] = True
                            selected_list.append(i)
                            counts += coverage[i]
                no_improve = 0
                max_no_improve = min(5000, max_no_improve * 2)
        
        # Verify coverage of best solution
        if len(best_list) > 0:
            best_cov = coverage[best_list].sum(axis=0)
            if (best_cov > 0).all():
                return best_list
            # Fallback: use greedy result
            return selected_list
        return selected_list

    def polish(selected_list):
        """Remove redundant blocks from the solution."""
        if not selected_list:
            return selected_list
        
        counts = coverage[selected_list].sum(axis=0)
        selected_set = set(selected_list)
        result = list(selected_list)
        
        # Try removing each block if it's redundant
        for i in range(len(result)):
            idx = result[i]
            cov = coverage[idx]
            # Check if any subset would become uncovered
            would_uncover = (cov & (counts == 1)).sum()
            if would_uncover == 0:
                # Remove it
                counts -= cov
                result.pop(i)
                i -= 1
        
        return result

    # Run multiple independent DLS searches
    best_overall = None
    best_overall_len = float('inf')
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart * 137
        result = dls_search(seed)
        
        # Polish the result
        polished = polish(result)
        
        # Verify
        if len(polished) > 0:
            cov_check = coverage[polished].sum(axis=0)
            if (cov_check > 0).all():
                if len(polished) < best_overall_len:
                    best_overall_len = len(polished)
                    best_overall = polished
    
    if best_overall is None:
        # Fallback: pure greedy
        uncovered = np.ones(N_SUBSETS, dtype=bool)
        selected = []
        remaining = N_SUBSETS
        while remaining > 0:
            gains = (coverage & uncovered[None, :]).sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            idx = np.argmax(gains)
            selected.append(idx)
            uncovered &= ~coverage[idx].astype(bool)
            remaining = uncovered.sum()
        best_overall = selected
    
    return [list(all_blocks[i]) for i in best_overall]
