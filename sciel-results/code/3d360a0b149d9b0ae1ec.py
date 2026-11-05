import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, compute which t-subsets it covers
    # We'll build the constraint matrix A where A[i, j] = 1 if block j covers t-subset i
    # But the matrix is 3876 x 50388, which is ~195M entries. Too large for dense.
    # We need a sparse representation or a smarter approach.
    
    # Alternative: Use a greedy approach but with a better heuristic.
    # The ILP formulation is too large for scipy.optimize.milp to handle efficiently
    # within the time limit for this problem size.
    
    # Let's try a different structural approach:
    # 1. Use a randomized greedy with multiple restarts
    # 2. Add a local search / local optimization phase
    # 3. Use a different tie-breaking strategy
    
    # Actually, let's try a different approach:
    # Use a "maximal" greedy that picks the block covering the most uncovered subsets,
    # but with randomization in tie-breaking and multiple restarts.
    
    # Precompute for each block, the list of t-subsets it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t) if sub in t_sub_to_idx]
        block_tsubs.append(tsubs)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple runs with different random seeds
    for run_id in range(200):  # Max 200 runs, but limited by time
        if time.time() > deadline - 8.0:
            break
            
        random.seed(run_id * 12345 + 67890)
        
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices)
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            
            while len(block) < k:
                # Find x that covers most uncovered t-subsets
                best_x = -1
                best_gain = -1
                tied_candidates = []
                
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in t_minus_1_subsets:
                        T = tuple(sorted(rest + (x,)))
                        if T in t_sub_to_idx:
                            idx = t_sub_to_idx[T]
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        tied_candidates = [x]
                    elif gain == best_gain:
                        tied_candidates.append(x)
                        
                if best_x == -1:
                    break
                    
                # Random tie-breaking
                if len(tied_candidates) > 1:
                    best_x = random.choice(tied_candidates)
                    
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False
            
        # End while uncovered
        
        # Local Optimization: Remove redundant blocks
        if len(blocks) < best_count:
            # Try to remove redundant blocks iteratively
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 3.0:
                    break
                    
                # Build coverage count
                coverage_count = np.zeros(num_t, dtype=np.int32)
                for b in blocks:
                    for sub in itertools.combinations(b, t):
                        if sub in t_sub_to_idx:
                            coverage_count[t_sub_to_idx[sub]] += 1
                
                # Check each block for redundancy
                redundant_indices = []
                for i, b in enumerate(blocks):
                    is_essential = False
                    for sub in itertools.combinations(b, t):
                        if sub in t_sub_to_idx:
                            if coverage_count[t_sub_to_idx[sub]] == 1:
                                is_essential = True
                                break
                    if not is_essential:
                        redundant_indices.append(i)
                        
                if redundant_indices:
                    # Remove one redundant block at a time
                    idx = random.choice(redundant_indices)
                    blocks.pop(idx)
                    changed = True
                    
        # End redundancy check
        
        # ILS perturbation: randomly remove some blocks and re-cover gaps
        if len(blocks) < best_count and time.time() < deadline - 5.0:
            perturbed = [list(b) for b in blocks]
            # Remove 10-20% of random blocks
            n_remove = max(1, len(perturbed) // 10)
            to_remove = random.sample(range(len(perturbed)), min(n_remove, len(perturbed)))
            for idx in sorted(to_remove, reverse=True):
                perturbed.pop(idx)
            
            # Re-cover any gaps
            uncovered = np.zeros(num_t, dtype=bool)
            for b in perturbed:
                for sub in itertools.combinations(b, t):
                    if sub in t_sub_to_idx:
                        uncovered[t_sub_to_idx[sub]] = True
            uncovered = ~uncovered
            
            # Fill gaps with greedy
            while uncovered.any():
                if time.time() > deadline - 3.0:
                    break
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                seed_idx = random.choice(uncovered_indices)
                seed_subset = all_t_subsets[seed_idx]
                block = list(seed_subset)
                
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    block_set = set(block)
                    t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                    
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in t_minus_1_subsets:
                            T = tuple(sorted(rest + (x,)))
                            if T in t_sub_to_idx:
                                idx = t_sub_to_idx[T]
                                if uncovered[idx]:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        break
                    block.append(best_x)
                
                block.sort()
                perturbed.append(block)
                
                for sub in itertools.combinations(block, t):
                    if sub in t_sub_to_idx:
                        uncovered[t_sub_to_idx[sub]] = False
            
            # Prune perturbed solution
            changed = True
            while changed and time.time() < deadline - 2.0:
                changed = False
                coverage_count = np.zeros(num_t, dtype=np.int32)
                for b in perturbed:
                    for sub in itertools.combinations(b, t):
                        if sub in t_sub_to_idx:
                            coverage_count[t_sub_to_idx[sub]] += 1
                
                redundant_indices = []
                for i, b in enumerate(perturbed):
                    is_essential = False
                    for sub in itertools.combinations(b, t):
                        if sub in t_sub_to_idx:
                            if coverage_count[t_sub_to_idx[sub]] == 1:
                                is_essential = True
                                break
                    if not is_essential:
                        redundant_indices.append(i)
                        
                if redundant_indices:
                    idx = random.choice(redundant_indices)
                    perturbed.pop(idx)
                    changed = True
            
            if len(perturbed) < len(blocks):
                blocks = perturbed
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    # If no blocks found, fallback to simple greedy
    if best_blocks is None:
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[0]
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in t_minus_1_subsets:
                        T = tuple(sorted(rest + (x,)))
                        if T in t_sub_to_idx:
                            idx = t_sub_to_idx[T]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False
                    
        best_blocks = blocks
        best_count = len(blocks)
        
    # Final validation: ensure all t-subsets are covered
    if best_blocks is not None:
        covered = set()
        for b in best_blocks:
            for sub in itertools.combinations(b, t):
                covered.add(sub)
        if len(covered) < num_t:
            # Cover remaining gaps
            remaining = [sub for sub in all_t_subsets if sub not in covered]
            for sub in remaining:
                # Find a block containing this subset
                found = False
                for b in best_blocks:
                    if all(x in b for x in sub):
                        found = True
                        break
                if not found:
                    # Add a new block containing this subset
                    block = list(sub)
                    while len(block) < k:
                        # Pick random unused point
                        block_set = set(block)
                        candidates = [x for x in range(v) if x not in block_set]
                        if not candidates:
                            break
                        block.append(random.choice(candidates))
                    block.sort()
                    best_blocks.append(block)
    
    return best_blocks
