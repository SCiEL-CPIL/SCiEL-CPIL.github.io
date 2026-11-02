import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precomputation: generate all blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Build coverage matrix: block -> set of subset indices it covers
    # For v=24, k=7, t=3: n_blocks=346104, n_subsets=2024
    # Each block covers C(7,3)=35 subsets
    block_coverage = []
    for block in all_blocks:
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(subset_to_idx[sub])
        block_coverage.append(covered)
    
    # Convert to numpy for vectorized operations
    # coverage_matrix[i][j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((n_blocks, n_subsets), dtype=np.uint8)
    for i, cov in enumerate(block_coverage):
        for j in cov:
            coverage_matrix[i, j] = 1
    
    # Vectorized greedy set cover
    def vectorized_greedy(seed_rng):
        blocks = []
        covered = np.zeros(n_subsets, dtype=np.uint8)
        
        while not covered.all():
            if time.time() > deadline:
                return None
            
            # Compute marginal gain for each block
            # gain[i] = number of uncovered subsets covered by block i
            gains = np.sum(coverage_matrix * (1 - covered), axis=1)
            
            # Select block with maximum gain (tie-break randomly)
            max_gain = gains.max()
            if max_gain == 0:
                return None  # Shouldn't happen if covered.all() is false
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                idx = seed_rng.choice(candidates.tolist())
            else:
                idx = candidates[0]
            
            blocks.append(all_blocks[idx])
            covered |= coverage_matrix[idx]
        
        return blocks
    
    # Local search: try to replace a block with another that covers its unique subsets
    def local_search(blocks, max_iter=5000):
        if not blocks:
            return blocks
        if time.time() > deadline - 5:
            return blocks
        
        blocks = list(blocks)
        n = len(blocks)
        rng = random.Random(42)
        
        # Precompute for each block, which subsets it uniquely covers
        # and build a quick lookup
        
        def compute_unique_subsets(blocks_list):
            """For each block, find subsets only it covers."""
            # subset -> list of block indices
            subset_to_blocks = {}
            for i, block in enumerate(blocks_list):
                for sub in itertools.combinations(block, t):
                    subset_to_blocks.setdefault(sub, []).append(i)
            return subset_to_blocks
        
        best = list(blocks)
        best_len = len(best)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            subset_to_blocks = compute_unique_subsets(best)
            
            # Try removing a block
            removed = False
            for i in range(len(best)):
                # Check if block i has any unique subsets
                block_i = best[i]
                has_unique = False
                for sub in itertools.combinations(block_i, t):
                    if sub in subset_to_blocks and subset_to_blocks[sub] == [i]:
                        has_unique = True
                        break
                
                if not has_unique:
                    # Block i is redundant
                    new_blocks = best[:i] + best[i+1:]
                    best = new_blocks
                    best_len = len(best)
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try swapping: replace block i with a different block j
            # such that j covers all unique subsets of i
            improved = False
            for i in range(len(best)):
                if time.time() > deadline - 5:
                    break
                block_i = best[i]
                # Find unique subsets of block i
                unique_subs = []
                for sub in itertools.combinations(block_i, t):
                    if sub in subset_to_blocks and subset_to_blocks[sub] == [i]:
                        unique_subs.append(sub)
                
                if not unique_subs:
                    continue
                
                # Try to find a replacement block that covers all unique_subs
                # and doesn't add too many new requirements
                # Sample some candidate blocks
                for attempt in range(20):
                    j = rng.randint(0, n_blocks - 1)
                    if j == i:
                        continue
                    candidate_block = all_blocks[j]
                    if set(candidate_block) == set(block_i):
                        continue
                    
                    # Check if candidate covers all unique subsets
                    cand_set = set(candidate_block)
                    covers_all = True
                    for sub in unique_subs:
                        if not all(x in cand_set for x in sub):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Check if removing i and adding j doesn't leave any subset uncovered
                    # i.e., for each subset covered only by i, j must cover it
                    # We already checked unique subsets. But we also need to check
                    # that no OTHER block becomes redundant in a way that breaks coverage.
                    # Actually, we just need: every subset was covered before, 
                    # and after swap, every subset is still covered.
                    # Subsets covered by i but not by j: need another block to cover them.
                    # This is complex. Simplification: just check unique subsets of i.
                    # If i has no unique subsets, it's redundant (handled above).
                    # If i has unique subsets, j must cover them.
                    # Other subsets covered by i are also covered by other blocks,
                    # so they remain covered.
                    
                    new_blocks = best[:i] + [candidate_block] + best[i+1:]
                    # Verify validity quickly: check that all subsets are covered
                    # This is expensive, so only do it if we think it might help
                    # For now, trust the unique subset check
                    
                    best = new_blocks
                    best_len = len(best)
                    improved = True
                    break
                
                if improved:
                    break
        
        return best
    
    # Run vectorized greedy multiple times with different seeds
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 30:
        seed_rng = random.Random(num_restarts)
        blocks = vectorized_greedy(seed_rng)
        if blocks is None:
            break
        
        # Apply local search to reduce block count
        refined = local_search(blocks, max_iter=2000)
        
        if len(refined) < best_count:
            best_count = len(refined)
            best_blocks = refined
        
        num_restarts += 1
    
    if best_blocks is None:
        # Fallback: single greedy run
        seed_rng = random.Random(999)
        blocks = vectorized_greedy(seed_rng)
        if blocks:
            return blocks
        return []
    
    return best_blocks
