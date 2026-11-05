import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 145  # Safety margin, evaluator kills at 150
    
    # 1. Generate all blocks and subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute which subsets each block covers
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute which blocks cover each subset
    # Store as numpy arrays for vectorized updates?
    # blocks_covering_subset[s] is a list of block indices.
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    # Convert to numpy arrays for faster access?
    # Actually, lists are fine for iteration, but for `potential[b] -= 1` loops,
    # we can use numpy fancy indexing if we convert the lists to arrays once.
    blocks_covering_subset_np = [np.array(lst, dtype=np.int32) for lst in blocks_covering_subset]

    # Helper: Remove redundant blocks from a solution
    def remove_redundant(selected_blocks, cover_count):
        changed = True
        while changed and (time.time() - start_time < timeout):
            changed = False
            # Iterate through selected blocks in reverse
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    changed = True
        return selected_blocks, cover_count

    # Helper: Run Greedy
    def run_greedy(randomize_ties=False):
        cover_count_local = np.zeros(N_subsets, dtype=np.int32)
        potential_local = np.array([len(s) for s in block_subsets], dtype=np.int32)
        selected_local = []
        
        while np.any(cover_count_local == 0):
            if time.time() - start_time > timeout:
                return None, None
                
            max_pot = np.max(potential_local)
            if max_pot == 0:
                break
                
            if randomize_ties:
                candidates = np.where(potential_local == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = np.argmax(potential_local)
                
            selected_local.append(best_idx)
            
            # Update cover_count
            subs = block_subsets[best_idx]
            for s in subs:
                old_count = cover_count_local[s]
                cover_count_local[s] += 1
                if old_count == 0:
                    # Vectorized update of potential
                    blocks_cov = blocks_covering_subset_np[s]
                    potential_local[blocks_cov] -= 1
                        
        return selected_local, cover_count_local

    # Helper: Perturb and Repair
    def perturb_repair(selected_blocks, cover_count):
        if not selected_blocks:
            return selected_blocks, cover_count
        
        # Make copies
        new_blocks = list(selected_blocks)
        new_counts = cover_count.copy()
        
        # 1. Remove a random block
        rem_idx = random.randint(0, len(new_blocks) - 1)
        b_out = new_blocks[rem_idx]
        new_blocks.pop(rem_idx)
        
        # Update counts and find uncovered
        uncovered = []
        for s in block_subsets[b_out]:
            new_counts[s] -= 1
            if new_counts[s] == 0:
                uncovered.append(s)
        
        # 2. Repair: Add blocks to cover 'uncovered'
        uncovered_set = set(uncovered)
        
        while uncovered_set:
            if time.time() - start_time > timeout:
                return None, None
            
            best_block = -1
            best_cov = 0
            
            # Find candidates
            candidates = set()
            for s in uncovered_set:
                candidates.update(blocks_covering_subset[s])
            
            # Filter out already selected
            selected_set = set(new_blocks)
            candidates.difference_update(selected_set)
            
            if not candidates:
                return None, None
                
            # Find best candidate
            # Optimization: If we have many candidates, this loop is slow.
            # But candidates is usually small (~20k).
            for b in candidates:
                cov = 0
                for s in block_subsets[b]:
                    if s in uncovered_set:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_block = b
                    if cov == len(uncovered_set):
                        break
            
            if best_block == -1 or best_cov == 0:
                return None, None
                
            # Add best_block
            new_blocks.append(best_block)
            selected_set.add(best_block)
            
            # Update counts and uncovered_set
            for s in block_subsets[best_block]:
                old_count = new_counts[s]
                new_counts[s] += 1
                if old_count == 0:
                    uncovered_set.discard(s)
                    
        # 3. Remove redundant
        new_blocks, new_counts = remove_redundant(new_blocks, new_counts)
        
        return new_blocks, new_counts

    # Initial Solution
    best_blocks, best_counts = run_greedy(randomize_ties=False)
    if best_blocks is None:
        # Fallback: Return empty or best effort?
        # Should not happen.
        return []
        
    best_blocks, best_counts = remove_redundant(best_blocks, best_counts)
    best_len = len(best_blocks)
    
    # Local Search / ILS
    current_blocks = best_blocks
    current_counts = best_counts
    
    while time.time() - start_time < timeout - 2:
        # Try perturbation
        new_blocks, new_counts = perturb_repair(current_blocks, current_counts)
        
        if new_blocks is not None and len(new_blocks) < best_len:
            best_len = len(new_blocks)
            best_blocks = new_blocks
            best_counts = new_counts
            current_blocks = new_blocks
            current_counts = new_counts
            
    # Final conversion
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
