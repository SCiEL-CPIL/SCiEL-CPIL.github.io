import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin for safety
    
    # Pre-computation
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    block_subsets = [[] for _ in range(n_blocks)]
    # For each subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    
    for b_idx, block in enumerate(k_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_subsets[b_idx].append(idx)
            subset_to_blocks[idx].append(b_idx)
    
    def greedy_construction(noise_scale=0.0):
        """Greedy set cover with optional random noise for diversity."""
        uncovered = [True] * n_subsets
        gains = np.array([len(bs) for bs in block_subsets], dtype=np.int32)
        selected = []
        selected_set = set()
        
        while True:
            if not any(uncovered):
                break
            if time.time() > deadline - 30:
                break
            
            if noise_scale > 0:
                noise = np.random.uniform(0, noise_scale, n_blocks)
                scores = gains.astype(float) + noise
            else:
                scores = gains.astype(float)
            
            best_idx = int(np.argmax(scores))
            if gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            for s in block_subsets[best_idx]:
                if uncovered[s]:
                    uncovered[s] = False
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        
        return selected, selected_set
    
    def compute_coverage(selected):
        """Compute coverage counts for a list of block indices."""
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b in selected:
            for s in block_subsets[b]:
                counts[s] += 1
        return counts
    
    def deletion_pass(selected, selected_set, coverage):
        """Remove redundant blocks. Returns True if any removed."""
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            # Random order for diversity
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                b_idx = selected[i]
                redundant = True
                for s in block_subsets[b_idx]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[b_idx]:
                        coverage[s] -= 1
                    selected.pop(i)
                    selected_set.discard(b_idx)
                    changed = True
        return changed
    
    def swap_pass(selected, selected_set, coverage):
        """Try 1-opt swaps: replace a block with a better one. Returns True if improved."""
        if time.time() > deadline - 10:
            return False
        
        improved = False
        max_swaps = 30
        swaps_done = 0
        
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for i in indices:
            if swaps_done >= max_swaps:
                break
            if time.time() > deadline - 10:
                break
            
            b_remove = selected[i]
            
            # Find subsets uniquely covered by b_remove
            critical_subsets = [s for s in block_subsets[b_remove] if coverage[s] == 1]
            
            if not critical_subsets:
                # Redundant block, just remove it
                for s in block_subsets[b_remove]:
                    coverage[s] -= 1
                selected.pop(i)
                selected_set.discard(b_remove)
                improved = True
                break
            
            # Find candidate blocks that cover all critical subsets
            # Use intersection of subset_to_blocks for critical subsets
            candidates = set(subset_to_blocks[critical_subsets[0]])
            for s in critical_subsets[1:]:
                candidates.intersection_update(subset_to_blocks[s])
            
            # Remove the block we're trying to replace and already-selected blocks
            candidates.discard(b_remove)
            candidates -= selected_set
            
            if not candidates:
                continue
            
            # Score candidates: prefer those that help barely-covered subsets
            best_candidate = -1
            best_score = float('-inf')
            
            # Limit candidates for speed
            cand_list = list(candidates)
            if len(cand_list) > 200:
                random.shuffle(cand_list)
                cand_list = cand_list[:200]
            
            for b_add in cand_list:
                score = 0
                for s in block_subsets[b_add]:
                    c = coverage[s]
                    if c == 1:
                        score += 3
                    elif c == 2:
                        score += 1
                    else:
                        score -= 1
                if score > best_score:
                    best_score = score
                    best_candidate = b_add
            
            if best_candidate != -1:
                # Perform swap
                for s in block_subsets[b_remove]:
                    coverage[s] -= 1
                for s in block_subsets[best_candidate]:
                    coverage[s] += 1
                selected[i] = best_candidate
                selected_set.discard(b_remove)
                selected_set.add(best_candidate)
                improved = True
                swaps_done += 1
        
        return improved
    
    def local_search(selected, selected_set, coverage):
        """Iterate deletion and swap passes until no improvement."""
        while time.time() < deadline - 20:
            improved = False
            deletion_pass(selected, selected_set, coverage)
            improved = swap_pass(selected, selected_set, coverage) or improved
            if not improved:
                break
            # After swaps, do another deletion pass
            deletion_pass(selected, selected_set, coverage)
        return selected, selected_set, coverage
    
    best_solution = None
    best_len = float('inf')
    
    # Multiple restarts with different noise levels
    noise_configs = [
        (0.0, 1),      # Deterministic greedy
        (0.1, 2),      # Light noise
        (0.3, 3),      # Medium noise
        (0.5, 4),      # Heavy noise
    ]
    
    for noise_scale, _ in noise_configs:
        if time.time() > deadline - 40:
            break
        
        selected, selected_set = greedy_construction(noise_scale)
        if not selected:
            continue
        
        coverage = compute_coverage(selected)
        
        # Local search
        selected, selected_set, coverage = local_search(selected, selected_set, coverage)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_solution = list(selected)
    
    # Final cleanup on best solution
    if best_solution is None:
        # Fallback: pure greedy
        selected, selected_set = greedy_construction(0.0)
        coverage = compute_coverage(selected)
        deletion_pass(selected, selected_set, coverage)
        best_solution = selected
    else:
        # Do one more deletion pass on the best
        selected_set = set(best_solution)
        coverage = compute_coverage(best_solution)
        deletion_pass(best_solution, selected_set, coverage)
    
    final_blocks = [list(k_blocks[i]) for i in best_solution]
    return final_blocks
