import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute: for each block, which t-subsets it covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def coverage_still_valid(coverage_count, block_idx_to_remove):
        """Check if removing block_idx_to_remove still leaves all t-subsets covered."""
        subs_in_block = block_t_indices[block_idx_to_remove]
        return np.all(coverage_count[subs_in_block] > 1)
    
    def aggressive_deletion(current_blocks, coverage_count):
        """Try harder to remove blocks, including blocks where most subsets are over-covered."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                # Count how many of this block's t-subsets are covered by other blocks
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    # Safe to remove
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
            # Second pass: try removing blocks where >= 90% of subsets are over-covered
            # and verify the few uncovered ones are covered elsewhere
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                # If at least 95% are covered by others, try removing
                if np.mean(other_covers > 0) >= 0.95:
                    # Check if the uncovered ones are covered by any other block in current set
                    uncovered_mask = other_covers <= 0
                    if np.any(uncovered_mask):
                        uncovered_subs = subs_in_block[uncovered_mask]
                        # Check if any other block covers these
                        can_remove = True
                        for unc_sub in uncovered_subs:
                            # Find if any other block in new_blocks covers this t-subset
                            found = False
                            for other_b in new_blocks:
                                if other_b != b_idx:
                                    # Check if unc_sub is in other_b's t-subsets
                                    if np.any(block_t_indices[other_b] == unc_sub):
                                        found = True
                                        break
                            if not found:
                                can_remove = False
                                break
                        if can_remove:
                            improved = True
                            coverage_count[subs_in_block] -= 1
                            continue
                    else:
                        improved = True
                        coverage_count[subs_in_block] -= 1
                        continue
                new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Track how many times each t-subset has been covered (for weighted scoring)
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Compute gain: number of new t-subsets covered
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among blocks with max gain, use weighted score based on rarity
            # Prefer blocks that cover rarer t-subsets
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 30:
                # Sample from top candidates
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(30, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            # Score candidates: gain + bonus for covering rare subsets
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                # Bonus: sum of inverse (1 + cover_freq) for newly covered subsets
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Try pairwise swaps: replace one block with another to potentially reduce size
        # or improve structure for further deletion
        swap_improved = True
        while swap_improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            swap_improved = False
            
            # Try removing each block and seeing if we can replace it with a better one
            # that allows further deletion
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                # Temporarily remove
                coverage_count[block_t_indices[b_idx]] -= 1
                
                # Find uncovered t-subsets
                uncovered = np.where(coverage_count == 0)[0]
                
                if len(uncovered) == 0:
                    # Can remove this block entirely
                    del current_blocks[i]
                    swap_improved = True
                    break
                
                # Try to find a block that covers all uncovered and has minimal overlap
                # with existing coverage
                best_swap = None
                best_swap_score = float('inf')
                
                for other_b in all_ks:
                    if other_b in current_blocks:
                        continue
                    other_subs = block_t_indices[other_b]
                    # Check if it covers all uncovered
                    if not np.all(coverage_count[other_subs] >= 0):  # always true, need different check
                        continue
                    
                    # Check if this block covers all currently uncovered t-subsets
                    covers_all_uncovered = True
                    for unc in uncovered:
                        if not np.any(other_subs == unc):
                            covers_all_uncovered = False
                            break
                    
                    if covers_all_uncovered:
                        # Score: how much new coverage this adds (lower is better for swap)
                        new_cover = np.sum((coverage_count[other_subs] == 0))
                        # We want to minimize new_cover since we're replacing one block
                        if new_cover < best_swap_score:
                            best_swap_score = new_cover
                            best_swap = other_b
                
                if best_swap is not None and best_swap_score == 0:
                    # Perfect swap: covers exactly the uncovered, no new coverage
                    current_blocks[i] = best_swap
                    coverage_count[block_t_indices[best_swap]] += 1
                    swap_improved = True
                    break
                else:
                    # Restore
                    coverage_count[block_t_indices[b_idx]] += 1
        
        # Final deletion pass
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
