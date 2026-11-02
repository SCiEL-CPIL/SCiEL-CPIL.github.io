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
    
    # Precompute: for each block, which t-subsets it covers (only valid entries)
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    block_t_valid = np.zeros((n_ks, t_count), dtype=bool)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            block_t_valid[i, j] = True
    
    # Build reverse index: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            t_to_blocks[block_t_indices[i, j]].append(i)
    
    # Convert to numpy arrays for faster access
    t_to_blocks_arr = [np.array(b, dtype=np.int32) if b else np.array([], dtype=np.int32) for b in t_to_blocks]
    
    best_solution = None
    best_score = float('inf')
    
    def fast_deletion(current_blocks, coverage_count):
        """Remove all blocks whose t-subsets are all covered by other blocks."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                valid_mask = block_t_valid[b_idx]
                if np.all(coverage_count[subs_in_block[valid_mask]] > 1):
                    improved = True
                    coverage_count[subs_in_block[valid_mask]] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks
    
    def swap_local_search(current_blocks, coverage_count):
        """Try to replace pairs of blocks with a single block that covers all their unique t-subsets."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks, coverage_count
        
        blocks_set = set(current_blocks)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            
            n = len(current_blocks)
            tried = set()
            for i in range(n):
                if time.time() - start_time > time_limit - 3.0:
                    return current_blocks, coverage_count
                b1 = current_blocks[i]
                for j in range(i + 1, n):
                    b2 = current_blocks[j]
                    pair_key = (min(b1, b2), max(b1, b2))
                    if pair_key in tried:
                        continue
                    tried.add(pair_key)
                    
                    subs1 = block_t_indices[b1]
                    valid1 = block_t_valid[b1]
                    subs2 = block_t_indices[b2]
                    valid2 = block_t_valid[b2]
                    
                    unique_to_1 = []
                    unique_to_2 = []
                    for sub in subs1[valid1]:
                        if coverage_count[sub] == 1:
                            unique_to_1.append(sub)
                    for sub in subs2[valid2]:
                        if coverage_count[sub] == 1:
                            unique_to_2.append(sub)
                    
                    if not unique_to_1 and not unique_to_2:
                        improved = True
                        coverage_count[subs1[valid1]] -= 1
                        coverage_count[subs2[valid2]] -= 1
                        current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                        blocks_set.discard(b1)
                        blocks_set.discard(b2)
                        break
                    elif unique_to_1 and not unique_to_2:
                        # Find a block that covers all unique_to_1
                        candidates = t_to_blocks_arr[unique_to_1[0]]
                        found = False
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand][block_t_valid[cand]].tolist())
                                if all(s in cand_subs for s in unique_to_1):
                                    improved = True
                                    coverage_count[subs1[valid1]] -= 1
                                    coverage_count[subs2[valid2]] -= 1
                                    coverage_count[block_t_indices[cand][block_t_valid[cand]]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    found = True
                                    break
                        if found:
                            break
                    elif unique_to_2 and not unique_to_1:
                        candidates = t_to_blocks_arr[unique_to_2[0]]
                        found = False
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand][block_t_valid[cand]].tolist())
                                if all(s in cand_subs for s in unique_to_2):
                                    improved = True
                                    coverage_count[subs1[valid1]] -= 1
                                    coverage_count[subs2[valid2]] -= 1
                                    coverage_count[block_t_indices[cand][block_t_valid[cand]]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    found = True
                                    break
                        if found:
                            break
                    else:
                        pass
            if improved:
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx][block_t_valid[b_idx]]] += 1
                blocks_set = set(current_blocks)
        return current_blocks, coverage_count
    
    def run_peeling_greedy(seed):
        """Hypergraph peeling: focus on most constrained uncovered triples."""
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Track coverage counts for scoring
        cover_count = np.zeros(n_ts, dtype=np.float64)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Find uncovered triples
            uncovered_mask = ~covered
            if not np.any(uncovered_mask):
                break
            
            # For each uncovered triple, count how many blocks can cover it
            # and are not already selected (approximate: use total blocks that cover it)
            uncovered_indices = np.where(uncovered_mask)[0]
            
            # Compute "degree" for each uncovered triple: number of blocks that can cover it
            # We use the precomputed t_to_blocks
            degrees = np.zeros(len(uncovered_indices), dtype=np.int32)
            for idx, tri_idx in enumerate(uncovered_indices):
                degrees[idx] = len(t_to_blocks[tri_idx])
            
            # Find the most constrained (minimum degree) uncovered triples
            min_degree = np.min(degrees)
            most_constrained = uncovered_indices[degrees == min_degree]
            
            # Randomly pick from the most constrained
            chosen_triple = most_constrained[random.randint(0, len(most_constrained) - 1)]
            
            # Among blocks that cover this triple, pick the one that covers the most uncovered triples
            candidate_blocks = t_to_blocks_arr[chosen_triple]
            
            if len(candidate_blocks) == 0:
                # Should not happen if problem is feasible
                break
            
            # Score each candidate by how many new uncovered triples it covers
            best_candidate = -1
            best_gain = -1
            best_score_val = -1
            
            # Limit candidates for speed
            if len(candidate_blocks) > 50:
                sampled = random.sample(list(candidate_blocks), min(50, len(candidate_blocks)))
            else:
                sampled = list(candidate_blocks)
            
            for cand in sampled:
                subs = block_t_indices[cand]
                valid = block_t_valid[cand]
                newly_covered = ~covered[subs[valid]]
                gain = np.sum(newly_covered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
                    # Tie-breaker: prefer blocks covering rarer triples
                    if np.any(newly_covered):
                        potential = np.sum(1.0 / (1.0 + cover_count[subs[valid][newly_covered]])**2)
                    else:
                        potential = 0.0
                    best_score_val = gain + potential * 0.1
                elif gain == best_gain:
                    if np.any(newly_covered):
                        potential = np.sum(1.0 / (1.0 + cover_count[subs[valid][newly_covered]])**2)
                    else:
                        potential = 0.0
                    score_val = gain + potential * 0.1
                    if score_val > best_score_val:
                        best_score_val = score_val
                        best_candidate = cand
            
            if best_candidate == -1:
                break
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            valid_mask = block_t_valid[best_candidate]
            newly_covered = ~covered[subs_to_cover[valid_mask]]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_count[subs_to_cover[valid_mask]] += 1
            covered[subs_to_cover[valid_mask]] = True
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx][block_t_valid[b_idx]]] += 1
        
        # Fast deletion
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx][block_t_valid[b_idx]]] += 1
        
        # Swap local search
        current_blocks, coverage_count = swap_local_search(current_blocks, coverage_count)
        
        # Final deletion pass
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    def run_standard_greedy(seed):
        """Standard greedy with potential-based scoring."""
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        cover_count = np.zeros(n_ts, dtype=np.float64)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Compute gain: number of new t-subsets covered
            covered_counts = np.zeros(n_ks, dtype=np.int32)
            for i in range(n_ks):
                valid = block_t_valid[i]
                covered_counts[i] = np.sum(covered[block_t_indices[i][valid]])
            
            gains = t_count - covered_counts
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 80:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(80, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                valid = block_t_valid[cand]
                newly_covered = ~covered[subs[valid]]
                if np.any(newly_covered):
                    potential = np.sum(1.0 / (1.0 + cover_count[subs[valid][newly_covered]])**2)
                else:
                    potential = 0.0
                score_cand = max_gain + potential * 0.5
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            valid_mask = block_t_valid[best_candidate]
            newly_covered = ~covered[subs_to_cover[valid_mask]]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_count[subs_to_cover[valid_mask]] += 1
            covered[subs_to_cover[valid_mask]] = True
            
        current_blocks = list(blocks_indices)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx][block_t_valid[b_idx]]] += 1
        
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx][block_t_valid[b_idx]]] += 1
        
        current_blocks, coverage_count = swap_local_search(current_blocks, coverage_count)
        
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        
        # Alternate between peeling and standard greedy
        if restart_count % 2 == 0:
            blocks, score = run_peeling_greedy(seed)
        else:
            blocks, score = run_standard_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_peeling_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            blocks, score = run_standard_greedy(0)
            if blocks is not None:
                best_solution = blocks
                best_score = score
            else:
                return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
