import numpy as np
import itertools
import time
import random
from collections import Counter
from scipy.optimize import milp, LinearConstraint, Bounds

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
    
    # Build reverse index: for each t-subset, which blocks cover it
    # This enables fast swap checks
    t_to_blocks = [set() for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            t_to_blocks[block_t_indices[i, j]].add(i)
    
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
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
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
            
            # For each pair of blocks, check if a single block can replace them
            # Only try pairs where the union of their unique t-subsets is small
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
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]
                    
                    # Unique to b1: covered by b1 but not by any other block except b2
                    # Unique to b2: covered by b2 but not by any other block except b1
                    # A replacement block must cover all of these
                    unique_to_1 = []
                    unique_to_2 = []
                    for sub in subs1:
                        if coverage_count[sub] == 1:
                            unique_to_1.append(sub)
                    for sub in subs2:
                        if coverage_count[sub] == 1:
                            unique_to_2.append(sub)
                    
                    if not unique_to_1 and not unique_to_2:
                        # Both blocks are redundant
                        improved = True
                        coverage_count[subs1] -= 1
                        coverage_count[subs2] -= 1
                        current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                        blocks_set.discard(b1)
                        blocks_set.discard(b2)
                        break
                    elif unique_to_1 and not unique_to_2:
                        # Need a block covering all of unique_to_1
                        # Find candidate blocks that cover all unique_to_1
                        candidates = t_to_blocks[unique_to_1[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                # Check if cand covers all unique_to_1
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_1):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    elif unique_to_2 and not unique_to_1:
                        # Need a block covering all of unique_to_2
                        candidates = t_to_blocks[unique_to_2[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_2):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    else:
                        # Both have unique subsets - skip (too complex)
                        pass
            if improved:
                # Rebuild coverage_count after modification
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1
                blocks_set = set(current_blocks)
        return current_blocks, coverage_count
    
    def solve_residual_milp(current_blocks, coverage_count, time_budget=15.0):
        """Solve the residual covering problem using scipy.optimize.milp.
        
        After greedy + local search, some t-subsets may still be uncovered or
        we may be able to remove redundant blocks. This function identifies
        candidate blocks (those that cover at least one currently-undercovered
        or singly-covered t-subset) and solves a small ILP to find the minimum
        set of additional/replacement blocks.
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit - time_budget - 2.0:
            return current_blocks, coverage_count
        
        # Identify which t-subsets need more coverage (count == 0 means uncovered,
        # but after greedy all should be covered; we look for improvement)
        # Strategy: try to find a smaller covering.
        # Identify "critical" t-subsets: those covered by exactly 1 block.
        critical_mask = (coverage_count == 1)
        n_critical = np.sum(critical_mask)
        
        if n_critical == 0:
            return current_blocks, coverage_count
        
        # Get indices of critical t-subsets
        critical_idx = np.where(critical_mask)[0]
        
        # For each critical t-subset, find which blocks cover it
        # Build a list of candidate blocks: any block that covers at least one critical t-subset
        # plus the blocks currently covering critical t-subsets
        
        # First, identify which blocks are "essential" (cover at least one critical t-subset)
        block_is_essential = np.zeros(n_ks, dtype=bool)
        for ci in critical_idx:
            blocks_covering = t_to_blocks[ci]
            for b in blocks_covering:
                block_is_essential[b] = True
        
        # Candidate blocks: essential blocks + blocks that could replace them
        # To keep the ILP small, limit candidates to blocks covering critical subsets
        # plus some random additional blocks
        essential_blocks = np.where(block_is_essential)[0]
        
        # Also add blocks that cover multiple critical subsets (good candidates)
        # Score each block by how many critical subsets it covers
        critical_coverage = np.zeros(n_ks, dtype=np.int32)
        for b in essential_blocks:
            subs = block_t_indices[b]
            critical_coverage[b] = np.sum(critical_mask[subs])
        
        # Select top candidates: blocks with highest critical coverage
        # Limit to at most 200 candidates to keep ILP tractable
        max_candidates = 200
        if len(essential_blocks) > max_candidates:
            # Sort by critical coverage, take top ones
            sorted_blocks = essential_blocks[np.argsort(-critical_coverage[essential_blocks])]
            candidate_blocks = sorted_blocks[:max_candidates]
        else:
            candidate_blocks = essential_blocks
        
        # Also include some random blocks for diversity
        n_random = min(50, n_ks - len(candidate_blocks))
        if n_random > 0:
            random_blocks = random.sample(range(n_ks), n_random)
            existing_set = set(candidate_blocks.tolist())
            for rb in random_blocks:
                if rb not in existing_set:
                    candidate_blocks = np.append(candidate_blocks, rb)
        
        n_cand = len(candidate_blocks)
        
        if n_cand == 0:
            return current_blocks, coverage_count
        
        # Build the constraint matrix: for each critical t-subset, require at least 1 block covers it
        # A[i, j] = 1 if candidate block j covers critical t-subset i
        A = np.zeros((n_critical, n_cand), dtype=np.int32)
        for i, ci in enumerate(critical_idx):
            for j, b in enumerate(candidate_blocks):
                if ci in t_to_blocks[ci] and b in t_to_blocks[ci]:
                    A[i, j] = 1
        
        # Remove rows that are all zeros (critical subsets not coverable by any candidate)
        # This shouldn't happen since essential blocks are included, but be safe
        row_sums = A.sum(axis=1)
        valid_rows = row_sums > 0
        if not np.all(valid_rows):
            A = A[valid_rows]
            n_critical_valid = A.shape[0]
        else:
            n_critical_valid = n_critical
        
        if A.shape[0] == 0:
            return current_blocks, coverage_count
        
        # Objective: minimize number of blocks selected
        c = np.ones(n_cand)
        
        # Constraints: A @ x >= 1 for each row, 0 <= x <= 1, x integer
        constraints = LinearConstraint(A, 1, np.inf)
        bounds = Bounds(0, 1)
        
        # Solve
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=np.ones(n_cand),
                bounds=bounds
            )
        except Exception:
            return current_blocks, coverage_count
        
        if not result.success:
            return current_blocks, coverage_count
        
        selected = np.where(result.x > 0.5)[0]
        
        if len(selected) == 0:
            return current_blocks, coverage_count
        
        # Build new solution: keep non-essential blocks, replace essential blocks with selected ones
        # Actually, we need to be careful: the ILP only ensures critical subsets are covered.
        # We need to verify the full solution is still valid.
        
        new_blocks_set = set()
        
        # Keep all blocks that are NOT in the candidate set (they're fine)
        cand_set = set(candidate_blocks.tolist())
        for b in current_blocks:
            if b not in cand_set:
                new_blocks_set.add(b)
        
        # Add selected candidate blocks
        for idx in selected:
            new_blocks_set.add(candidate_blocks[idx])
        
        # Verify: check if all t-subsets are still covered
        # Build new coverage
        new_coverage = np.zeros(n_ts, dtype=np.int32)
        for b in new_blocks_set:
            new_coverage[block_t_indices[b]] += 1
        
        if np.all(new_coverage >= 1):
            # Valid! Use the new solution if it's smaller
            if len(new_blocks_set) < len(current_blocks):
                new_blocks = list(new_blocks_set)
                # Run fast deletion on the new solution
                new_blocks = fast_deletion(new_blocks, new_coverage)
                return new_blocks, new_coverage
            else:
                return current_blocks, coverage_count
        else:
            # Not valid, return original
            return current_blocks, coverage_count
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
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
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 80:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(80, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            # Score candidates: gain + bonus for covering rare subsets
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
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
        
        # Fast deletion
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Swap local search
        current_blocks, coverage_count = swap_local_search(current_blocks, coverage_count)
        
        # Final deletion pass
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        # MILP residual solve
        if time.time() - start_time < time_limit - 8.0:
            current_blocks, coverage_count = solve_residual_milp(current_blocks, coverage_count)
        
        # Final cleanup
        if time.time() - start_time < time_limit - 3.0:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            current_blocks = fast_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 12.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            # If we've found a good solution, do a final polish
            if best_score <= 78 and time.time() - start_time < time_limit - 20.0:
                # Try more restarts with the best solution as starting point
                pass
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    # Final verification and cleanup
    if time.time() - start_time < time_limit - 3.0:
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        best_solution = fast_deletion(best_solution, coverage_count)
    
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
