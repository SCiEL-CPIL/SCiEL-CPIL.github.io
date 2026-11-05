import random
import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # 2. Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # 3. Build sparse coverage structure: for each block, list of t-subset indices it covers
    block_covered = []
    for i, block in enumerate(all_blocks):
        covered = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_covered.append(covered)

    # Build inverse: for each t-subset, which blocks cover it
    t_sub_covered_by = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_covered):
        for idx in covered:
            t_sub_covered_by[idx].append(i)

    # 4. DLS (Distributed Local Search) on dual hypergraph
    # State: binary vector indicating selected blocks
    # Objective: minimize number of selected blocks while covering all t-subsets
    
    best_solution = None
    best_score = float('inf')
    rng = np.random.default_rng(42)

    # Precompute some useful arrays
    # coverage_count[i] = how many selected blocks cover t-subset i
    # uncovered_count = number of uncovered t-subsets

    def evaluate_solution(selected_set):
        """Return (num_blocks, uncovered_count) for a given set of selected block indices"""
        if not selected_set:
            return 0, num_t
        coverage = np.zeros(num_t, dtype=np.int32)
        for b in selected_set:
            for idx in block_covered[b]:
                coverage[idx] += 1
        uncovered = np.sum(coverage == 0)
        return len(selected_set), uncovered

    # Initialize with a greedy solution to have a good starting point
    uncovered = np.ones(num_t, dtype=np.uint8)
    greedy_selected = []
    
    while uncovered.any():
        if time.time() > deadline - 60.0:
            break
        # Find block with max gain
        best_gain = 0
        best_block = -1
        for b in range(num_blocks):
            gain = 0
            for idx in block_covered[b]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
        if best_block == -1 or best_gain == 0:
            break
        greedy_selected.append(best_block)
        for idx in block_covered[best_block]:
            uncovered[idx] = 0

    # Convert to set for faster membership
    initial_selected = set(greedy_selected)
    num_sel, num_uncov = evaluate_solution(initial_selected)
    best_solution = set(initial_selected)
    best_score = num_sel
    current_selected = set(initial_selected)
    current_num_sel = num_sel
    current_num_uncov = num_uncov

    # DLS iteration
    # Penalty: for each t-subset, if uncovered, add penalty to all blocks that could cover it
    # Weighted by how rare the coverage is (fewer blocks covering it = higher weight)
    
    # Precompute rarity weight for each t-subset
    rarity_weight = np.zeros(num_t, dtype=np.float64)
    for i in range(num_t):
        num_coverers = len(t_sub_covered_by[i])
        rarity_weight[i] = 1.0 / max(num_coverers, 1)

    max_iters = 50000
    iter_count = 0
    
    while time.time() < deadline - 10.0 and iter_count < max_iters:
        iter_count += 1
        
        if current_num_uncov == 0:
            # All covered, try to remove blocks
            if current_num_sel == 0:
                break
            # Try removing a random block
            candidates = list(current_selected)
            random.shuffle(candidates)
            removed_any = False
            for b in candidates:
                if time.time() > deadline - 10.0:
                    break
                # Check if removing b leaves any t-subset uncovered
                # A t-subset is still covered if at least one other selected block covers it
                can_remove = True
                for idx in block_covered[b]:
                    # Check if any other selected block covers idx
                    other_covers = False
                    for other in t_sub_covered_by[idx]:
                        if other != b and other in current_selected:
                            other_covers = True
                            break
                    if not other_covers:
                        can_remove = False
                        break
                if can_remove:
                    current_selected.remove(b)
                    current_num_sel -= 1
                    removed_any = True
                    if current_num_sel < best_score:
                        best_score = current_num_sel
                        best_solution = set(current_selected)
                    break
            if not removed_any:
                # Try adding then removing? Or just continue
                pass
        else:
            # Have uncovered t-subsets, need to add blocks
            # Use DLS scoring: for each candidate block, compute score based on
            # - how many uncovered t-subsets it covers (reward)
            # - how many already-covered t-subsets it covers (penalty, weighted by rarity)
            
            # Get list of uncovered t-subset indices
            uncovered_indices = [i for i in range(num_t) if current_num_uncov > 0 and _is_uncovered(i, current_selected)]
            
            if not uncovered_indices:
                # Recount
                coverage = np.zeros(num_t, dtype=np.int32)
                for b in current_selected:
                    for idx in block_covered[b]:
                        coverage[idx] += 1
                uncovered_indices = [i for i in range(num_t) if coverage[i] == 0]
                current_num_uncov = len(uncovered_indices)
            
            if not uncovered_indices:
                continue
            
            # Score each potential block to add
            # We only consider blocks that cover at least one uncovered t-subset
            candidate_blocks = set()
            for idx in uncovered_indices:
                for b in t_sub_covered_by[idx]:
                    candidate_blocks.add(b)
            
            if not candidate_blocks:
                break
            
            best_block = -1
            best_score_val = float('-inf')
            
            for b in candidate_blocks:
                if b in current_selected:
                    continue
                # Compute DLS score
                score = 0.0
                for idx in block_covered[b]:
                    # Is this t-subset currently uncovered?
                    is_uncov = False
                    cov_count = 0
                    for other in t_sub_covered_by[idx]:
                        if other in current_selected:
                            cov_count += 1
                    is_uncov = (cov_count == 0)
                    
                    if is_uncov:
                        # Reward for covering uncovered, weighted by rarity
                        score += rarity_weight[idx] * 10.0
                    else:
                        # Penalty for covering already-covered, weighted by rarity
                        score -= rarity_weight[idx] * 1.0
                
                # Add small random tie-breaker
                score += rng.random() * 0.1
                
                if score > best_score_val:
                    best_score_val = score
                    best_block = b
            
            if best_block == -1:
                break
            
            # Add the block
            current_selected.add(best_block)
            current_num_sel += 1
            
            # Recount uncovered
            coverage = np.zeros(num_t, dtype=np.int32)
            for b in current_selected:
                for idx in block_covered[b]:
                    coverage[idx] += 1
            current_num_uncov = np.sum(coverage == 0)
            
            if current_num_uncov == 0 and current_num_sel < best_score:
                best_score = current_num_sel
                best_solution = set(current_selected)

    # 5. Redundancy Removal on best solution
    if best_solution is None or not best_solution:
        return []

    final_selected = set(best_solution)
    
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for b in list(final_selected):
            if time.time() > deadline - 5.0:
                break
            # Check if b is essential
            is_essential = False
            for idx in block_covered[b]:
                # Check if any other selected block covers idx
                other_covers = False
                for other in t_sub_covered_by[idx]:
                    if other != b and other in final_selected:
                        other_covers = True
                        break
                if not other_covers:
                    is_essential = True
                    break
            if not is_essential:
                final_selected.remove(b)
                changed = True

    final_blocks = [list(all_blocks[i]) for i in sorted(final_selected)]
    return final_blocks


def _is_uncovered(idx, selected_set):
    """Check if t-subset idx is uncovered by any block in selected_set"""
    for b in t_sub_covered_by[idx]:
        if b in selected_set:
            return False
    return True
