import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _precompute(v, k, t, time_budget):
    """Precompute all t-subsets, all k-blocks, and incidence structures."""
    start = time.time()
    
    # Generate t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Build incidence: for each k-block, which t-subsets does it cover
    block_t_sets = []
    for block in k_blocks:
        covered = frozenset(t_index[c] for c in itertools.combinations(block, t))
        block_t_sets.append(covered)
    
    # Build reverse incidence: for each t-subset, which k-blocks cover it
    t_to_blocks = defaultdict(list)
    for i, covered in enumerate(block_t_sets):
        for idx in covered:
            t_to_blocks[idx].append(i)
    t_to_blocks = [tuple(t_to_blocks[i]) for i in range(num_t)]
    
    elapsed = time.time() - start
    remaining = max(0, time_budget - elapsed)
    
    return {
        't_subsets': t_subsets,
        'k_blocks': k_blocks,
        'block_t_sets': block_t_sets,
        't_to_blocks': t_to_blocks,
        'num_t': num_t,
        'num_k': num_k,
        'remaining': remaining,
        'elapsed': elapsed
    }


def _greedy_cover(data, weights, deadline, rng):
    """Run weighted greedy set cover with optional randomization."""
    num_t = data['num_t']
    num_k = data['num_k']
    block_t_sets = data['block_t_sets']
    t_to_blocks = data['t_to_blocks']
    
    # Initialize gain for each block
    gain = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_sets[i]:
            gain[i] = np.sum(weights[list(block_t_sets[i])])
    
    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)
    
    while uncovered_count > 0 and time.time() < deadline:
        # Add small random perturbation for diversity
        if rng is not None:
            perturbation = rng.uniform(0, 0.1) * (1.0 / (num_k + 1))
            gain_no_mask = gain + perturbation
        else:
            gain_no_mask = gain
        
        gain_with_mask = gain_no_mask.copy()
        gain_with_mask[selected_mask] = -np.inf
        
        best_idx = int(np.argmax(gain_with_mask))
        best_gain = gain_with_mask[best_idx]
        
        if np.isinf(best_gain) or best_gain <= 0:
            break
        
        selected.append(best_idx)
        selected_mask[best_idx] = True
        
        # Update uncovered and gains
        for idx in block_t_sets[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = weights[idx]
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w
    
    return selected


def _remove_redundant(selected, block_t_sets, num_t, deadline):
    """Remove redundant blocks from the selected set."""
    if not selected or time.time() > deadline:
        return selected
    
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_sets[i]:
            cover_count[idx] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for i in selected:
            needed = False
            for idx in block_t_sets[i]:
                if cover_count[idx] == 1:
                    needed = True
                    break
            if needed:
                new_selected.append(i)
            else:
                changed = True
                for idx in block_t_sets[i]:
                    cover_count[idx] -= 1
        if changed:
            selected = new_selected
    
    return selected


def _benders_refine(selected, block_t_sets, t_to_blocks, num_t, num_k, deadline, rng):
    """
    Benders-style iterative refinement:
    1. Try removing each block; if coverage breaks, record which t-subsets are lost (no-good cut)
    2. Use cut information to guide which blocks are most critical
    3. Attempt targeted swaps using cut-guided priority
    """
    if not selected or time.time() > deadline:
        return selected
    
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_sets[i]:
            cover_count[idx] += 1
    
    selected_set = set(selected)
    improved = True
    
    # Track "criticality" - how many t-subsets each block uniquely covers
    criticality = np.zeros(num_k, dtype=np.int32)
    for i in selected:
        for idx in block_t_sets[i]:
            if cover_count[idx] == 1:
                criticality[i] += 1
    
    while improved and time.time() < deadline:
        improved = False
        current_size = len(selected)
        
        # Phase 1: Remove blocks with zero criticality
        for i in list(selected):
            if time.time() > deadline:
                break
            if criticality[i] == 0:
                # Safe to remove
                for idx in block_t_sets[i]:
                    cover_count[idx] -= 1
                    if cover_count[idx] == 1:
                        # Find which block now uniquely covers this
                        for b in t_to_blocks[idx]:
                            if b in selected_set and b != i:
                                criticality[b] += 1
                                break
                selected.remove(i)
                selected_set.discard(i)
                criticality[i] = 0
                improved = True
        
        if time.time() > deadline:
            break
        
        # Phase 2: Benders-style swap - try to replace low-criticality blocks
        # with blocks that cover more unique t-subsets
        if len(selected) > 0 and time.time() < deadline:
            # Sort selected by criticality (try removing low-criticality first)
            candidates = sorted(selected, key=lambda x: criticality[x])[:15]
            
            for rem_idx in candidates:
                if time.time() > deadline:
                    break
                if rem_idx not in selected:
                    continue
                if criticality[rem_idx] > 0:
                    continue  # Can't remove blocks with unique coverage
                
                # Temporarily remove
                for idx in block_t_sets[rem_idx]:
                    cover_count[idx] -= 1
                
                # Check if coverage is still complete
                if np.all(cover_count > 0):
                    # Find best replacement block
                    best_gain = 0
                    best_add = -1
                    
                    # Use a scoring that favors blocks covering many t-subsets
                    # with low current coverage (reduces redundancy)
                    for b in range(num_k):
                        if b in selected_set:
                            continue
                        gain = 0
                        for idx in block_t_sets[b]:
                            if cover_count[idx] == 0:
                                gain += 100  # Critical: covers uncovered
                            elif cover_count[idx] == 1:
                                gain += 10   # Helps reduce uniqueness risk
                            else:
                                gain += 1    # Mild benefit
                        if gain > best_gain:
                            best_gain = gain
                            best_add = b
                    
                    if best_add >= 0 and best_gain > 0:
                        # Perform swap
                        selected.remove(rem_idx)
                        selected_set.discard(rem_idx)
                        selected.append(best_add)
                        selected_set.add(best_add)
                        
                        # Update cover counts and criticality
                        for idx in block_t_sets[best_add]:
                            old_count = cover_count[idx]
                            cover_count[idx] += 1
                            if old_count == 1:
                                # This t-subset was uniquely covered, now it's not
                                # Find which block had the unique coverage
                                for b in t_to_blocks[idx]:
                                    if b in selected_set and b != best_add and cover_count[idx] > 1:
                                        # Can't easily track which specific block had it
                                        pass
                            elif old_count == 0:
                                # Was uncovered, now covered by best_add uniquely
                                criticality[best_add] += 1
                        
                        # Recompute criticality for affected blocks
                        for idx in block_t_sets[best_add]:
                            if cover_count[idx] == 1:
                                for b in t_to_blocks[idx]:
                                    if b in selected_set:
                                        criticality[b] = 1
                                        break
                            
                        improved = True
                else:
                    # Restore
                    for idx in block_t_sets[rem_idx]:
                        cover_count[idx] += 1
    
    return _remove_redundant(selected, block_t_sets, num_t, deadline)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute
    data = _precompute(v, k, t, 20)
    num_t = data['num_t']
    num_k = data['num_k']
    block_t_sets = data['block_t_sets']
    t_to_blocks = data['t_to_blocks']
    k_blocks = data['k_blocks']
    
    if elapsed() > deadline - 30:
        # Emergency fallback: simple greedy
        weights = np.ones(num_t, dtype=np.float64)
        selected = _greedy_cover(data, weights, deadline - 5, None)
        selected = _remove_redundant(selected, block_t_sets, num_t, deadline - 2)
        return [list(k_blocks[i]) for i in selected]
    
    # Compute rarity-based weights
    t_cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        for idx in block_t_sets[i]:
            t_cover_count[idx] += 1
    
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    base_weights = 1.0 / safe_counts
    
    best_selected = None
    best_count = float('inf')
    
    # Multiple restarts with different weight strategies
    max_restarts = 6
    restart_time_budget = (deadline - 50) / max_restarts
    
    for restart in range(max_restarts):
        if elapsed() > deadline - 40:
            break
        
        # Create diverse weights
        if restart == 0:
            weights = base_weights.copy()
        elif restart == 1:
            # Boost rare t-subsets more
            weights = base_weights ** 2
        elif restart == 2:
            # Uniform weights
            weights = np.ones(num_t, dtype=np.float64)
        elif restart == 3:
            # Inverse frequency with stronger exponent
            weights = base_weights ** 3
        else:
            # Random perturbation of base weights
            rng = random.Random(restart * 12345 + 67890)
            perturbation = np.array([rng.uniform(0.5, 1.5) for _ in range(num_t)])
            weights = base_weights * perturbation
        
        # Run greedy with optional randomization
        rng = random.Random(restart * 98765 + 43210) if restart > 0 else None
        selected = _greedy_cover(data, weights, deadline - 35, rng)
        
        # Remove redundant blocks
        selected = _remove_redundant(selected, block_t_sets, num_t, deadline - 20)
        
        # Benders-style refinement
        if elapsed() < deadline - 15 and len(selected) < best_count:
            selected = _benders_refine(
                selected, block_t_sets, t_to_blocks, num_t, num_k,
                deadline - 5, random.Random(restart * 54321 + 10987)
            )
        
        # Check coverage
        if selected:
            cover_check = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_sets[i]:
                    cover_check[idx] += 1
            
            if np.all(cover_check > 0) and len(selected) < best_count:
                best_count = len(selected)
                best_selected = selected[:]
        
        if elapsed() > deadline - 35:
            break
    
    # Final high-precision polish: multiple passes of redundancy removal
    if best_selected is not None and elapsed() < deadline - 10:
        polish_deadline = deadline - 5
        for polish_pass in range(3):
            if elapsed() > polish_deadline:
                break
            best_selected = _remove_redundant(best_selected, block_t_sets, num_t, polish_deadline)
            if elapsed() < polish_deadline - 5:
                best_selected = _benders_refine(
                    best_selected, block_t_sets, t_to_blocks, num_t, num_k,
                    polish_deadline, random.Random(99999 + polish_pass)
                )
    
    # Final fallback if no valid solution found
    if best_selected is None:
        weights = base_weights.copy()
        selected = _greedy_cover(data, weights, deadline - 5, None)
        selected = _remove_redundant(selected, block_t_sets, num_t, deadline - 2)
        best_selected = selected
    
    return [list(k_blocks[i]) for i in best_selected]
