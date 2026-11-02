import itertools
import random
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_PRECOMP_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[key]
    
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = []
    block_triples = []
    blocks_covering_triple = [[] for _ in range(num_triples)]
    
    # Precompute all k-subsets
    for block in itertools.combinations(points, k):
        block_id = len(all_blocks)
        all_blocks.append(block)
        
        block_triple_indices = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            block_triple_indices.append(idx)
            blocks_covering_triple[idx].append(block_id)
        
        block_triples.append(block_triple_indices)
    
    # Build vectorized cover matrix: cover_matrix[block_idx] = array of triple indices
    c_k_t = len(block_triples[0]) if block_triples else 0
    cover_matrix = np.zeros((len(all_blocks), c_k_t), dtype=np.int32)
    for i, btrips in enumerate(block_triples):
        cover_matrix[i] = btrips
    
    _PRECOMP_CACHE[key] = (all_blocks, block_triples, blocks_covering_triple, num_triples, cover_matrix)
    return _PRECOMP_CACHE[key]

def _vectorized_greedy(v, k, t, all_blocks, block_triples, blocks_covering_triple, num_triples, cover_matrix, rng, time_limit):
    """Fast vectorized greedy: pick block with max uncovered coverage."""
    uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    used = set()
    
    while uncovered.any() and time.time() < time_limit:
        # Vectorized: count uncovered triples covered by each block
        gains = np.sum(uncovered[cover_matrix], axis=1)
        
        # Mask out already-used blocks
        for b_idx in used:
            gains[b_idx] = 0
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        
        candidates = np.where(gains == max_gain)[0]
        chosen = int(candidates[rng.randint(0, len(candidates) - 1)])
        
        selected.append(chosen)
        used.add(chosen)
        uncovered[cover_matrix[chosen]] = False
    
    return selected, set(np.where(uncovered)[0].tolist())

def _greedy_construct(v, k, t, all_blocks, block_triples, blocks_covering_triple, num_triples, rng, time_limit):
    """Randomized greedy: pick random uncovered triple, then best block for it."""
    uncovered = set(range(num_triples))
    selected_blocks = []
    used_blocks = set()
    
    while uncovered and time.time() < time_limit:
        chosen_triple = rng.choice(list(uncovered))
        candidates = blocks_covering_triple[chosen_triple]
        
        best_block_idx = -1
        best_score = -1
        best_tie_breakers = []
        
        for b_idx in candidates:
            if b_idx in used_blocks:
                continue
            score = sum(1 for t_idx in block_triples[b_idx] if t_idx in uncovered)
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
                best_tie_breakers = [b_idx]
            elif score == best_score and best_score > 0:
                best_tie_breakers.append(b_idx)
        
        if best_block_idx == -1:
            break
        
        if len(best_tie_breakers) > 1:
            best_block_idx = rng.choice(best_tie_breakers)
        
        selected_blocks.append(best_block_idx)
        used_blocks.add(best_block_idx)
        
        for t_idx in block_triples[best_block_idx]:
            if t_idx in uncovered:
                uncovered.remove(t_idx)
    
    return selected_blocks, uncovered

def _polish(selected_blocks, block_triples, num_triples):
    """Remove redundant blocks (those with no uniquely covered triple)."""
    if not selected_blocks:
        return selected_blocks
    
    polish_improved = True
    while polish_improved:
        polish_improved = False
        coverage = [0] * num_triples
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                coverage[t_idx] += 1
        
        to_remove = []
        for i, b_idx in enumerate(selected_blocks):
            essential = False
            for t_idx in block_triples[b_idx]:
                if coverage[t_idx] == 1:
                    essential = True
                    break
            if not essential:
                to_remove.append(i)
        
        if to_remove:
            remove_set = set(to_remove)
            selected_blocks = [b for i, b in enumerate(selected_blocks) if i not in remove_set]
            polish_improved = True
    
    return selected_blocks

def _swap_local_search(selected_blocks, block_triples, blocks_covering_triple, num_triples, time_limit, max_rounds=3):
    """Try to swap blocks to enable further removals."""
    if not selected_blocks:
        return selected_blocks
    
    for _ in range(max_rounds):
        if time.time() > time_limit:
            break
        
        coverage = [0] * num_triples
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                coverage[t_idx] += 1
        
        # Find blocks with unique triples
        unique_blocks = []
        for i, b_idx in enumerate(selected_blocks):
            unique_triples = [t_idx for t_idx in block_triples[b_idx] if coverage[t_idx] == 1]
            if unique_triples:
                unique_blocks.append((i, b_idx, unique_triples))
        
        if not unique_blocks:
            break
        
        improved = False
        # For each block with unique triples, try to find a replacement
        for i, b_idx, unique_triples in unique_blocks:
            if time.time() > time_limit:
                break
            
            # Find blocks that cover all unique_triples
            for candidate in blocks_covering_triple[unique_triples[0]]:
                if candidate in selected_blocks:
                    continue
                
                # Check if candidate covers all unique triples
                cand_triples = set(block_triples[candidate])
                if all(t in cand_triples for t in unique_triples):
                    # Check if swap enables removal of another block
                    # Remove b_idx, add candidate, see if any other block becomes redundant
                    new_blocks = [b for j, b in enumerate(selected_blocks) if j != i]
                    new_blocks.append(candidate)
                    
                    new_coverage = [0] * num_triples
                    for nb in new_blocks:
                        for t_idx in block_triples[nb]:
                            new_coverage[t_idx] += 1
                    
                    # Check if any block in new_blocks (except candidate) is now redundant
                    removable = False
                    for j, nb in enumerate(new_blocks):
                        if nb == candidate:
                            continue
                        has_unique = any(new_coverage[t_idx] == 1 for t_idx in block_triples[nb])
                        if not has_unique:
                            removable = True
                            break
                    
                    if removable:
                        # Perform swap and removal
                        new_blocks = [b for j, b in enumerate(new_blocks) if b != candidate]
                        # Remove the redundant one
                        for j, nb in enumerate(new_blocks):
                            has_unique = any(new_coverage[t_idx] == 1 for t_idx in block_triples[nb])
                            if not has_unique:
                                new_blocks.pop(j)
                                break
                        
                        selected_blocks = new_blocks
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return _polish(selected_blocks, block_triples, num_triples)

def _solve_residual_milp(residual_triples, all_blocks, block_triples, blocks_covering_triple, num_triples, time_limit):
    """Solve residual covering with MILP."""
    if not residual_triples:
        return []
    
    residual_set = set(residual_triples)
    num_residual = len(residual_set)
    
    candidate_blocks = set()
    for t_idx in residual_set:
        for b_idx in blocks_covering_triple[t_idx]:
            candidate_blocks.add(b_idx)
    
    candidate_blocks = sorted(candidate_blocks)
    num_candidates = len(candidate_blocks)
    
    if num_candidates == 0:
        return []
    
    A = np.zeros((num_residual, num_candidates), dtype=np.int8)
    residual_sorted = sorted(residual_set)
    residual_to_local = {t_idx: i for i, t_idx in enumerate(residual_sorted)}
    
    for j, b_idx in enumerate(candidate_blocks):
        for t_idx in block_triples[b_idx]:
            if t_idx in residual_to_local:
                A[residual_to_local[t_idx], j] = 1
    
    c = np.ones(num_candidates)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(num_candidates)
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                      options={'time_limit': time_limit, 'mip_gap': 0.0})
        if result.status == 0 and result.x is not None:
            return [candidate_blocks[j] for j in range(num_candidates) if result.x[j] > 0.5]
    except Exception:
        pass
    
    return []

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_blocks, block_triples, blocks_covering_triple, num_triples, cover_matrix = _get_precomputed(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    # Phase 1: Many fast vectorized greedy restarts
    restart = 0
    while time.time() < deadline - 40.0:
        rng = random.Random(restart)
        selected, uncovered = _vectorized_greedy(v, k, t, all_blocks, block_triples, 
                                                  blocks_covering_triple, num_triples, 
                                                  cover_matrix, rng, time.time() + 3.0)
        selected = _polish(selected, block_triples, num_triples)
        
        if uncovered:
            milp_time = min(8.0, deadline - time.time() - 10.0)
            if milp_time > 1.0:
                milp_sol = _solve_residual_milp(list(uncovered), all_blocks, block_triples,
                                                 blocks_covering_triple, num_triples, milp_time)
                if milp_sol:
                    combined = selected + [b for b in milp_sol if b not in selected]
                    combined = _polish(combined, block_triples, num_triples)
                    if len(combined) < len(selected):
                        selected = combined
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]
            if best_size <= 69:
                break
        
        restart += 1
    
    # Phase 2: Focused local search on best solution
    if best_solution and time.time() < deadline - 5.0:
        polished = _swap_local_search(best_solution[:], block_triples, blocks_covering_triple,
                                       num_triples, deadline - 5.0, max_rounds=5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = polished
    
    # Phase 3: A few more randomized greedy attempts with different strategy
    while time.time() < deadline - 5.0:
        rng = random.Random()
        selected, uncovered = _greedy_construct(v, k, t, all_blocks, block_triples,
                                                 blocks_covering_triple, num_triples,
                                                 rng, time.time() + 5.0)
        selected = _polish(selected, block_triples, num_triples)
        
        if uncovered:
            milp_time = min(5.0, deadline - time.time() - 5.0)
            if milp_time > 1.0:
                milp_sol = _solve_residual_milp(list(uncovered), all_blocks, block_triples,
                                                 blocks_covering_triple, num_triples, milp_time)
                if milp_sol:
                    combined = selected + [b for b in milp_sol if b not in selected]
                    combined = _polish(combined, block_triples, num_triples)
                    if len(combined) < len(selected):
                        selected = combined
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]
            if best_size <= 69:
                break
    
    if best_solution is None:
        return []
    
    return [list(all_blocks[b_idx]) for b_idx in best_solution]
