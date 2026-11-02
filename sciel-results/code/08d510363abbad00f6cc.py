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
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # cover_matrix: (num_blocks, C(k,t)) with triple indices
    c_k_t = len(list(itertools.combinations(range(k), t)))
    cover_matrix = np.zeros((num_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triple_to_idx[triple]
    
    # blocks_covering_triple: for each triple, list of block indices
    blocks_covering_triple = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        for j in range(c_k_t):
            t_idx = cover_matrix[i, j]
            blocks_covering_triple[t_idx].append(i)
    
    _PRECOMP_CACHE[key] = (all_blocks, cover_matrix, blocks_covering_triple, num_triples, num_blocks, c_k_t)
    return _PRECOMP_CACHE[key]

def _solve_residual_milp(residual_triples, cover_matrix, blocks_covering_triple, num_triples, time_limit):
    """Solve the residual covering problem using MILP."""
    if not residual_triples:
        return []
    
    residual_set = set(residual_triples)
    num_residual = len(residual_set)
    
    # Find candidate blocks that cover at least one residual triple
    candidate_blocks = set()
    for t_idx in residual_set:
        for b_idx in blocks_covering_triple[t_idx]:
            candidate_blocks.add(b_idx)
    
    candidate_blocks = sorted(candidate_blocks)
    num_candidates = len(candidate_blocks)
    
    if num_candidates == 0:
        return []
    
    # Build incidence matrix
    A = np.zeros((num_residual, num_candidates), dtype=np.int8)
    residual_to_local = {t_idx: i for i, t_idx in enumerate(sorted(residual_set))}
    
    for j, b_idx in enumerate(candidate_blocks):
        for t_idx in cover_matrix[b_idx]:
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
            selected = [candidate_blocks[j] for j in range(num_candidates) if result.x[j] > 0.5]
            return selected
    except Exception:
        pass
    
    return []

def _vectorized_greedy(cover_matrix, blocks_covering_triple, num_triples, num_blocks, rng, time_limit):
    """Fast vectorized greedy construction."""
    uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    
    while uncovered.any() and time.time() < time_limit:
        # Vectorized gain calculation
        gains = np.sum(uncovered[cover_matrix], axis=1)
        max_gain = np.max(gains)
        
        if max_gain == 0:
            break
        
        candidates = np.where(gains == max_gain)[0]
        chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
        
        selected.append(chosen_idx)
        uncovered[cover_matrix[chosen_idx]] = False
    
    return selected, set(np.where(uncovered)[0].tolist())

def _polish_aggressive(selected_blocks, cover_matrix, num_triples, deadline):
    """Aggressive local search: remove redundant blocks, then try swaps."""
    if not selected_blocks:
        return selected_blocks
    
    coverage = np.zeros(num_triples, dtype=np.int32)
    for b_idx in selected_blocks:
        coverage[cover_matrix[b_idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        to_remove = []
        
        for b_idx in selected_blocks:
            if not np.any(coverage[cover_matrix[b_idx]] == 1):
                to_remove.append(b_idx)
        
        if to_remove:
            remove_set = set(to_remove)
            selected_blocks = [b for b in selected_blocks if b not in remove_set]
            coverage = np.zeros(num_triples, dtype=np.int32)
            for b_idx in selected_blocks:
                coverage[cover_matrix[b_idx]] += 1
            improved = True
    
    # Try swap-based refinement: for blocks with unique triples, try replacing with better ones
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            coverage = np.zeros(num_triples, dtype=np.int32)
            for b_idx in selected_blocks:
                coverage[cover_matrix[b_idx]] += 1
            
            any_swap = False
            for i, b_idx in enumerate(selected_blocks):
                unique_in_block = cover_matrix[b_idx][coverage[cover_matrix[b_idx]] == 1]
                if len(unique_in_block) == 0:
                    continue
                
                # Try to find a replacement that covers all unique triples
                best_replace = -1
                best_gain = -1
                for cand in blocks_covering_triple[unique_in_block[0]]:
                    if cand in selected_blocks:
                        continue
                    # Check if cand covers all unique_in_block
                    if np.all(np.isin(unique_in_block, cover_matrix[cand])):
                        # Calculate net gain: how many new triples cand covers vs b_idx
                        gain = np.sum(coverage[cover_matrix[cand]] == 0) - np.sum(coverage[cover_matrix[b_idx]] == 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_replace = cand
                
                if best_replace != -1 and best_gain > 0:
                    selected_blocks[i] = best_replace
                    coverage = np.zeros(num_triples, dtype=np.int32)
                    for b_idx2 in selected_blocks:
                        coverage[cover_matrix[b_idx2]] += 1
                    any_swap = True
                    break
            
            if not any_swap:
                break
    
    return selected_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_blocks, cover_matrix, blocks_covering_triple, num_triples, num_blocks, c_k_t = _get_precomputed(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    # Phase 1: Multiple vectorized greedy attempts with MILP completion
    attempts = 0
    while time.time() < deadline - 30.0:
        rng = random.Random()
        
        greedy_time_limit = time.time() + 3.0
        selected_blocks, uncovered = _vectorized_greedy(
            cover_matrix, blocks_covering_triple, num_triples, num_blocks, rng, greedy_time_limit
        )
        
        # Polish
        selected_blocks = _polish_aggressive(selected_blocks, cover_matrix, num_triples, deadline)
        
        # If still uncovered, try MILP
        if uncovered and time.time() < deadline - 15.0:
            milp_time = min(8.0, deadline - time.time() - 5.0)
            if milp_time > 1.0:
                milp_sol = _solve_residual_milp(
                    list(uncovered), cover_matrix, blocks_covering_triple, num_triples, milp_time
                )
                if milp_sol:
                    combined = selected_blocks + [b for b in milp_sol if b not in selected_blocks]
                    combined = _polish_aggressive(combined, cover_matrix, num_triples, deadline)
                    if len(combined) < len(selected_blocks):
                        selected_blocks = combined
        
        # Final polish
        selected_blocks = _polish_aggressive(selected_blocks, cover_matrix, num_triples, deadline)
        
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            attempts += 1
            
            if best_size <= 69:
                break
    
    # Phase 2: More greedy attempts with different random seeds
    while time.time() < deadline - 5.0:
        rng = random.Random()
        
        greedy_time_limit = time.time() + 5.0
        selected_blocks, uncovered = _vectorized_greedy(
            cover_matrix, blocks_covering_triple, num_triples, num_blocks, rng, greedy_time_limit
        )
        
        selected_blocks = _polish_aggressive(selected_blocks, cover_matrix, num_triples, deadline)
        
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            
            if best_size <= 69:
                break
    
    if best_solution is None:
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_solution]
