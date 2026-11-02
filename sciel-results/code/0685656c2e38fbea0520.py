import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

# Precomputation for v=24, t=3
V = 24
T = 3
K = 7

def _precompute():
    global triple_to_idx, idx_to_triple, all_triples, incidence
    all_triples = list(itertools.combinations(range(V), T))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    idx_to_triple = all_triples
    # Build incidence matrix: rows = triples, cols = blocks
    # For each triple, find all 7-blocks containing it
    n_triples = len(all_triples)
    all_blocks = list(itertools.combinations(range(V), K))
    n_blocks = len(all_blocks)
    
    # Build sparse incidence matrix
    row_idx = []
    col_idx = []
    global block_to_triples
    block_to_triples = {}
    for bi, block in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(block, T):
            ti = triple_to_idx[tuple(sorted(tri))]
            row_idx.append(ti)
            col_idx.append(bi)
            tris.append(ti)
        block_to_triples[tuple(block)] = np.array(tris, dtype=np.int32)
    
    data = np.ones(len(row_idx), dtype=np.int8)
    incidence = csr_matrix((data, (row_idx, col_idx)), shape=(n_triples, n_blocks))
    global all_blocks_list
    all_blocks_list = all_blocks

_precompute()

def _greedy(v, k, t, deadline, rng):
    """Fast greedy using vectorized gain computation"""
    start_time = time.time()
    uncovered = np.ones(len(idx_to_triple), dtype=bool)
    blocks = []
    
    while True:
        if time.time() > deadline:
            break
        # Find an uncovered triple
        uncovered_idx = np.where(uncovered)[0]
        if len(uncovered_idx) == 0:
            break
        
        seed_triple = idx_to_triple[uncovered_idx[0]]
        seed_set = set(seed_triple)
        
        # Find all blocks containing the seed triple
        candidate_blocks = []
        for block in all_blocks_list:
            if seed_set.issubset(set(block)):
                candidate_blocks.append(block)
        
        if not candidate_blocks:
            break
        
        # Score each candidate by how many uncovered triples it covers
        best_block = None
        best_score = -1
        for block in candidate_blocks:
            tris = block_to_triples[tuple(block)]
            score = np.sum(uncovered[tris])
            if score > best_score:
                best_score = score
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Mark covered
        tris = block_to_triples[tuple(best_block)]
        uncovered[tris] = False
        
    return blocks

def _remove_redundant(blocks, v, t, deadline):
    """Remove redundant blocks using incremental coverage counting"""
    if not blocks:
        return blocks
    
    n_triples = len(idx_to_triple)
    cover_count = np.zeros(n_triples, dtype=np.int32)
    block_triples = []
    
    for b in blocks:
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            cover_count[idx] += 1
            tris.append(idx)
        block_triples.append(tris)
    
    active = [True] * len(blocks)
    changed = True
    
    while changed:
        changed = False
        if time.time() > deadline:
            break
        
        for i in range(len(blocks)):
            if not active[i]:
                continue
            
            # Check if block i is redundant: all its triples have cover_count > 1
            tris = block_triples[i]
            if np.all(cover_count[tris] > 1):
                active[i] = False
                changed = True
                cover_count[tris] -= 1
                    
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _ilp_solve(deadline, upper_bound=None):
    """Solve the set cover ILP using scipy's milp solver"""
    start_time = time.time()
    n_triples = len(idx_to_triple)
    n_blocks = len(all_blocks_list)
    
    # Objective: minimize sum of x_i
    c = np.ones(n_blocks)
    
    # Constraints: A @ x >= 1 for each triple
    A = incidence.toarray()  # n_triples x n_blocks
    lb = np.ones(n_triples)
    ub = np.full(n_triples, np.inf)
    
    # Variable bounds: 0 <= x_i <= 1
    var_lb = np.zeros(n_blocks)
    var_ub = np.ones(n_blocks)
    integrality = np.ones(n_blocks)  # all binary
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(var_lb, var_ub)
    
    # Add time limit in seconds
    time_limit = max(1, int(deadline - time.time() - 5))
    
    try:
        result = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': time_limit, 'mip_rel_gap': 0.01}
        )
        
        if result.status == 0:  # Optimal or near-optimal found
            selected = np.where(result.x > 0.5)[0]
            if upper_bound is not None and len(selected) > upper_bound:
                return None
            return [list(all_blocks_list[i]) for i in selected]
        elif result.status in [1, 2]:  # Locally optimal or infeasible
            # Return best feasible solution if available
            if hasattr(result, 'x') and result.x is not None:
                selected = np.where(result.x > 0.5)[0]
                if len(selected) > 0:
                    if upper_bound is not None and len(selected) > upper_bound:
                        return None
                    return [list(all_blocks_list[i]) for i in selected]
            return None
        else:
            return None
    except Exception as e:
        return None

def _local_search(blocks, v, t, deadline, max_iter=5000):
    """Local search: try removing blocks and replacing with others"""
    if not blocks:
        return blocks
    
    n_triples = len(idx_to_triple)
    cover_count = np.zeros(n_triples, dtype=np.int32)
    block_triples = []
    
    for b in blocks:
        tris = block_to_triples[tuple(b)]
        cover_count[tris] += 1
        block_triples.append(tris)
    
    best_blocks = blocks
    best_count = len(blocks)
    iterations = 0
    
    while iterations < max_iter and time.time() < deadline:
        iterations += 1
        improved = False
        
        # Try removing each block if possible
        for i in range(len(best_blocks)):
            tris_i = block_triples[i]
            if np.all(cover_count[tris_i] > 1):
                # Remove block i
                best_blocks.pop(i)
                block_triples.pop(i)
                cover_count[tris_i] -= 1
                best_count -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing a block with a better one
        for i in range(len(best_blocks)):
            tris_i = block_triples[i]
            # Count how many triples would become uncovered if we remove block i
            would_uncover = np.sum(cover_count[tris_i] == 1)
            
            if would_uncover == 0:
                # This block is redundant, remove it
                best_blocks.pop(i)
                block_triples.pop(i)
                cover_count[tris_i] -= 1
                best_count -= 1
                improved = True
                break
            elif would_uncover == 1:
                # Only one triple would be uncovered; try to find a new block covering it
                unique_tris = tris_i[cover_count[tris_i] == 1]
                target_triple_idx = unique_tris[0]
                
                existing_blocks = set(tuple(b) for b in best_blocks)
                found_new = False
                for block in all_blocks_list:
                    if tuple(block) in existing_blocks:
                        continue
                    # Check if it covers the target triple
                    if target_triple_idx in block_to_triples[tuple(block)]:
                        new_tris = block_to_triples[tuple(block)]
                        # Remove old block i
                        cover_count[tris_i] -= 1
                        # Add new block
                        cover_count[new_tris] += 1
                        
                        # Check validity: all triples still covered
                        if np.all(cover_count >= 1):
                            best_blocks[i] = list(block)
                            block_triples[i] = new_tris
                            improved = True
                            found_new = True
                            break
                        else:
                            # Revert
                            cover_count[new_tris] -= 1
                            cover_count[tris_i] += 1
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best_blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # Safety margin
    rng = random.Random(42)
    
    # Phase 1: Fast Greedy Construction
    blocks = _greedy(v, k, t, deadline, rng)
    
    # Phase 2: Remove Redundant Blocks
    blocks = _remove_redundant(blocks, v, t, deadline)
    
    best_blocks = blocks
    best_count = len(blocks)
    
    # Phase 3: Local Search
    if time.time() < deadline - 20.0:
        ls_blocks = _local_search(blocks, v, t, deadline - 10.0)
        if len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # Phase 4: More local search on best result
    if time.time() < deadline - 10.0 and best_blocks:
        final_blocks = _local_search(best_blocks, v, t, deadline - 5.0)
        if len(final_blocks) < best_count:
            best_blocks = final_blocks
    
    return best_blocks
