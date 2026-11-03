import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute for each possible block (k-subset) the list of t-subset indices it covers
    # There are C(19,8) = 75582 possible blocks.
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # Block coverage: block_idx -> list of t_idx
    block_t_idx = [None] * num_blocks
    for bi, block in enumerate(all_blocks):
        block_t_idx[bi] = [t_to_idx[ts] for ts in itertools.combinations(block, t)]

    # For each t-subset, list of blocks that cover it
    # t_idx -> list of block_idx
    t_to_blocks = [[] for _ in range(num_t)]
    for bi in range(num_blocks):
        for ti in block_t_idx[bi]:
            t_to_blocks[ti].append(bi)

    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = np.ones(num_t, dtype=bool)
        uncovered_count = num_t
        solution = []
        
        while uncovered_count > 0:
            if elapsed() > deadline - 5:
                break
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_t = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            # Find best block covering target_t
            candidates = t_to_blocks[target_t]
            if not candidates:
                break
                
            best_block = -1
            best_gain = 0
            
            # Sample candidates if too many, otherwise check all
            if len(candidates) > 200:
                sampled = random.sample(candidates, 200)
            else:
                sampled = candidates
                
            for bi in sampled:
                gain = 0
                for ti in block_t_idx[bi]:
                    if uncovered[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block == -1 or best_gain == 0:
                break
                
            solution.append(all_blocks[best_block])
            
            # Mark covered
            for ti in block_t_idx[best_block]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    uncovered_count -= 1
                    
        return solution

    def local_search(initial_solution):
        solution = list(initial_solution)
        if not solution:
            return []
            
        # Build coverage count
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in solution:
            bi = block_to_idx[tuple(b)]
            for ti in block_t_idx[bi]:
                cover_count[ti] += 1
                
        improved = True
        local_start = time.time()
        
        while improved and (time.time() - local_start) < 15.0:
            improved = False
            
            # Phase 1: Drop redundant blocks
            i = 0
            while i < len(solution):
                b = solution[i]
                bi = block_to_idx[tuple(b)]
                redundant = True
                for ti in block_t_idx[bi]:
                    if cover_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_t_idx[bi]:
                        cover_count[ti] -= 1
                    solution.pop(i)
                    improved = True
                else:
                    i += 1
            if (time.time() - local_start) > 15.0:
                break
                
            # Phase 2: Swap blocks
            random.shuffle(solution)
            for i, b_out in enumerate(solution):
                if (time.time() - local_start) > 15.0:
                    break
                    
                bi_out = block_to_idx[tuple(b_out)]
                # Find t-subsets uniquely covered by b_out
                unique_tis = [ti for ti in block_t_idx[bi_out] if cover_count[ti] == 1]
                if not unique_tis:
                    # Redundant, drop it
                    for ti in block_t_idx[bi_out]:
                        cover_count[ti] -= 1
                    solution.pop(i)
                    improved = True
                    break
                    
                # Try to find a replacement block that covers all unique_tis
                # Use first unique ti to generate candidates
                u_ti = unique_tis[0]
                candidates = t_to_blocks[u_ti]
                
                best_cand = -1
                best_score = -1
                
                # Filter candidates that cover all unique_tis
                valid_candidates = []
                for bi in candidates:
                    if bi == bi_out:
                        continue
                    # Check if covers all unique_tis
                    block_set = set(block_t_idx[bi])
                    covers_all = all(u in block_set for u in unique_tis)
                    if covers_all:
                        valid_candidates.append(bi)
                        
                if not valid_candidates:
                    continue
                    
                # Score: maximize overlap with already-covered t-subsets
                for bi in valid_candidates:
                    score = 0
                    for ti in block_t_idx[bi]:
                        if cover_count[ti] >= 2:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = bi
                        
                if best_cand != -1:
                    # Perform swap
                    for ti in block_t_idx[bi_out]:
                        cover_count[ti] -= 1
                    for ti in block_t_idx[best_cand]:
                        cover_count[ti] += 1
                    solution[i] = all_blocks[best_cand]
                    improved = True
                    break
                    
        return solution

    best_solution = None
    best_size = float('inf')
    
    num_restarts = 30
    for i in range(num_restarts):
        if elapsed() > deadline - 20:
            break
            
        sol = run_greedy(i)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
        if elapsed() > deadline - 10:
            break
            
        improved_sol = local_search(sol)
        if len(improved_sol) < best_size:
            best_size = len(improved_sol)
            best_solution = improved_sol
            
    if not best_solution:
        best_solution = run_greedy(0)
        
    return [list(b) for b in best_solution]
