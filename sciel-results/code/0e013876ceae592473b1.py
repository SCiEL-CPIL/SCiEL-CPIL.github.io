import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute all k-subsets and their triple indices using vectorization
    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)
    c_k_t = len(list(itertools.combinations(range(k), t)))
    
    # cover_matrix[i] contains the triple indices covered by all_k_subsets[i]
    cover_matrix = np.zeros((num_k_subsets, c_k_t), dtype=np.int32)
    
    # Vectorized precomputation of cover_matrix
    for i, block in enumerate(all_k_subsets):
        block_triples = list(itertools.combinations(block, t))
        cover_matrix[i] = [triple_to_idx[triple] for triple in block_triples]
    
    # Map block tuple to its index for fast lookup
    block_to_idx = {block: i for i, block in enumerate(all_k_subsets)}
    
    def get_block_triples(block):
        idx = block_to_idx[tuple(sorted(block))]
        return cover_matrix[idx]
    
    def fast_greedy_cover(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
                
            # Vectorized gain calculation
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks, coverage_count
    
    def local_search(blocks):
        block_tuples = [tuple(sorted(b)) for b in blocks]
        
        # Build coverage from scratch
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        for b in block_tuples:
            for t_idx in get_block_triples(b):
                coverage_count[t_idx] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Deletion
            to_remove = []
            for i, b in enumerate(block_tuples):
                if time.time() > deadline:
                    break
                b_triples = get_block_triples(b)
                if not np.any(coverage_count[b_triples] == 1):
                    to_remove.append(i)
                    
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(block_tuples) if i not in remove_set]
                coverage_count = np.zeros(num_triples, dtype=np.int32)
                for b in new_blocks:
                    for t_idx in get_block_triples(b):
                        coverage_count[t_idx] += 1
                block_tuples = new_blocks
                improved = True
                continue
            
            # Pass 2: Exchange
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                unique_triples = b_triples[coverage_count[b_triples] == 1]
                
                if len(unique_triples) == 0:
                    continue
                
                # Determine mandatory points for replacement
                mandatory_points = set()
                for t_idx in unique_triples:
                    mandatory_points.update(all_triples[t_idx])
                
                if len(mandatory_points) > k:
                    continue
                
                available_points = [p for p in points if p not in mandatory_points]
                num_extra_needed = k - len(mandatory_points)
                
                if num_extra_needed > len(available_points):
                    continue
                
                best_new_block = None
                best_score = -1
                best_redundancy = -1
                
                candidates = list(itertools.combinations(available_points, num_extra_needed))
                random.shuffle(candidates)
                
                for combo in candidates:
                    new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                    if new_block == b:
                        continue
                    
                    new_triples = get_block_triples(new_block)
                    
                    gain = 0
                    redundancy_gain = 0
                    for t_idx in new_triples:
                        if coverage_count[t_idx] == 0:
                            gain += 1
                        elif coverage_count[t_idx] == 1:
                            redundancy_gain += 1
                        
                    if gain > best_score or (gain == best_score and redundancy_gain > best_redundancy):
                        best_score = gain
                        best_redundancy = redundancy_gain
                        best_new_block = new_block
                        if gain > 0:
                            break
                            
                if best_new_block is not None and best_score > 0:
                    # Perform swap
                    old_triples = get_block_triples(b)
                    for t_idx in old_triples:
                        coverage_count[t_idx] -= 1
                        
                    new_triples = get_block_triples(best_new_block)
                    for t_idx in new_triples:
                        coverage_count[t_idx] += 1
                        
                    block_tuples[i] = best_new_block
                    improved = True
                    break
                    
        # Final polish pass
        if time.time() < deadline - 5:
            improved = True
            max_polish_rounds = 3
            polish_round = 0
            while improved and time.time() < deadline - 2 and polish_round < max_polish_rounds:
                improved = False
                polish_round += 1
                
                # Deletion pass
                to_remove = []
                for i, b in enumerate(block_tuples):
                    if time.time() > deadline - 2:
                        break
                    b_triples = get_block_triples(b)
                    if not np.any(coverage_count[b_triples] == 1):
                        to_remove.append(i)
                        
                if to_remove:
                    remove_set = set(to_remove)
                    new_blocks = [b for i, b in enumerate(block_tuples) if i not in remove_set]
                    coverage_count = np.zeros(num_triples, dtype=np.int32)
                    for b in new_blocks:
                        for t_idx in get_block_triples(b):
                            coverage_count[t_idx] += 1
                    block_tuples = new_blocks
                    improved = True
                    continue
                
                # Exchange pass
                for i in range(len(block_tuples)):
                    if time.time() > deadline - 2:
                        break
                    b = block_tuples[i]
                    b_triples = get_block_triples(b)
                    unique_triples = b_triples[coverage_count[b_triples] == 1]
                    
                    if len(unique_triples) == 0:
                        continue
                    
                    mandatory_points = set()
                    for t_idx in unique_triples:
                        mandatory_points.update(all_triples[t_idx])
                    
                    if len(mandatory_points) > k:
                        continue
                    
                    available_points = [p for p in points if p not in mandatory_points]
                    num_extra_needed = k - len(mandatory_points)
                    
                    if num_extra_needed > len(available_points):
                        continue
                    
                    best_new_block = None
                    best_score = -1
                    best_redundancy = -1
                    
                    candidates = list(itertools.combinations(available_points, num_extra_needed))
                    random.shuffle(candidates)
                    
                    for combo in candidates:
                        new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                        if new_block == b:
                            continue
                        
                        new_triples = get_block_triples(new_block)
                        gain = 0
                        redundancy_gain = 0
                        for t_idx in new_triples:
                            if coverage_count[t_idx] == 0:
                                gain += 1
                            elif coverage_count[t_idx] == 1:
                                redundancy_gain += 1
                            
                        if gain > best_score or (gain == best_score and redundancy_gain > best_redundancy):
                            best_score = gain
                            best_redundancy = redundancy_gain
                            best_new_block = new_block
                            if gain > 0:
                                break
                            
                    if best_new_block is not None and best_score > 0:
                        old_triples = get_block_triples(b)
                        for t_idx in old_triples:
                            coverage_count[t_idx] -= 1
                        
                        new_triples = get_block_triples(best_new_block)
                        for t_idx in new_triples:
                            coverage_count[t_idx] += 1
                        
                        block_tuples[i] = best_new_block
                        improved = True
                        break
        
        return [list(b) for b in block_tuples]
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with adaptive budget
    min_restarts = 3
    max_restarts = 15
    
    # Start with a few quick restarts
    for i in range(min_restarts):
        if time.time() > deadline:
            break
            
        blocks, _ = fast_greedy_cover(random_seed=i)
        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 70:
            break
    
    # If we have time, do more restarts
    if best_blocks is not None and time.time() < deadline - 10:
        for i in range(min_restarts, max_restarts):
            if time.time() > deadline - 5:
                break
                
            blocks, _ = fast_greedy_cover(random_seed=i + 100)
            blocks = local_search(blocks)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
            if best_count <= 70:
                break
    
    # Final: if we have significant time left, do another round of local search on best
    if best_blocks is not None and time.time() < deadline - 15:
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is None:
        return []
        
    return best_blocks
