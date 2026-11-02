import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute triple indices for each 7-subset on demand, stored as numpy arrays
    block_triple_cache = {}
    
    def get_block_triples_np(block):
        key = tuple(sorted(block))
        if key in block_triple_cache:
            return block_triple_cache[key]
        triples = list(itertools.combinations(key, t))
        arr = np.array([triple_to_idx[triple] for triple in triples], dtype=np.int32)
        block_triple_cache[key] = arr
        return arr
    
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered_set = set(range(num_triples))
        blocks = []
        
        while uncovered_set:
            if time.time() > deadline:
                break
                
            # Pick an uncovered triple randomly
            target_triple_idx = random.choice(list(uncovered_set))
            target_triple = all_triples[target_triple_idx]
            
            remaining_points = [p for p in points if p not in target_triple]
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of remaining points to form the block
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                
                # Calculate gain using vectorized numpy
                b_triples = get_block_triples_np(block)
                gain = np.sum(coverage_count[b_triples] == 0)
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 35:  # Maximum possible gain
                        break
                    
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            # Update coverage
            b_triples = get_block_triples_np(best_block)
            coverage_count[b_triples] += 1
            for idx in b_triples:
                if coverage_count[idx] == 1:
                    uncovered_set.discard(idx)
                    
        return blocks

    def local_search(blocks):
        block_tuples = [tuple(sorted(b)) for b in blocks]
        
        # Build coverage from scratch
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        for b in block_tuples:
            coverage_count[get_block_triples_np(b)] += 1
        
        improved = True
        max_rounds = 50
        round_num = 0
        
        while improved and time.time() < deadline and round_num < max_rounds:
            improved = False
            round_num += 1
            
            # Pass 1: Deletion - remove blocks with no unique coverage
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples_np(b)
                if np.all(coverage_count[b_triples] > 1):
                    coverage_count[b_triples] -= 1
                    del block_tuples[i]
                    improved = True
                    break
            
            if time.time() > deadline or improved:
                continue
            
            # Pass 2: Exchange - try to replace blocks that have unique triples
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples_np(b)
                unique_mask = (coverage_count[b_triples] == 1)
                unique_triples_idx = b_triples[unique_mask]
                
                if len(unique_triples_idx) == 0:
                    continue
                
                # Determine mandatory points from unique triples
                mandatory_points = set()
                for t_idx in unique_triples_idx:
                    mandatory_points.update(all_triples[t_idx])
                
                if len(mandatory_points) > k:
                    continue
                
                available_points = [p for p in points if p not in mandatory_points]
                num_extra_needed = k - len(mandatory_points)
                
                if num_extra_needed > len(available_points):
                    continue
                
                # Find best replacement
                best_new_block = None
                best_gain = 0
                best_redundancy = 0
                
                candidates = list(itertools.combinations(available_points, num_extra_needed))
                random.shuffle(candidates)
                
                for combo in candidates:
                    new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                    if new_block == b:
                        continue
                    
                    new_triples = get_block_triples_np(new_block)
                    
                    # Calculate gain: how many currently-uncovered triples does new block cover?
                    gain = np.sum(coverage_count[new_triples] == 0)
                    redundancy = np.sum(coverage_count[new_triples] == 1)
                    
                    if gain > best_gain or (gain == best_gain and redundancy > best_redundancy):
                        best_gain = gain
                        best_redundancy = redundancy
                        best_new_block = new_block
                        if gain > 0:
                            break
                            
                if best_new_block is not None and best_gain > 0:
                    # Perform swap
                    old_triples = get_block_triples_np(b)
                    coverage_count[old_triples] -= 1
                    
                    new_triples = get_block_triples_np(best_new_block)
                    coverage_count[new_triples] += 1
                    
                    block_tuples[i] = best_new_block
                    improved = True
                    break
        
        return [list(b) for b in block_tuples]

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with adaptive budget
    min_restarts = 4
    max_restarts = 20
    
    for i in range(min_restarts):
        if time.time() > deadline - 5:
            break
            
        blocks = greedy_construct(random_seed=i)
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
                
            blocks = greedy_construct(random_seed=i + 100)
            blocks = local_search(blocks)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
            if best_count <= 70:
                break
    
    # Final polish: run local search again on best solution
    if best_blocks is not None and time.time() < deadline - 15:
        improved_blocks = local_search(best_blocks)
        if len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is None:
        return []
        
    return best_blocks
