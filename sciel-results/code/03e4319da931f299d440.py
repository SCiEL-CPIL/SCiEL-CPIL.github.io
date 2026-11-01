import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    def get_block_triples(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    best_blocks = None
    best_count = float('inf')
    
    # Run greedy a few times with randomization
    num_greedy_runs = 5
    for run in range(num_greedy_runs):
        if time.time() > deadline - 10: # Leave time for optimization
            break
            
        random.seed(run) # Different seed for each run
        
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered = set(range(num_triples))
        blocks = []
        
        # Greedy loop
        while uncovered:
            if time.time() > deadline - 10:
                break
                
            # Pick a random uncovered triple
            target_triple_idx = random.choice(list(uncovered))
            target_triple = all_triples[target_triple_idx]
            
            # Find best block containing target_triple
            remaining_points = [p for p in points if p not in target_triple]
            best_block = None
            best_gain = -1
            
            # Iterate through combinations
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                gain = 0
                for b_triple in itertools.combinations(block, t):
                    if triple_to_idx[b_triple] in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            # Update coverage
            b_triples = get_block_triples(best_block)
            for idx in b_triples:
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered.discard(idx)
                    
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks] # Deep copy

    if best_blocks is None:
        # Fallback: return empty or a trivial valid set if something went wrong
        # But greedy should always produce a valid set.
        return []

    # Optimization Phase on best_blocks
    block_tuples = [tuple(b) for b in best_blocks]
    block_triples_map = {b: get_block_triples(b) for b in block_tuples}
    
    # Recalculate coverage_count for the best_blocks
    coverage_count = np.zeros(num_triples, dtype=np.int32)
    for b in block_tuples:
        for t_idx in block_triples_map[b]:
            coverage_count[t_idx] += 1

    def can_delete(block_idx):
        b = block_tuples[block_idx]
        for t_idx in block_triples_map[b]:
            if coverage_count[t_idx] == 1:
                return False
        return True

    def get_unique_triples(block):
        return [t_idx for t_idx in block_triples_map[block] if coverage_count[t_idx] == 1]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Deletion
        for i in range(len(block_tuples)):
            if time.time() > deadline:
                break
            if can_delete(i):
                b = block_tuples[i]
                for t_idx in block_triples_map[b]:
                    coverage_count[t_idx] -= 1
                del block_tuples[i]
                del block_triples_map[b]
                improved = True
                break
        
        if time.time() > deadline:
            break
            
        if improved:
            continue
            
        # Pass 2: Exchange
        for i in range(len(block_tuples)):
            if time.time() > deadline:
                break
                
            b = block_tuples[i]
            unique_triples = get_unique_triples(b)
            
            if not unique_triples:
                continue
                
            mandatory_points = set()
            for t_idx in unique_triples:
                mandatory_points.update(all_triples[t_idx])
                
            if len(mandatory_points) > k:
                continue
                
            available_points = [p for p in points if p not in mandatory_points]
            
            num_attempts = 10
            found_replacement = False
            for _ in range(num_attempts):
                if time.time() > deadline:
                    break
                    
                if len(available_points) < k - len(mandatory_points):
                    break
                extra_points = random.sample(available_points, k - len(mandatory_points))
                new_block_tuple = tuple(sorted(mandatory_points + extra_points))
                
                if new_block_tuple == b or new_block_tuple in block_tuples:
                    continue
                    
                new_block_triples = get_block_triples(new_block_tuple)
                
                # Perform swap
                for t_idx in block_triples_map[b]:
                    coverage_count[t_idx] -= 1
                del block_triples_map[b]
                
                block_tuples[i] = new_block_tuple
                block_triples_map[new_block_tuple] = new_block_triples
                for t_idx in new_block_triples:
                    coverage_count[t_idx] += 1
                    
                improved = True
                found_replacement = True
                break
                
            if found_replacement:
                break

    return [list(b) for b in block_tuples]
