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
    
    # Precompute for each point, which triples contain it (for fast mandatory point logic)
    point_in_triples = defaultdict(list)
    for i, triple in enumerate(all_triples):
        for p in triple:
            point_in_triples[p].append(i)
    
    # Cache for block triple indices
    block_triple_cache = {}
    
    def get_block_triples(block):
        key = tuple(sorted(block))
        if key in block_triple_cache:
            return block_triple_cache[key]
        result = [triple_to_idx[triple] for triple in itertools.combinations(key, t)]
        block_triple_cache[key] = result
        return result
    
    def get_block_triples_array(block):
        """Return numpy array of triple indices for a block."""
        tris = get_block_triples(block)
        return np.array(tris, dtype=np.int32)
    
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick an uncovered triple (randomly to allow different paths)
            target_triple_idx = random.choice(list(uncovered))
            target_triple = all_triples[target_triple_idx]
            
            remaining_points = [p for p in points if p not in target_triple]
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of remaining points to form the block
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                
                # Calculate gain using cached triple indices
                b_triples = get_block_triples(block)
                gain = sum(1 for idx in b_triples if idx in uncovered)
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 35:  # Maximum possible gain, can't do better
                        break
                    
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            # Update coverage
            b_triples = get_block_triples(best_block)
            for idx in b_triples:
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks, coverage_count

    def local_search(blocks, coverage_count=None):
        block_tuples = [tuple(sorted(b)) for b in blocks]
        # Rebuild coverage from scratch to be safe and consistent
        coverage_count = np.zeros(num_triples, dtype=np.int32)
        for b in block_tuples:
            for t_idx in get_block_triples(b):
                coverage_count[t_idx] += 1
                
        # Precompute reciprocal coverage for faster density scoring
        inv_cov = 1.0 / (coverage_count + 1)  # 1/(cov+1) for added triples
        inv_cov_current = 1.0 / np.maximum(coverage_count, 1)  # 1/cov for removed triples
        
        def update_inv_cov():
            nonlocal inv_cov, inv_cov_current
            inv_cov = 1.0 / (coverage_count + 1)
            inv_cov_current = 1.0 / np.maximum(coverage_count, 1)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Deletion - remove blocks where all triples are covered >= 2 times
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                if all(coverage_count[t_idx] >= 2 for t_idx in b_triples):
                    for t_idx in b_triples:
                        coverage_count[t_idx] -= 1
                    del block_tuples[i]
                    update_inv_cov()
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            if improved:
                continue
            
            # Pass 2: Exchange using density-based scoring
            for i in range(len(block_tuples)):
                if time.time() > deadline:
                    break
                b = block_tuples[i]
                b_triples = get_block_triples(b)
                b_set = set(b_triples)
                unique_triples = [t_idx for t_idx in b_triples if coverage_count[t_idx] == 1]
                
                if not unique_triples:
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
                best_density_score = 0.0  # Only accept if strictly positive
                
                # Sample candidates if search space is too large
                candidates = list(itertools.combinations(available_points, num_extra_needed))
                if len(candidates) > 3000:
                    random.shuffle(candidates)
                    candidates = candidates[:3000]
                
                for combo in candidates:
                    new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                    if new_block == b:
                        continue
                    
                    new_triples = get_block_triples(new_block)
                    new_set = set(new_triples)
                    
                    # Check that all unique triples are still covered
                    if not all(t_idx in new_set for t_idx in unique_triples):
                        continue
                    
                    removed = b_set - new_set
                    added = new_set - b_set
                    
                    # Fast density calculation using precomputed arrays
                    density_delta = 0.0
                    for t_idx in removed:
                        density_delta -= inv_cov_current[t_idx]
                    for t_idx in added:
                        density_delta += inv_cov[t_idx]
                    
                    if density_delta > best_density_score:
                        best_density_score = density_delta
                        best_new_block = new_block
                
                if best_new_block is not None:
                    # Perform swap
                    for t_idx in b_triples:
                        coverage_count[t_idx] -= 1
                    for t_idx in get_block_triples(best_new_block):
                        coverage_count[t_idx] += 1
                    block_tuples[i] = best_new_block
                    update_inv_cov()
                    improved = True
                    break
                    
        # Final polish: more aggressive multi-round optimization
        if time.time() < deadline - 5:
            for polish_round in range(5):
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # Deletion pass
                for i in range(len(block_tuples)):
                    if time.time() > deadline - 2:
                        break
                    b = block_tuples[i]
                    b_triples = get_block_triples(b)
                    if all(coverage_count[t_idx] >= 2 for t_idx in b_triples):
                        for t_idx in b_triples:
                            coverage_count[t_idx] -= 1
                        del block_tuples[i]
                        update_inv_cov()
                        improved = True
                        break
                
                if not improved:
                    # Exchange pass
                    for i in range(len(block_tuples)):
                        if time.time() > deadline - 2:
                            break
                        b = block_tuples[i]
                        b_triples = get_block_triples(b)
                        b_set = set(b_triples)
                        unique_triples = [t_idx for t_idx in b_triples if coverage_count[t_idx] == 1]
                        
                        if not unique_triples:
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
                        best_density_score = 0.0
                        
                        candidates = list(itertools.combinations(available_points, num_extra_needed))
                        if len(candidates) > 3000:
                            random.shuffle(candidates)
                            candidates = candidates[:3000]
                        
                        for combo in candidates:
                            new_block = tuple(sorted(list(mandatory_points) + list(combo)))
                            if new_block == b:
                                continue
                            
                            new_triples = get_block_triples(new_block)
                            new_set = set(new_triples)
                            
                            if not all(t_idx in new_set for t_idx in unique_triples):
                                continue
                            
                            removed = b_set - new_set
                            added = new_set - b_set
                            
                            density_delta = 0.0
                            for t_idx in removed:
                                density_delta -= inv_cov_current[t_idx]
                            for t_idx in added:
                                density_delta += inv_cov[t_idx]
                            
                            if density_delta > best_density_score:
                                best_density_score = density_delta
                                best_new_block = new_block
                        
                        if best_new_block is not None:
                            for t_idx in b_triples:
                                coverage_count[t_idx] -= 1
                            for t_idx in get_block_triples(best_new_block):
                                coverage_count[t_idx] += 1
                            block_tuples[i] = best_new_block
                            update_inv_cov()
                            improved = True
                            break
        
        return [list(b) for b in block_tuples]

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with adaptive budget
    min_restarts = 3
    max_restarts = 20
    
    for i in range(min_restarts):
        if time.time() > deadline:
            break
            
        blocks, _ = greedy_construct(random_seed=i)
        blocks = local_search(blocks, None)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 70:
            break
    
    # If we have time, do more restarts with different seeds
    if best_blocks is not None and time.time() < deadline - 10:
        for i in range(min_restarts, max_restarts):
            if time.time() > deadline - 5:
                break
                
            blocks, _ = greedy_construct(random_seed=i + 100)
            blocks = local_search(blocks, None)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                
            if best_count <= 70:
                break
    
    # Final: if we have significant time left, do another round of local search on best
    if best_blocks is not None and time.time() < deadline - 15:
        improved_blocks = local_search(best_blocks, None)
        if len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is None:
        return []
        
    return best_blocks
