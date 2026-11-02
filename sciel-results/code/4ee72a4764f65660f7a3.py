import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Number of t-subsets in a k-subset
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # Precompute block_triples: for each block, the indices of triples it covers
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    def greedy_with_randomization(rng, deadline):
        """Greedy set cover with random tie-breaking."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate gain for each block
            gains = np.sum(uncovered[block_triples], axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Random tie-breaking among max gains
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(blocks[best_idx])
            uncovered[block_triples[best_idx]] = False
            
        return [list(b) for b in selected_blocks]
    
    def greedy_with_weighted_heuristic(rng, deadline):
        """
        Greedy with a weighted heuristic that considers both immediate gain
        and a secondary criterion based on point frequencies in uncovered triples.
        """
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        # Track point frequencies in uncovered triples
        # For each point, count how many uncovered triples contain it
        point_freq = np.zeros(v, dtype=np.int32)
        for i, triple in enumerate(triples):
            if uncovered[i]:
                for p in triple:
                    point_freq[p] += 1
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate primary gain
            gains = np.sum(uncovered[block_triples], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                # Secondary heuristic: prefer blocks whose points have higher average frequency
                # This is a soft preference to break ties more intelligently
                secondary_scores = []
                for idx in candidates:
                    block = blocks[idx]
                    # Average frequency of points in this block
                    avg_freq = np.mean(point_freq[list(block)])
                    secondary_scores.append(avg_freq)
                
                secondary_scores = np.array(secondary_scores)
                max_sec = np.max(secondary_scores)
                sec_candidates = candidates[secondary_scores == max_sec]
                
                if len(sec_candidates) == 1:
                    best_idx = sec_candidates[0]
                else:
                    best_idx = rng.choice(sec_candidates)
            
            selected_blocks.append(blocks[best_idx])
            
            # Update uncovered and point frequencies
            block = blocks[best_idx]
            covered_triples = block_triples[best_idx]
            for tri_idx in covered_triples:
                if uncovered[tri_idx]:
                    uncovered[tri_idx] = False
                    for p in triples[tri_idx]:
                        point_freq[p] -= 1
                        
        return [list(b) for b in selected_blocks]
    
    def local_search_improve(blocks_list, deadline):
        """
        Local search: try to remove redundant blocks or swap blocks to reduce count.
        """
        if time.time() > deadline:
            return blocks_list
            
        n = len(blocks_list)
        if n == 0:
            return blocks_list
            
        # Compute coverage counts for each triple
        triple_counts = np.zeros(n_triples, dtype=np.int32)
        for block in blocks_list:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                triple_counts[idx] += 1
                
        improved = True
        current_blocks = list(blocks_list)
        
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Try removing each block
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                    
                block = current_blocks[i]
                # Check if all triples in this block are covered by other blocks
                can_remove = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if triple_counts[idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block
                    current_blocks.pop(i)
                    # Update counts
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        triple_counts[idx] -= 1
                    improved = True
                    break  # Restart scanning from beginning
                    
        return current_blocks
    
    def swap_local_search(blocks_list, deadline):
        """
        Try to swap a block with another that covers the same critical triples
        but allows further reduction.
        """
        if time.time() > deadline:
            return blocks_list
            
        current_blocks = list(blocks_list)
        n = len(current_blocks)
        if n < 2:
            return current_blocks
            
        # Compute coverage counts
        triple_counts = np.zeros(n_triples, dtype=np.int32)
        for block in current_blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                triple_counts[idx] += 1
                
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                    
                block_i = current_blocks[i]
                # Find triples that are only covered by block_i (critical triples)
                critical_triples = []
                for triple in itertools.combinations(block_i, t):
                    idx = triple_to_idx[triple]
                    if triple_counts[idx] == 1:
                        critical_triples.append(idx)
                        
                if not critical_triples:
                    continue
                    
                # Try swapping with another block
                for j in range(len(current_blocks)):
                    if j == i or time.time() > deadline:
                        continue
                        
                    block_j = current_blocks[j]
                    # Check if block_j covers all critical triples of block_i
                    block_j_set = set(block_j)
                    covers_all = True
                    for tri_idx in critical_triples:
                        tri = triples[tri_idx]
                        if not all(p in block_j_set for p in tri):
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Check if removing block_i and adding block_j is valid
                        # and if block_j has fewer unique critical triples
                        # This is complex; for simplicity, just try the swap and verify
                        new_blocks = current_blocks[:i] + current_blocks[i+1:] + [block_j]
                        
                        # Verify validity
                        new_counts = np.zeros(n_triples, dtype=np.int32)
                        for block in new_blocks:
                            for triple in itertools.combinations(block, t):
                                idx = triple_to_idx[triple]
                                new_counts[idx] += 1
                                
                        if np.all(new_counts >= 1):
                            # Check if this leads to a smaller or equal set
                            # For now, accept if same size (may enable further reductions)
                            if len(new_blocks) <= len(current_blocks):
                                current_blocks = new_blocks
                                # Recompute counts
                                triple_counts = new_counts
                                improved = True
                                break
                                
                if improved:
                    break
                    
        return current_blocks
    
    # Run multiple greedy strategies and local search
    rng = random.Random(42)
    best_solution = None
    best_size = float('inf')
    
    # Deadline for individual greedy runs
    greedy_deadline = deadline - 20.0
    
    # Run standard greedy with random restarts
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > greedy_deadline:
            break
        rng_iter = random.Random(42 + i)
        solution = greedy_with_randomization(rng_iter, time.time() + 5.0)
        if solution and len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
            
    # Run weighted heuristic greedy
    for i in range(5):
        if time.time() > greedy_deadline:
            break
        rng_iter = random.Random(1000 + i)
        solution = greedy_with_weighted_heuristic(rng_iter, time.time() + 5.0)
        if solution and len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
            
    # Apply local search to the best solution
    if best_solution is not None:
        improved = local_search_improve(best_solution, deadline - 5.0)
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = improved
            
        # Try swap local search
        swapped = swap_local_search(best_solution, deadline - 2.0)
        if len(swapped) < best_size:
            best_size = len(swapped)
            best_solution = swapped
            
    # If we still have time, try more diverse randomizations
    if best_solution is not None and time.time() < deadline - 10.0:
        # Try greedy with different random seeds and more aggressive tie-breaking
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            rng_iter = random.Random(5000 + i)
            solution = greedy_with_randomization(rng_iter, time.time() + 3.0)
            if solution:
                # Apply local search
                improved = local_search_improve(solution, deadline - 2.0)
                if len(improved) < best_size:
                    best_size = len(improved)
                    best_solution = improved
                    
    if best_solution is None:
        # Fallback: return a valid covering (may not be optimal)
        # Use a simple greedy without time limit
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        while np.any(uncovered):
            gains = np.sum(uncovered[block_triples], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_blocks.append(blocks[best_idx])
            uncovered[block_triples[best_idx]] = False
        best_solution = [list(b) for b in selected_blocks]
        
    return best_solution
