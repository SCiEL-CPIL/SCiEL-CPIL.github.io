import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] contains the indices of the t-subsets covered by blocks[i]
    # Shape: (n_blocks, c_k_t)
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # Also compute frequency of each triple to use for weighting
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1
            
    # Weights: 1 / frequency. Higher weight for rarer triples.
    weights = 1.0 / np.maximum(triple_freq, 1)
    
    best_solution = None
    best_size = float('inf')
    
    # Local search polish: try to replace a block with another that covers same or more uncovered triples
    def local_search_polish(selected_indices, block_triples, n_triples, c_k_t):
        """Try to reduce solution size by replacing blocks."""
        if len(selected_indices) <= 1:
            return selected_indices
        
        current = list(selected_indices)
        improved = True
        max_iterations = 50
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Compute cover counts
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current:
                cover_count[block_triples[b_idx]] += 1
            
            # Try removing each block if redundant
            for i in range(len(current)):
                b_idx = current[i]
                b_triples = block_triples[b_idx]
                if np.min(cover_count[b_triples]) > 1:
                    # Remove this block
                    cover_count[b_triples] -= 1
                    current.pop(i)
                    improved = True
                    break
        
        return current
    
    # Weighted greedy with randomization
    def weighted_greedy_cover(weights, block_triples, n_triples, temperature=0.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            weighted_uncovered = np.where(uncovered, weights, 0.0)
            gains = np.sum(weighted_uncovered[block_triples], axis=1)
            
            if temperature > 0:
                # Softmax sampling for diversity
                max_gain = np.max(gains)
                if max_gain > 0:
                    exp_gains = np.exp((gains - max_gain) / temperature)
                    probs = exp_gains / np.sum(exp_gains)
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    best_idx = np.argmax(gains)
            else:
                best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    # Run multiple greedy passes with different weight perturbations
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        
        # Perturb weights slightly for diversity
        if n_restarts == 1:
            perturbed_weights = weights.copy()
        else:
            noise = np.random.normal(0, 0.1, size=n_triples)
            perturbed_weights = weights * (1 + noise)
            perturbed_weights = np.maximum(perturbed_weights, 0.01)
        
        # Run greedy
        selected = weighted_greedy_cover(perturbed_weights, block_triples, n_triples)
        
        # Polish
        selected = local_search_polish(selected, block_triples, n_triples, c_k_t)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
        
        # Early exit if we hit the lower bound (69)
        if best_size <= 69:
            break
    
    if best_solution is None:
        # Fallback: return a valid covering
        selected = weighted_greedy_cover(weights, block_triples, n_triples)
        selected = local_search_polish(selected, block_triples, n_triples, c_k_t)
        best_solution = [list(blocks[i]) for i in selected]
        
    return best_solution
