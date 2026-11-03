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
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Greedy Set Cover with random tie-breaking
    def greedy_pass(randomize=False, rng=None):
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                return None
                
            # Calculate gain for each block: number of currently uncovered triples it covers
            gains = np.sum(uncovered[block_triples], axis=1)
            
            # Find the block with maximum gain
            if randomize and rng is not None:
                max_gain = np.max(gains)
                if max_gain == 0:
                    return selected_blocks
                candidates = np.where(gains == max_gain)[0]
                best_idx = rng.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    return selected_blocks
                    
            # Add this block to our solution
            selected_blocks.append(blocks[best_idx])
            
            # Update uncovered status
            uncovered[block_triples[best_idx]] = False
            
        return selected_blocks

    # Run initial greedy pass
    best_solution = greedy_pass(randomize=False)
    best_size = len(best_solution)
    
    # Run multiple random greedy passes
    rng = random.Random(42)
    num_restarts = 20
    for _ in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        result = greedy_pass(randomize=True, rng=rng)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_solution = result
            
    # Local search: try to remove redundant blocks
    def local_search(blocks_list, time_limit):
        # Convert to set of frozensets for faster lookup
        block_set = set(frozenset(b) for b in blocks_list)
        
        # Count how many times each triple is covered
        triple_counts = np.zeros(n_triples, dtype=np.int32)
        for b in blocks_list:
            for triple in itertools.combinations(b, t):
                triple_counts[triple_to_idx[triple]] += 1
                
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each block
            for i in range(len(blocks_list)):
                if time.time() > time_limit:
                    break
                    
                b = blocks_list[i]
                # Check if removing this block leaves any triple uncovered
                can_remove = True
                for triple in itertools.combinations(b, t):
                    idx = triple_to_idx[triple]
                    if triple_counts[idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove the block
                    blocks_list.pop(i)
                    for triple in itertools.combinations(b, t):
                        idx = triple_to_idx[triple]
                        triple_counts[idx] -= 1
                    improved = True
                    break
                    
        return blocks_list

    # Apply local search to the best solution
    if time.time() < deadline - 5.0:
        best_solution = local_search(best_solution, deadline - 5.0)
        best_size = len(best_solution)
        
    # Convert back to list of lists of ints
    result = [list(b) for b in best_solution]
    
    return result
