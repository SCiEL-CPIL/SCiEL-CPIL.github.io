import numpy as np
import itertools
import time

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
        for triple in itertools.combinations(block, t):
            block_triples[i, :] = 0 # Reset? No, we need to fill it properly.
            # Actually, we should just fill the array.
            # It's better to create a list first then convert to array for each block?
            # No, pre-filling a 2D array is fine.
            pass
            
    # Re-do precomputation more efficiently
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Greedy Set Cover
    uncovered = np.ones(n_triples, dtype=bool)
    selected_blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Calculate gain for each block: number of currently uncovered triples it covers
        # uncovered is boolean, block_triples are indices
        # We want to count how many of the triples in each block are still uncovered.
        # gains[i] = sum(uncovered[block_triples[i]])
        
        # This is a fancy indexing operation.
        # block_triples is (n_blocks, c_k_t)
        # uncovered is (n_triples,)
        # uncovered[block_triples] will be (n_blocks, c_k_t) boolean array.
        # np.sum(..., axis=1) will be (n_blocks,)
        
        gains = np.sum(uncovered[block_triples], axis=1)
        
        # Find the block with maximum gain
        best_idx = np.argmax(gains)
        
        # If gain is 0, we are stuck (should not happen if a solution exists)
        if gains[best_idx] == 0:
            break
            
        # Add this block to our solution
        selected_blocks.append(blocks[best_idx])
        
        # Update uncovered status
        uncovered[block_triples[best_idx]] = False
        
    # Convert back to list of lists of ints
    result = [list(b) for b in selected_blocks]
    
    # Optional: Local search to improve?
    # For now, return the greedy result.
    # We can add a simple local search: try to replace a block with another that covers the same critical triples?
    # Or just run multiple random greedy passes?
    
    # Let's add a simple local search phase if time permits.
    # Actually, the greedy with NumPy is very fast. We can run multiple instances with random tie-breaking.
    
    best_solution = result
    best_size = len(result)
    
    # Run a few more random greedy passes
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            gains = np.sum(uncovered[block_triples], axis=1)
            
            # Random tie-breaking among max gains
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = np.random.choice(candidates)
            
            selected_blocks.append(blocks[best_idx])
            uncovered[block_triples[best_idx]] = False
            
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = [list(b) for b in selected_blocks]
            
    return best_solution
