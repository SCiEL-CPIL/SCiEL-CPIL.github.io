import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # For v=24, k=7, t=3
    # Total triples: C(24,3) = 2024
    # Total blocks: C(24,7) = 346104
    
    # Precompute data structures
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # block_triples[i] is a list of triple indices covered by block i
    # triple_to_blocks[j] is a list of block indices covering triple j
    # To save memory and time, we use arrays.
    # block_triples is a list of lists of ints.
    # triple_to_blocks is a list of lists of ints.
    
    # Optimization: Instead of storing full lists for triple_to_blocks which is huge (12M entries),
    # we can compute on the fly or use a different structure.
    # However, for WalkSAT, we need to quickly evaluate clause satisfaction.
    
    # Let's stick to the explicit structure but optimize the loop.
    # block_triples: list of lists
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # triple_to_blocks: list of lists
    # This is memory heavy (~100MB+) but necessary for fast updates.
    # Let's try to be efficient.
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)

    # WalkSAT / Local Search
    # Variables: x[i] = 1 if block i is selected.
    # Objective: Minimize sum(x[i]) + Penalty * sum(1 if triple j is uncovered)
    # Actually, we want to find a set of blocks that covers all triples.
    # If we just minimize sum(x[i]), we get 0 blocks.
    # So we must enforce coverage.
    # Standard approach for Covering Design via MaxSAT:
    # Maximize (Number of Covered Triples) - Lambda * (Number of Blocks)
    # Or: Minimize (Number of Blocks) subject to (Number of Uncovered Triples == 0).
    # Since finding a valid solution is the primary goal, and then minimizing size,
    # we can use a penalty method.
    
    # Let's use a simple local search on the set of selected blocks.
    # State: A set of selected blocks (or boolean array).
    # Move: Add a block, Remove a block, or Swap a block.
    # Cost: Number of blocks + M * Number of uncovered triples.
    # We start with a greedy solution and then try to reduce the number of blocks
    # while maintaining coverage, or find a better solution from scratch.
    
    # Better approach for this specific task:
    # 1. Get a valid solution using Greedy (fast).
    # 2. Try to minimize the size using local search (remove blocks if possible, swap to reduce count).
    #    But removing a block might uncover triples.
    #    We can use a "Minimizing Covering" local search.
    
    # Let's implement a local search that starts from a random valid cover (or greedy)
    # and tries to reduce the number of blocks.
    
    # Step 1: Generate a valid solution using Greedy (as before, but faster implementation)
    def greedy_construct():
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        blocks_used = []
        
        while uncovered:
            # Find max gain
            max_gain = -1
            best_block = -1
            # Optimization: Use numpy for max? 
            # gains is a list of ints. 
            # We can just loop.
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_block = i
            
            if max_gain == 0:
                break # Should not happen
                
            blocks_used.append(best_block)
            
            # Update gains
            for t_idx in block_triples[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        
        return blocks_used

    # Step 2: Local Search to minimize size
    # State: set of block indices.
    # We want to minimize len(state) such that all triples are covered.
    
    def minimize_cover(initial_blocks):
        # Convert to set for fast lookup
        current_blocks = set(initial_blocks)
        
        # Precompute for each block, the triples it covers.
        # We need to know which triples are *only* covered by this block (critical triples).
        # If a block has no critical triples, we can remove it.
        
        # To check if a block can be removed:
        # For each triple T in block B:
        #   Count how many blocks in current_blocks cover T.
        #   If count == 1 for any T, then B is essential.
        #   If count > 1 for all T, then B is redundant.
        
        # We maintain a count for each triple: cover_count[t]
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b in current_blocks:
            for t_idx in block_triples[b]:
                cover_count[t_idx] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove each block
            # Random order to avoid bias
            blocks_list = list(current_blocks)
            random.shuffle(blocks_list)
            
            for b in blocks_list:
                if b not in current_blocks:
                    continue
                # Check if b is redundant
                redundant = True
                for t_idx in block_triples[b]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove b
                    current_blocks.remove(b)
                    for t_idx in block_triples[b]:
                        cover_count[t_idx] -= 1
                    improved = True
        
        # Now try swaps? 
        # Swap one block for another? 
        # Usually, removing is the first step.
        # Swapping is harder. We can try:
        # Pick a block B in current.
        # Pick a block C not in current.
        # If removing B and adding C maintains coverage and reduces size?
        # No, size is same.
        # We want to reduce size. So only removal helps.
        # Unless we swap 2 blocks for 1? 
        # That's a 2-opt move.
        
        # Let's stick to removal for now. It's safe.
        
        return list(current_blocks)

    # Run Greedy multiple times to get different starting points, then minimize.
    start_time = time.time()
    max_time = 110
    
    best_solution = None
    best_size = float('inf')
    
    # Run greedy 10 times
    for i in range(10):
        random.seed(i)
        # Add randomness to greedy
        # Modify greedy to use random tie-breaking
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples]
        blocks_used = []
        
        while uncovered:
            max_gain = -1
            candidates = []
            for idx in range(num_blocks):
                if gains[idx] > max_gain:
                    max_gain = gains[idx]
                    candidates = [idx]
                elif gains[idx] == max_gain:
                    candidates.append(idx)
            
            if max_gain == 0:
                break
                
            b = random.choice(candidates)
            blocks_used.append(b)
            
            for t_idx in block_triples[b]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
        
        # Minimize
        minimized = minimize_cover(blocks_used)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_solution = minimized
            
        if time.time() - start_time > 100:
            break

    if best_solution is None:
        # Fallback
        best_solution = minimize_cover(greedy_construct())
        
    return [list(all_blocks[i]) for i in best_solution]
