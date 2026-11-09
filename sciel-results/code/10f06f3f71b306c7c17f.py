import itertools
import random
import time
import sys

def construct(v, k, t):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # Parameters
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute blocks and their triples
    # For v=24, k=7, there are 346,104 blocks.
    # Each block has C(7,3)=35 triples.
    # Storing all blocks and their triple indices is memory efficient enough.
    all_blocks = list(itertools.combinations(points, k))
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
        
    # Greedy Set Cover
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = [0] * n_triples
        blocks_used = []
        
        # To speed up greedy, we can just iterate. 
        # 346k blocks * 35 triples * ~80 iterations = ~1M operations per step? No, 
        # we need to find max gain. 
        # Naive: for each block, count uncovered triples. 346k * 35 = 12M checks per step.
        # 80 steps * 12M = 960M checks. In Python, this is too slow (seconds per step).
        
        # Optimization: 
        # We can use a "lazy" approach or precompute better.
        # Or, we can use a heuristic that doesn't check ALL blocks every time.
        # However, for v=24, 346k is manageable if we use NumPy?
        # Let's try a NumPy-based vectorized gain calculation.
        
        import numpy as np
        
        # Create a matrix: blocks x triples? 346104 x 2024 is 700M entries. Too big for memory (2.8GB).
        # So we cannot use a dense matrix.
        
        # Alternative: 
        # Maintain a list of uncovered triples.
        # For each candidate block, calculate gain = number of its triples in uncovered set.
        # We can represent uncovered as a boolean array of size n_triples.
        # For each block, we have a list of 35 triple indices.
        # Gain = sum(1 for idx in block_triples[i] if uncovered[idx])
        
        # We can speed this up by only considering blocks that contain at least one uncovered triple.
        # But maintaining that index is hard.
        
        # Let's stick to a pure Python greedy but optimized.
        # If it's too slow, we reduce the number of restarts.
        
        uncovered_count = n_triples
        while uncovered_count > 0:
            best_block = -1
            best_gain = -1
            # Optimization: If we have many uncovered, gain is high. 
            # We can sample blocks? No, greedy needs best.
            
            # We can use a trick: 
            # Pick a random uncovered triple. Only blocks containing it are candidates for "must cover it".
            # But a block might cover many others.
            
            # Let's try to just run the loop. 
            # 346,104 blocks. 
            # If we do this 100 times, it might be slow in Python.
            
            # Let's use a different heuristic: 
            # "Smallest Last" / "Most Constrained"
            # Find the uncovered triple with the fewest covering blocks available? 
            # No, that's for exact cover.
            
            # Let's try to accelerate the inner loop using PyPy-like thinking or just simple loops.
            # Actually, 346k * 35 = 12M operations. In Python, 12M simple ops is ~0.5-1s.
            # 80 blocks * 1s = 80s. This is borderline for 1 restart.
            # We need to be faster.
            
            # Idea: Precompute for each triple, the list of blocks containing it.
            # triple_to_blocks = [[] for _ in range(n_triples)]
            # for i, tris in enumerate(block_triples):
            #     for tri in tris:
            #         triple_to_blocks[tri].append(i)
            
            # Now, to find the best block:
            # We can't just pick one. We need the block with max total gain.
            # But we can restrict candidates to blocks that cover at least one "critical" triple?
            
            # Let's try a "Randomized Greedy" where we pick a random uncovered triple, 
            # and then pick the best block among those covering that triple.
            # This is a known heuristic (Greedy with focus).
            
            if uncovered_count > 0:
                # Pick a random uncovered triple
                # We need to efficiently pick a random uncovered triple.
                # We can maintain a list of uncovered triple indices.
                
                # To avoid O(N) scan for random uncovered, we can just use the 
                # "best block" search but limit it?
                
                # Let's try the standard greedy but with a time check.
                # If it takes too long, we break.
                
                pass # Placeholder

    # Given the speed constraints of Python, a full greedy over 346k blocks 80 times might be too slow.
    # Let's use a **Local Search** on a smaller candidate set or a **Greedy** that is faster.
    
    # Alternative Strategy: 
    # Use a **Genetic Algorithm** or **Simulated Annealing** on a smaller set of "good" blocks?
    # Or, use the fact that C(24,7,3) is small.
    
    # Let's try a **Greedy** that only considers a subset of blocks? No, that reduces quality.
    
    # Let's optimize the Greedy loop.
    # We can use NumPy for the gain calculation if we store block_triples as a NumPy array of shape (N_blocks, 35).
    # And uncovered as a boolean array of shape (N_triples,).
    # Gain for block i = np.sum(uncovered[block_triples[i]])
    # We can do this for all blocks at once?
    # Gains = np.sum(uncovered[block_triples], axis=1)
    # This is a vectorized operation. 
    # block_triples shape: (346104, 35). 
    # uncovered shape: (2024,).
    # Indexing: uncovered[block_triples] creates an array of shape (346104, 35).
    # Summing along axis 1 gives (346104,).
    # This is very fast in NumPy.
    
    import numpy as np
    
    block_triples_np = np.array(block_triples, dtype=np.int32)
    
    def fast_greedy(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks_used = []
        
        while np.any(uncovered):
            # Calculate gain for all blocks
            # uncovered[block_triples_np] -> (346104, 35) boolean array
            # sum along axis 1 -> (346104,) integer array
            gains = np.sum(uncovered[block_triples_np], axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if uncovered is not empty and blocks exist
            
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to add diversity
            if len(candidates) > 1:
                idx = rng.choice(candidates)
            else:
                idx = candidates[0]
                
            blocks_used.append(all_blocks[idx])
            
            # Update uncovered
            tris = block_triples_np[idx]
            uncovered[tris] = False
            
            # Time check
            if time.time() > deadline - 5:
                break
                
        return blocks_used

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    n_restarts = 50 # Adjust based on speed
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        blocks = fast_greedy(i)
        
        # Local Search: Remove redundant blocks
        # Convert to set of tuples for fast lookup
        # We need to check if removing a block leaves all its triples covered by others.
        
        # Optimization for removal:
        # Count how many blocks cover each triple.
        # If a triple is covered by only 1 block, that block is essential.
        # If a block's all triples are covered by >= 2 blocks, it can be removed.
        
        def local_search_remove(blocks):
            # Create a map from triple to list of block indices covering it
            # This is expensive to rebuild from scratch every time.
            # Instead, maintain a count array.
            
            # Initial counts
            counts = np.zeros(n_triples, dtype=np.int32)
            for b in blocks:
                # Need triple indices for b
                # We need a mapping from block tuple to block_triples_np index
                # Or just recompute triple indices for the blocks in the list.
                pass
            
            # Better: Keep the indices of the blocks in all_blocks.
            # fast_greedy returns tuples. We should return indices.
            
            pass

    # Let's refactor fast_greedy to return indices, and local search to work on indices.
    
    def fast_greedy_indices(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks_used_idx = []
        
        while np.any(uncovered):
            gains = np.sum(uncovered[block_triples_np], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                idx = rng.choice(candidates)
            else:
                idx = candidates[0]
                
            blocks_used_idx.append(idx)
            
            tris = block_triples_np[idx]
            uncovered[tris] = False
            
            if time.time() > deadline - 5:
                break
                
        return blocks_used_idx

    def local_search(indices):
        # indices is a list of block indices
        # We want to remove redundant blocks.
        # We maintain a count of how many blocks cover each triple.
        
        # Initialize counts
        counts = np.zeros(n_triples, dtype=np.int32)
        for idx in indices:
            counts[block_triples_np[idx]] += 1
            
        # List of active block indices
        active = list(indices)
        
        # Iteratively remove blocks that are not essential
        changed = True
        while changed:
            changed = False
            new_active = []
            for idx in active:
                # Check if all triples in this block are covered by at least 2 blocks
                # i.e., counts[tri] > 1 for all tri in block_triples_np[idx]
                tris = block_triples_np[idx]
                # If any triple has count 1, the block is essential
                if np.all(counts[tris] > 1):
                    # Remove it
                    counts[tris] -= 1
                    changed = True
                else:
                    new_active.append(idx)
            active = new_active
            
        return active

    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        idxs = fast_greedy_indices(i)
        reduced_idxs = local_search(idxs)
        
        if len(reduced_idxs) < best_len:
            best_len = len(reduced_idxs)
            best_blocks = [list(all_blocks[i]) for i in reduced_idxs]
            
    if best_blocks is None:
        # Fallback to simple greedy if something failed
        best_blocks = [list(all_blocks[i]) for i in fast_greedy_indices(0)]
        
    return best_blocks
