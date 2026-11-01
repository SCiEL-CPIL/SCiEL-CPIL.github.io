import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precompute triple indices
    # Map (i, j, k) with i < j < k to index 0..C(v,t)-1
    # C(24, 3) = 2024
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute for each point x and pair (a, b) the index of the triple (min, mid, max)
    # Actually, we just need to check if a triple is uncovered.
    # To speed up gain calculation:
    # gain(x, block) = sum(1 for a,b in combinations(block, 2) if uncovered[triple_to_idx[sorted((x,a,b))]])
    
    # To make it even faster, precompute for each x and each pair (a,b) the index
    # idx_map[x][a][b] = index of triple containing x,a,b
    # Size 24 * 24 * 24 = 13824, very small.
    idx_map = [[[0 for _ in range(v)] for _ in range(v)] for _ in range(v)]
    for i in range(v):
        for j in range(v):
            for l in range(v):
                if i != j and j != l and i != l:
                    tri = tuple(sorted((i, j, l)))
                    idx_map[i][j][l] = triple_to_idx[tri]

    start_time = time.time()
    best_blocks = None
    best_score = float('inf')
    
    # Number of restarts based on time
    # Each greedy run takes some time. Let's say 0.1s per run? 
    # We have ~110s budget.
    
    for seed in range(1000): # Cap restarts, but break on time
        if time.time() - start_time > 110:
            break
            
        random.seed(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered triple to start the block
            # This ensures the block covers at least this triple
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            start_triple = triples[start_idx]
            
            block = list(start_triple)
            
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # Calculate gain for each candidate x not in block
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Iterate over pairs in block
                    # block size is small (< k=7)
                    for a_idx in range(len(block)):
                        a = block[a_idx]
                        for b_idx in range(a_idx + 1, len(block)):
                            b = block[b_idx]
                            idx = idx_map[x][a][b]
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                        
                if best_gain == 0:
                    # If no gain, pick random to break symmetry, though this shouldn't happen if uncovered exists
                    # Actually, if uncovered exists, there must be some gain?
                    # Not necessarily for the current block extension.
                    # But we started with an uncovered triple, so the first 3 points cover it.
                    # As we add points, we might not cover new ones.
                    # If best_gain is 0, we just pick a random point.
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        x = random.choice(candidates)
                    else:
                        break
                else:
                    x = random.choice(best_candidates)
                    
                block.append(x)
                
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tri in itertools.combinations(block, t):
                idx = triple_to_idx[tri]
                uncovered[idx] = False
                
        # Check if valid (should be)
        if not uncovered.any():
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
        # If we found a very good solution, we might stop early?
        # Best known is 76. If we get 76, we can stop? 
        # But maybe we can do better? Unlikely, but 76 is the target.
        if best_score <= 76:
            break

    if best_blocks is None:
        # Fallback to seed logic if something went wrong, or just return last attempt
        # But the loop should always produce a valid covering.
        return []
        
    return best_blocks
