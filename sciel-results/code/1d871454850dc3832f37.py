import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs_in_block = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)
    
    # For each t-subset index, list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    def greedy(randomize=False):
        cover_count = [0] * n_tsubs
        block_gain = [len(tsubs) for tsubs in block_tsubs]
        remaining = set(range(n_blocks))
        selected = []
        
        # To speed up max finding, we can just loop. 
        # If randomize, we break ties randomly.
        
        while True:
            if all(cover_count[s] > 0 for s in range(n_tsubs)):
                break
            
            best_block = -1
            best_gain = -1
            candidates = []
            
            for b in remaining:
                if block_gain[b] > best_gain:
                    best_gain = block_gain[b]
                    best_block = b
                    if randomize:
                        candidates = [b]
                elif block_gain[b] == best_gain and randomize:
                    candidates.append(b)
            
            if randomize and candidates:
                best_block = random.choice(candidates)
            
            if best_block == -1 or best_gain <= 0:
                break
            
            selected.append(best_block)
            remaining.discard(best_block)
            
            for s_idx in block_tsubs[best_block]:
                if cover_count[s_idx] == 0:
                    cover_count[s_idx] = 1
                    for b in tsub_to_blocks[s_idx]:
                        if b in remaining:
                            block_gain[b] -= 1
        
        return selected

    def local_search_delete(blocks):
        selected = list(blocks)
        # Calculate initial cover counts
        cover_count = [0] * n_tsubs
        for b_idx in selected:
            for s_idx in block_tsubs[b_idx]:
                cover_count[s_idx] += 1
        
        improved = True
        while improved:
            improved = False
            # Shuffle to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                b_idx = selected[i]
                # Check if redundant
                redundant = True
                for s_idx in block_tsubs[b_idx]:
                    if cover_count[s_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove
                    selected.pop(i)
                    for s_idx in block_tsubs[b_idx]:
                        cover_count[s_idx] -= 1
                    improved = True
                    break # Restart loop after modification
        
        return selected

    best_solution = None
    start_time = time.time()
    
    # Run a few iterations
    while time.time() - start_time < 115:
        sol = greedy(randomize=True)
        sol = local_search_delete(sol)
        
        if best_solution is None or len(sol) < len(best_solution):
            best_solution = sol
            # If we found a very good one, we might stop early? 
            # Theoretical lower bound is 131. Best known 151.
            if len(sol) <= 151:
                # Can we do better? Maybe. But 151 is the best known.
                # Let's keep searching for a bit to see if we hit 150? Unlikely.
                pass

    # Fallback if no solution found (should not happen)
    if best_solution is None:
        best_solution = greedy(randomize=False)
        best_solution = local_search_delete(best_solution)

    return [list(all_blocks[b]) for b in best_solution]
