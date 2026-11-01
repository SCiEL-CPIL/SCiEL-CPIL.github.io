import itertools
import numpy as np
import time
import random

# Global cache for precomputed data
_PRECOMPUTED = None

def _precompute(v, k, t):
    global _PRECOMPUTED
    if _PRECOMPUTED is not None:
        return _PRECOMPUTED
        
    # 1. Generate all triplets and map to indices
    triplets = list(itertools.combinations(range(v), t))
    triplet_map = {triplet: i for i, triplet in enumerate(triplets)}
    n_triplets = len(triplets)
    
    # 2. Generate all blocks and their triplet indices
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    block_triplets_list = []
    for b in blocks:
        b_triplets = []
        for tri in itertools.combinations(b, t):
            b_triplets.append(triplet_map[tri])
        block_triplets_list.append(b_triplets)
        
    block_triplets = np.array(block_triplets_list, dtype=np.int32)
    
    # 3. Build mapping from triplet index to list of block indices
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    for i, b_triplets in enumerate(block_triplets_list):
        for t_idx in b_triplets:
            triplet_to_blocks[t_idx].append(i)
            
    # Convert to numpy arrays for faster access? 
    # Actually, lists are fine for fetching the candidate list, 
    # but we need to index block_triplets with these candidates.
    # block_triplets[candidates] works if candidates is a list or array.
    
    _PRECOMPUTED = {
        'blocks': blocks,
        'block_triplets': block_triplets,
        'triplet_to_blocks': triplet_to_blocks,
        'n_triplets': n_triplets
    }
    return _PRECOMPUTED

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    data = _precompute(v, k, t)
    blocks = data['blocks']
    block_triplets = data['block_triplets']
    triplet_to_blocks = data['triplet_to_blocks']
    n_triplets = data['n_triplets']
    
    best_solution = None
    best_size = float('inf')
    
    # Number of restarts to attempt
    # We will run as many as possible before deadline
    max_restarts = 200 
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        covered = np.zeros(n_triplets, dtype=bool)
        current_blocks = []
        
        while not covered.all():
            if time.time() > deadline:
                break
                
            # Find an uncovered triplet
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick a random uncovered triplet
            target_triplet_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            # Get all blocks covering this triplet
            candidates = triplet_to_blocks[target_triplet_idx]
            
            if not candidates:
                # Should not happen for valid v,k,t
                break
                
            # Calculate gain for each candidate
            # gain = number of uncovered triplets covered by the block
            # block_triplets[candidates] has shape (num_cands, t_combinations)
            # We need to count how many of these are NOT in covered
            # covered is a boolean array of size n_triplets
            
            # Extract triplet indices for candidates
            cand_triplets = block_triplets[candidates]
            
            # Check which are uncovered
            # ~covered is boolean array
            # We need to index ~covered with cand_triplets
            # This results in a boolean array of shape (num_cands, 35)
            is_uncovered = ~covered[cand_triplets]
            
            # Sum along axis 1 to get gain per candidate
            gains = np.sum(is_uncovered, axis=1)
            
            # Find the best block
            # To add randomness, we can pick from the top K
            # Sort gains descending
            # np.argsort is O(N log N). N is ~6000. This is fast.
            # Or just use argmax for standard greedy. 
            # Randomized: pick randomly from top 10%
            
            # Let's try picking the max, but with a small probability pick a random one from top 5
            # Or just standard greedy first to see baseline speed.
            
            # Let's use a "soft" max or top-k selection to escape local minima.
            # Find indices of top 20 candidates
            top_k = min(20, len(candidates))
            # Get indices of top k gains
            # Using argsort on 6000 elements is fine.
            top_indices = np.argsort(gains)[-top_k:]
            
            # Pick randomly from these top indices
            best_idx_in_top = random.choice(top_indices)
            best_candidate_idx = candidates[best_idx_in_top]
            
            # Add block to solution
            current_blocks.append(blocks[best_candidate_idx])
            
            # Update covered
            covered[block_triplets[best_candidate_idx]] = True
            
        if time.time() > deadline:
            break
            
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_solution = current_blocks
            # If we reach the known lower bound or optimal, we can stop early?
            # Lower bound is 69. Known best is 76.
            # If we hit 76, we can keep trying to see if we get 75.
            # But 76 is the target.
            
    # Return best solution found
    if best_solution is None:
        # Fallback to a simple greedy if something went wrong
        # But this shouldn't happen
        return []
        
    # Convert tuples to lists to match interface expectation (list of lists)
    return [list(b) for b in best_solution]
