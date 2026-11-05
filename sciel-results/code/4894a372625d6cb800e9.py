import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = []
block_to_subsets = [] # List of lists of subset indices
subset_to_blocks = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    idxs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        idxs.append(idx)
        subset_to_blocks[idx].append(len(all_blocks))
    all_blocks.append(block)
    block_to_subsets.append(idxs)

N_BLOCKS = len(all_blocks)

# Precompute masks for fast bit operations? 
# 3876 bits fits in ~60 64-bit ints. Python int is fine.
block_masks = []
for idxs in block_to_subsets:
    m = 0
    for idx in idxs:
        m |= (1 << idx)
    block_masks.append(m)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115

    # State
    current_blocks = []
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    
    def add_block(b_idx):
        current_blocks.append(b_idx)
        for s_idx in block_to_subsets[b_idx]:
            counts[s_idx] += 1

    def remove_block(b_idx):
        current_blocks.remove(b_idx)
        for s_idx in block_to_subsets[b_idx]:
            counts[s_idx] -= 1

    def is_redundant(b_idx):
        for s_idx in block_to_subsets[b_idx]:
            if counts[s_idx] == 1:
                return False
        return True

    def find_replacement(uncovered_idxs, exclude_blocks):
        """
        Find a block that covers all subsets in uncovered_idxs.
        uncovered_idxs: list of subset indices
        exclude_blocks: set of block indices to ignore
        """
        if not uncovered_idxs:
            return None # Any block works, but we prefer not to add if not needed? 
                        # Actually if uncovered is empty, we don't need to add.
        
        # Sort by frequency of subset (rarest first) to minimize candidate set
        uncovered_idxs_sorted = sorted(uncovered_idxs, key=lambda x: len(subset_to_blocks[x]))
        
        candidates = set(subset_to_blocks[uncovered_idxs_sorted[0]])
        for idx in uncovered_idxs_sorted[1:]:
            if not candidates:
                break
            candidates.intersection_update(subset_to_blocks[idx])
            
        candidates.difference_update(exclude_blocks)
        
        if candidates:
            return random.choice(list(candidates))
        return None

    def local_search():
        improved = True
        while improved:
            improved = False
            # 1. Try to remove redundant blocks
            # Shuffle indices to avoid bias
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            for i in indices:
                b_idx = current_blocks[i]
                if is_redundant(b_idx):
                    remove_block(b_idx)
                    improved = True
                    break # Restart loop to check for new redundancies
            
            # 2. Try 2-for-1 swap
            # Only if we have enough blocks
            if len(current_blocks) >= 2:
                # Pick two random blocks
                # To be efficient, we can pick pairs involving "high overlap" blocks?
                # Random is simpler.
                # We try a few random pairs
                for _ in range(20): # Limit attempts per local search step
                    if time.time() > deadline:
                        return
                    i1 = random.randint(0, len(current_blocks) - 1)
                    i2 = random.randint(0, len(current_blocks) - 1)
                    if i1 == i2:
                        continue
                    
                    b1 = current_blocks[i1]
                    b2 = current_blocks[i2]
                    
                    # Determine which subsets become uncovered if we remove b1 and b2
                    # A subset is uncovered if count[b1] + count[b2] == 2 and no other block covers it?
                    # No, count is total.
                    # If we remove b1 and b2, the new count for s is counts[s] - (1 if s in b1 else 0) - (1 if s in b2 else 0).
                    # We need new count > 0 for all s.
                    # So for s in b1 U b2, we need counts[s] - delta(s) > 0.
                    # i.e., counts[s] > delta(s).
                    
                    uncovered_idxs = []
                    # Check subsets in b1
                    for s_idx in block_to_subsets[b1]:
                        delta = 1
                        if s_idx in block_to_subsets[b2]: # Check if s in b2
                             # Efficient check: use set or mask
                             pass
                        # We need to know if s_idx is in b2.
                        # Let's use a set for b2's subsets for O(1) lookup
                        pass
                    
                    # Optimization: Precompute set of subsets for each block?
                    # block_to_subsets is a list. Converting to set each time is expensive.
                    # Let's store block_subsets_set globally.
                    
                    # Re-compute logic with global sets
                    s1 = block_subsets_set[b1]
                    s2 = block_subsets_set[b2]
                    
                    union = s1 | s2
                    for s_idx in union:
                        delta = (1 if s_idx in s1 else 0) + (1 if s_idx in s2 else 0)
                        if counts[s_idx] == delta:
                            uncovered_idxs.append(s_idx)
                    
                    if not uncovered_idxs:
                        # Both blocks are redundant together?
                        # Remove one, then the other might be redundant
                        remove_block(b1)
                        improved = True
                        break
                    else:
                        # Try to find a replacement
                        # Exclude b1 and b2
                        repl = find_replacement(uncovered_idxs, {b1, b2})
                        if repl is not None:
                            remove_block(b1)
                            remove_block(b2)
                            add_block(repl)
                            improved = True
                            break

    # Precompute sets for fast intersection/union
    block_subsets_set = [set(idxs) for idxs in block_to_subsets]

    # Initialization
    # Greedy with random weights
    best_blocks = None
    best_len = float('inf')
    
    for _ in range(5): # Try a few initializations
        if time.time() > deadline: break
        
        # Reset state
        current_blocks = []
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        
        # Weights: inverse frequency + noise
        freqs = np.array([len(subset_to_blocks[i]) for i in range(N_SUBSETS)], dtype=float)
        weights = 1.0 / freqs
        weights += np.random.uniform(0, 0.1, N_SUBSETS)
        
        uncovered_mask = (1 << N_SUBSETS) - 1 # All uncovered
        # Using a bitmask for uncovered might be slow for 3876 bits? 
        # Python big ints are okay, but updating is O(size).
        # Using a boolean array is faster for updates.
        uncovered_arr = np.ones(N_SUBSETS, dtype=bool)
        
        while uncovered_arr.any():
            # Calculate gain for each block
            # gain[b] = sum(weights[s] for s in block_to_subsets[b] if uncovered_arr[s])
            # Vectorized approach:
            # We can precompute block_to_subsets as a matrix? 
            # Too big.
            # Loop is slow.
            
            # Alternative: Pick a random uncovered subset, and pick a random block containing it?
            # This is a "randomized greedy" or "set cover heuristic".
            # It might be faster and good enough.
            
            # Let's stick to a faster greedy:
            # Pick the block that covers the most uncovered subsets.
            # To do this fast, we can sample.
            
            # Actually, for 19,7,4, the number of blocks is 50k.
            # Looping 50k times in Python per step is too slow if we do 100 steps.
            # 50k * 100 = 5M operations. That's okay in 1s? Maybe.
            
            # Let's try a different init: 
            # Start with empty set.
            # While uncovered:
            #   Pick a random uncovered subset s.
            #   Pick a random block b containing s.
            #   Add b.
            
            # This is very fast.
            
            while uncovered_arr.any():
                # Pick random uncovered subset
                uncovered_indices = np.where(uncovered_arr)[0]
                s_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                
                # Pick random block containing s_idx
                candidates = subset_to_blocks[s_idx]
                b_idx = candidates[random.randint(0, len(candidates) - 1)]
                
                add_block(b_idx)
                
                # Update uncovered_arr
                for s in block_to_subsets[b_idx]:
                    if counts[s] == 1: # Newly covered
                        uncovered_arr[s] = False

        # Prune
        local_search()
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks.copy()
            # Update global state for next iterations?
            # We will restart from best_blocks in the main loop.

    # Main Loop: Iterated Local Search
    # Start from best_blocks
    current_blocks = best_blocks.copy()
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for b in current_blocks:
        for s in block_to_subsets[b]:
            counts[s] += 1

    temperature = 10.0
    cooling = 0.95
    
    while time.time() < deadline:
        # Local Search
        local_search()
        
        current_len = len(current_blocks)
        
        if current_len < best_len:
            best_len = current_len
            best_blocks = current_blocks.copy()
            temperature = 10.0 # Reset temp on improvement
        
        # Perturbation
        # Add 3 random blocks
        for _ in range(3):
            b_idx = random.randint(0, N_BLOCKS - 1)
            if b_idx not in current_blocks:
                add_block(b_idx)
        
        # Acceptance
        new_len = len(current_blocks)
        if new_len < best_len:
            # Already accepted in check above? 
            # No, we accept the perturbed state if it leads to improvement after next local search?
            # Standard ILS: Perturb, Local Search, Accept if better.
            # Here we perturb, then loop back to Local Search.
            # We should decide whether to keep the perturbed state or revert.
            # If we revert, we need to store the state.
            
            # Let's use Simulated Annealing on the length.
            # If new_len > best_len (current best before perturbation was best_len)
            # Actually, we compare current_len (after LS) with best_len.
            # If current_len > best_len, we should revert to best_blocks.
            # But we already modified current_blocks.
            
            # Strategy: 
            # 1. Save best_blocks.
            # 2. Perturb current_blocks.
            # 3. Local Search current_blocks.
            # 4. If len(current) < len(best), update best.
            # 5. Else, revert current to best.
            
            if current_len > best_len:
                # Revert
                current_blocks = best_blocks.copy()
                counts = np.zeros(N_SUBSETS, dtype=np.int32)
                for b in current_blocks:
                    for s in block_to_subsets[b]:
                        counts[s] += 1
            else:
                # Keep current
                pass
                
        temperature *= cooling
        if temperature < 0.1:
            temperature = 10.0 # Restart

    return [list(all_blocks[i]) for i in best_blocks]
