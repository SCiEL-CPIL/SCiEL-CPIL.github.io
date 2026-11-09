"""Improved greedy covering with multi-start and time guard."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill (evaluator says 150s wall-clock)
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple random seeds for diversity
    rng = random.Random(42)
    
    # Try several restarts with different random orderings
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Create a random permutation of t-subsets to break symmetry
        order = list(range(n_tsubsets))
        if restart > 0:
            rng.shuffle(order)
        
        # uncovered: boolean array
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick the first uncovered t-subset in our order
            # Find first index in 'order' that is uncovered
            found = False
            for idx in order:
                if uncovered[idx]:
                    start_subset = all_tsubsets[idx]
                    found = True
                    break
            
            if not found:
                break
            
            block = list(start_subset)
            
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                block_set = set(block)
                available = [x for x in range(v) if x not in block_set]
                
                if not available:
                    break
                
                # For each candidate, compute how many uncovered t-subsets it would cover
                # A t-subset {a1,...,a_{t-1}, x} is covered if {a1,...,a_{t-1}} is a subset of block
                # and x is the new point
                # More precisely: adding x to block covers all t-subsets S such that
                # x in S, S\{x} is a subset of block, and S is currently uncovered
                
                # We need: S contains x, S\{x} ⊆ block, and S is uncovered
                # S\{x} has t-1 elements, all in block
                # So we iterate over all (t-1)-subsets of block, and for each, check if
                # the t-subset (that (t-1)-subset plus x) is uncovered
                
                best_x = None
                best_gain = -1
                
                # Precompute (t-1)-subsets of current block
                block_t1 = list(itertools.combinations(block, t - 1))
                
                for x in available:
                    gain = 0
                    for rest in block_t1:
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_idx:
                            i = tsub_idx[ts]
                            if uncovered[i]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == 0:
                            # No point gives positive gain; pick any
                            break
                
                if best_x is None:
                    best_x = available[0]
                block.append(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            
            # Mark all t-subsets of this block as covered
            for ts in itertools.combinations(block, t):
                i = tsub_idx[ts]
                uncovered[i] = False
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # If we've already beaten the known best of 335, we can stop early
        if best_count <= 335:
            break
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return best_blocks
# EVOLVE-BLOCK-END
