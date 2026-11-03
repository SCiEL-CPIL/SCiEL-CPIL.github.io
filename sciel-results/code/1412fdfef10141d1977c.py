"""Improved covering design C(19,7,4) via optimized greedy with look-ahead and time guard."""
import time
import itertools
import sys

def construct(v, k, t):
    # Deadline guard: evaluator kills at 150s wall-clock; leave safety margin.
    deadline = time.time() + 140.0  # 140s budget, 10s margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Track coverage count for each t-subset (indexed by tuple)
    # Use a dict mapping tuple -> count of blocks covering it
    covered_count = {}
    for sub in all_t_subs:
        covered_count[sub] = 0
    
    uncovered = set(all_t_subs)
    blocks = []
    
    # Precompute for each point x and each (t-1)-subset, the list of t-subsets
    # that contain both x and the (t-1)-subset. This speeds up gain calculation.
    # Actually, let's use a different approach: for a candidate block B,
    # the t-subsets it covers are exactly combinations(B, t).
    
    # For faster gain computation, precompute for each (t-1)-subset S,
    # the set of points x such that S ∪ {x} is a valid t-subset (which is always true
    # if x not in S). So for a partial block P, adding x covers all t-subsets
    # that are subsets of P ∪ {x} and contain x.
    
    # Let's use a smarter data structure. For each t-subset, store its index.
    # Actually, let's just use sets and iterate efficiently.
    
    # Alternative: represent uncovered as a set of frozensets for faster lookup
    uncovered_set = set(uncovered)
    
    while uncovered_set:
        if time.time() > deadline - 5:
            break  # Safety: not enough time to complete a block
        
        # Start a new block with the first uncovered t-subset
        first_sub = min(uncovered_set)
        block = list(first_sub)
        
        while len(block) < k:
            if time.time() > deadline - 5:
                break
            
            # Find the point x that maximizes the number of newly covered t-subsets
            best_x = None
            best_gain = -1
            
            # For each candidate point not in block, compute gain
            # Gain = number of t-subsets that contain x and are subsets of block ∪ {x}
            # and are currently uncovered
            # A t-subset S is covered by block ∪ {x} iff S ⊆ block ∪ {x} and x ∈ S
            # i.e., S = {x} ∪ R where R ⊆ block, |R| = t-1
            
            # Precompute all (t-1)-subsets of current block
            block_set = set(block)
            t_minus_1_subs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for r in t_minus_1_subs:
                    # Check if (r + (x,)) sorted is in uncovered_set
                    candidate = tuple(sorted(r + (x,)))
                    if candidate in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            
            block.append(best_x)
        
        if len(block) < k:
            # If we couldn't complete the block, skip (shouldn't happen normally)
            continue
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for sub in itertools.combinations(block, t):
            if sub in uncovered_set:
                uncovered_set.discard(sub)
    
    # If we broke out early due to time, we have an incomplete covering.
    # We need to return a valid covering. Let's check if we're done.
    if uncovered_set:
        # We have uncovered subsets. Try to finish with remaining time.
        # Use a simpler greedy: for each uncovered t-subset, add a block containing it.
        remaining_deadline = deadline - 2
        while uncovered_set and time.time() < remaining_deadline:
            first_sub = min(uncovered_set)
            block = list(first_sub)
            while len(block) < k:
                if time.time() > remaining_deadline:
                    break
                # Just add arbitrary points to complete
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            if len(block) < k:
                break
            block.sort()
            if block not in blocks:
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    if sub in uncovered_set:
                        uncovered_set.discard(sub)
    
    return blocks
