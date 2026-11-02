import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Pre-compute all t-subsets for efficiency if needed, but for v=19, t=5, 
    # generating on the fly or using sets is fine.
    # However, checking membership in a set of tuples is fast.
    
    best_blocks = None
    best_len = float('inf')
    
    # Use a fixed seed for reproducibility if needed, but random is better for exploration.
    # We will loop with different random states.
    
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        blocks = _greedy_construct(v, k, t)
        
        # 2. Redundancy Removal
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks] # Deep copy
            
            # Early exit if we hit the theoretical lower bound or known best?
            # Lower bound for C(19,8,5) is 264. Known best is 335.
            # If we find something <= 335, it's good. If we hit 264, we can stop.
            if best_len <= 335:
                # We can try to improve further, but 335 is the record.
                # Let's continue searching for a few more seconds just in case.
                pass
        
        # If time is tight, break
        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        # Fallback: simple deterministic greedy (should not happen)
        best_blocks = _greedy_construct(v, k, t)
        
    return best_blocks

def _greedy_construct(v, k, t):
    """
    Randomized greedy algorithm to construct a covering.
    """
    # Set of all t-subsets
    all_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to start a new block
        # Using min() is deterministic, random.choice is better for diversity
        # But we need to convert set to list for random choice, which is O(N).
        # For 11k items, it's fast.
        seed_subset = random.choice(list(uncovered))
        
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            # Candidates are points not in block
            candidates = [x for x in range(v) if x not in block]
            
            for x in candidates:
                # Calculate gain: how many uncovered t-subsets are covered by adding x?
                # A subset S is covered by B U {x} if x in S and S \ {x} subset B.
                # So we look for subsets S in uncovered such that x in S and S \ {x} is a (t-1)-subset of B.
                
                gain = 0
                # Iterate over all (t-1)-subsets of the current block
                # There are C(len(block), t-1) of them.
                for rest in itertools.combinations(block, t - 1):
                    # Construct the t-subset containing x and rest
                    # We need to ensure x is not in rest (it isn't, since x not in block)
                    # The tuple must be sorted to match the keys in 'uncovered'
                    s = tuple(sorted(list(rest) + [x]))
                    if s in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain:
                    # Tie-breaking: random choice among best candidates
                    # To avoid O(N) list of ties, we can just pick randomly if gain > best_gain
                    # or use a small random probability to replace.
                    if random.random() < 0.5:
                        best_x = x
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks

def _remove_redundant_blocks(blocks, v, t):
    """
    Remove blocks that are not necessary for the covering.
    """
    if not blocks:
        return blocks
    
    # Pre-compute all t-subsets for each block
    block_subsets = []
    for b in blocks:
        block_subsets.append(set(itertools.combinations(b, t)))
    
    # Count how many times each t-subset is covered
    subset_count = Counter()
    for subs in block_subsets:
        for s in subs:
            subset_count[s] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_subsets = []
        for i, b in enumerate(blocks):
            # Check if any subset in block i is covered only by block i
            is_redundant = True
            for s in block_subsets[i]:
                if subset_count[s] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                for s in block_subsets[i]:
                    subset_count[s] -= 1
                changed = True
            else:
                new_blocks.append(b)
                new_block_subsets.append(block_subsets[i])
        
        blocks = new_blocks
        block_subsets = new_block_subsets
        
    return blocks
