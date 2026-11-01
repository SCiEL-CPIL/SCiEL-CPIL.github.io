"""Covering design C(19,8,5) - optimized greedy with randomized restarts."""
import time
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Precompute all t-subsets as frozensets for fast membership testing
    all_tsubsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts we can afford
    # Each restart takes roughly O(num_blocks * k * v * C(k-1, t-1)) operations
    # For v=19, k=8, t=5: C(7,4)=35, so per block ~19*35=665 ops, ~335 blocks -> ~223k ops
    # That's fast in Python if we optimize the inner loop
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 13)
        
        # Greedy construction with randomization
        # Track uncovered t-subsets as a set of frozensets
        uncovered = set(frozenset(s) for s in all_tsubsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to start the block
            target = random.choice(tuple(uncovered))
            block = set(target)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_gain = -1
                best_point = -1
                best_tie_points = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Compute how many new t-subsets adding x would cover
                    gain = 0
                    # For each (t-1)-subset of block, check if adding x creates an uncovered t-subset
                    for rest in itertools.combinations(sorted(block), t - 1):
                        new_subset = frozenset(list(rest) + [x])
                        if new_subset in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_tie_points = [x]
                    elif gain == best_gain and gain > 0:
                        best_tie_points.append(x)
                
                if best_point == -1:
                    # No point adds coverage, just pick a random one
                    candidates = [x for x in range(v) if x not in block]
                    if not candidates:
                        break
                    best_point = random.choice(candidates)
                elif len(best_tie_points) > 1:
                    # Random tie-breaking for diversity
                    best_point = random.choice(best_tie_points)
                
                block.add(best_point)
            
            block = sorted(block)
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for subset in itertools.combinations(block, t):
                fs = frozenset(subset)
                uncovered.discard(fs)
        
        if not uncovered and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            if time.time() > deadline:
                break
        
        # If we're running out of time, stop
        if time.time() > deadline - 5:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(frozenset(s) for s in all_tsubsets)
        blocks = []
        while uncovered:
            target = min(uncovered, key=lambda s: list(s))
            block = set(target)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(sorted(block), t - 1):
                        new_subset = frozenset(list(rest) + [x])
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block]
                    best_point = min(candidates) if candidates else 0
                block.add(best_point)
            blocks.append(sorted(block))
            for subset in itertools.combinations(blocks[-1], t):
                uncovered.discard(frozenset(subset))
        best_blocks = blocks
    
    return best_blocks
