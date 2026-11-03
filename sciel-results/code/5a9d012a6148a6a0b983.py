"""Rewritten program using randomized greedy with urgency scoring and restarts."""
import itertools
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    total_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(total_tsubs)
    
    # Precompute index for each t-subset
    tsub_to_idx = {ts: i for i, ts in enumerate(total_tsubs)}
    
    def greedy_once(seed):
        rng = random.Random(seed)
        covered = np.zeros(n_tsubs, dtype=bool)
        blocks = []
        
        # Count how many k-blocks contain each t-subset (for urgency)
        # A t-subset is contained in C(v-t, k-t) blocks
        # But we track remaining "potential" differently
        
        # Urgency: number of uncovered t-subsets that MUST be covered
        # We'll use a simpler approach: pick random uncovered t-subset, 
        # but weight by how "hard" it is (fewer remaining candidate blocks)
        
        # For each point, track how many uncovered t-subsets it participates in
        # This helps with gain calculation
        
        while not covered.all():
            # Find all uncovered t-subsets
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            
            # Pick a t-subset to start with - use urgency: 
            # prefer t-subsets that are in fewer remaining possible blocks
            # Approximate urgency by: for a t-subset S, the number of 
            # k-blocks containing S is C(v-t, k-t), which is constant.
            # Instead, use: how many times has each point in S been used?
            # Points used less often are "more available"
            
            # Simpler: just pick random uncovered t-subset
            start_idx = uncovered_idx[rng.randint(0, len(uncovered_idx) - 1)]
            start_tsub = total_tsubs[start_idx]
            
            block = list(start_tsub)
            
            # Track which t-subsets this block covers so far
            # When adding point x, gain = number of uncovered t-subsets 
            # that contain x and (t-1) points from current block
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                # For each candidate point not in block
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain: how many uncovered t-subsets would be newly covered
                    # A t-subset covered by adding x must contain x and (t-1) points from block
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x == -1 or best_gain <= 0:
                    # No good point, pick random
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_x = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update covered
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    covered[idx] = True
        
        return blocks
    
    # Try multiple seeds and keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts - balance quality vs time
    n_restarts = 20
    
    for i in range(n_restarts):
        blocks = greedy_once(i * 12345 + 67890)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return best_blocks
