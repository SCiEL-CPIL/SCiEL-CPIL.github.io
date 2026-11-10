"""Optimized covering design C(19,8,5) using greedy with randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a greedy set-cover approach
    with randomized restarts and a time budget.
    """
    start_time = time.time()
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    
    # Precompute for each (k-1)-subset, which t-subsets it can cover
    # A (k-1)-subset S can cover t-subsets T where T \ S has exactly 1 element
    # i.e., T = S' where S' is a t-subset that contains k-1 elements of S plus one more
    
    # Actually, let's think differently. For a block B (k-subset), it covers all
    # t-subsets contained in B.
    
    # For the greedy approach: we maintain a count of how many uncovered t-subsets
    # each potential block would cover. But that's expensive.
    
    # Alternative: standard greedy - at each step, pick the block that covers the
    # most uncovered t-subsets. To do this efficiently, we can use the following:
    # For each point x, track how many uncovered t-subsets contain x.
    # When adding point x to a partial block, the gain is the number of uncovered
    # t-subsets that contain x and all other elements of the partial block (as a subset).
    
    # Let's use a different approach: maintain for each t-subset whether it's covered.
    # For each candidate point x, compute how many new t-subsets it covers given the
    # current partial block.
    
    def greedy_cover(seed, time_limit):
        """Run one greedy covering attempt with a given random seed."""
        random.seed(seed)
        
        # uncovered[i] = True if t-subset i is not yet covered
        uncovered = [True] * total_tsubsets
        num_uncovered = total_tsubsets
        blocks = []
        
        # For efficiency, we'll track for each point and each (t-1)-subset of remaining
        # points, how many uncovered t-subsets this combination covers.
        # This is complex. Let's use a simpler but effective approach.
        
        # For each point x, precompute which t-subsets contain x
        point_to_tsubsets = defaultdict(list)
        for i, ts in enumerate(all_tsubsets):
            for x in ts:
                point_to_tsubsets[x].append(i)
        
        while num_uncovered > 0:
            if time.time() - start_time > time_limit:
                return None  # timeout
            
            # Start a new block: pick the first uncovered t-subset
            # Find an uncovered t-subset
            first_uncovered = None
            for i in range(total_tsubsets):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered is None:
                break
            
            block = list(all_tsubsets[first_uncovered])
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_point = -1
                best_gain = 0
                
                # For each candidate point not in block, compute gain
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain = number of uncovered t-subsets that contain x
                    # and have all other (t-1) elements in block
                    # i.e., t-subsets T where x in T and T \ {x} ⊆ block
                    # This means T = (T \ {x}) ∪ {x} where T \ {x} is a (t-1)-subset of block
                    gain = 0
                    # Iterate over (t-1)-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1 or best_gain == 0:
                    # Pick a random point
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_point = random.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = False
                    num_uncovered -= 1
        
        return blocks
    
    # Try multiple seeds and keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # First, try a deterministic greedy (no randomization)
    blocks = greedy_cover(42, _DEADLINE - (time.time() - start_time))
    if blocks is not None:
        best_blocks = blocks
        best_len = len(blocks)
    
    # Try more seeds if time permits
    seed = 0
    while time.time() - start_time < _DEADLINE - 5:
        seed += 1
        blocks = greedy_cover(seed, _DEADLINE - (time.time() - start_time))
        if blocks is not None and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
        if best_len <= 300:  # good enough, stop early
            break
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
