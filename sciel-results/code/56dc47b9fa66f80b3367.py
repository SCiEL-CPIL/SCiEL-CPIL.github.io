import itertools
import time
import random
from collections import defaultdict

def _greedy_random(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    
    # Pre-compute all t-subsets as frozensets for O(1) membership
    uncovered = set(frozenset(combo) for combo in itertools.combinations(range(v), t))
    blocks = []
    
    if not uncovered:
        return []

    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: how many uncovered triples are covered by adding x
                gain = 0
                if t == 3:
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tri = frozenset((block[i], block[j], x))
                            if tri in uncovered:
                                gain += 1
                else:
                    for comb in itertools.combinations(block, t-1):
                        tri = frozenset(list(comb) + [x])
                        if tri in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            # Pick a random candidate from the best ones
            if candidates:
                next_point = rng.choice(candidates)
            else:
                next_point = rng.choice([x for x in all_points if x not in block_set])
            
            block.append(next_point)
            block_set.add(next_point)
        
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for tri in itertools.combinations(block, t):
            uncovered.discard(frozenset(tri))
            
    return blocks

def _local_search(v, k, t, blocks, rng, deadline):
    """Try to remove blocks if the covering remains valid."""
    blocks = [list(b) for b in blocks]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle block order for randomness
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            # Try removing block i
            candidate_blocks = blocks[:i] + blocks[i+1:]
            
            # Check if covering is still valid
            # Build set of all t-subsets covered by candidate_blocks
            covered = set()
            for b in candidate_blocks:
                for tri in itertools.combinations(b, t):
                    covered.add(frozenset(tri))
            
            # Check if all t-subsets are covered
            all_subs = set(frozenset(combo) for combo in itertools.combinations(range(v), t))
            if all_subs.issubset(covered):
                # Removal successful
                blocks = candidate_blocks
                improved = True
                break
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline:
        # Check if we have enough time for a full greedy + local search
        if time.time() + 10.0 > deadline:
            break
            
        blocks = _greedy_random(v, k, t, seed)
        
        # Apply local search if we have time
        if time.time() + 5.0 < deadline:
            rng = random.Random(seed + 1000000)
            blocks = _local_search(v, k, t, blocks, rng, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # Stop early if we hit theoretical lower bound
        if best_len == 69:
            break
            
        seed += 1
        
    return best_blocks
