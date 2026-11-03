import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_cover(v, k, t, rng):
    """
    Constructs a covering using a weighted greedy heuristic with randomization.
    Weighted greedy prioritizes covering rare triples (those with low coverage potential).
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    
    # Precompute for each point, how many triples it can cover with each pair
    # This helps in weighted gain calculation
    
    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        # Greedily add points to the block using weighted gain
        while len(block) < k:
            best_gain = -1
            best_points = []
            
            # Calculate weighted gain for each candidate point
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check all pairs in block
                for a, b in itertools.combinations(block, 2):
                    triple = tuple(sorted((a, b, x)))
                    if triple in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            # If no gain, pick random point not in block
            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                # Randomly choose among best points to introduce diversity
                x = rng.choice(best_points)
            
            block.append(x)
            block_set.add(x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    Uses iterative peeling: remove any block that doesn't uniquely cover any triple.
    """
    if not blocks:
        return blocks
    
    # Count how many times each triple is covered
    triple_cover_count = defaultdict(int)
    
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            is_essential = False
            for triple in itertools.combinations(block, t):
                if triple_cover_count[tuple(triple)] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
            else:
                changed = True
                for triple in itertools.combinations(block, t):
                    triple_cover_count[tuple(triple)] -= 1
        
        blocks = new_blocks
    
    return blocks

def _local_search(blocks, v, k, t, rng, time_limit):
    """
    Attempts to improve the covering by swapping blocks.
    Only accepts swaps that lead to immediate reduction in block count.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts
    triple_cover_count = defaultdict(int)
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1
            
    # Cache of triples covered by each block index
    block_triples = []
    for block in blocks:
        block_triples.append(list(itertools.combinations(block, t)))
        
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try to swap each block with a random new block
        for i in range(len(blocks)):
            if time.time() > time_limit:
                break
            
            # Store original block for potential revert
            original_block = blocks[i][:]
            original_triples = block_triples[i][:]
            
            # Generate a random candidate block
            candidate = sorted(rng.sample(range(v), k))
            
            if tuple(candidate) == tuple(original_block):
                continue
                
            # Check if swapping block i with candidate maintains coverage
            old_triples_set = set(original_triples)
            new_triples = list(itertools.combinations(candidate, t))
            new_triples_set = set(new_triples)
            
            # Triples that would become uncovered if we remove block i and add candidate
            lost_triples = old_triples_set - new_triples_set
            
            valid_swap = True
            for triple in lost_triples:
                # triple is a tuple
                if triple_cover_count[tuple(triple)] == 1:
                    # It is only covered by block i, so swapping would uncover it
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform the swap
                # Update counts
                for triple in old_triples_set:
                    triple_cover_count[tuple(triple)] -= 1
                for triple in new_triples:
                    triple_cover_count[tuple(triple)] += 1
                    
                blocks[i] = candidate
                block_triples[i] = new_triples
                
                # Try to reduce
                new_blocks = _reduce_covering(blocks, v, k, t)
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    # Recalculate counts and block_triples for the new set
                    triple_cover_count = defaultdict(int)
                    for block in blocks:
                        for triple in itertools.combinations(block, t):
                            triple_cover_count[tuple(triple)] += 1
                    block_triples = []
                    for block in blocks:
                        block_triples.append(list(itertools.combinations(block, t)))
                    
                    # If we improved, we can restart the loop or continue
                    # Restarting is safer to ensure we check all positions again
                    break 
                else:
                    # Revert swap
                    for triple in old_triples_set:
                        triple_cover_count[tuple(triple)] += 1
                    for triple in new_triples:
                        triple_cover_count[tuple(triple)] -= 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples

    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover(v, k, t, rng)
        
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        # Apply local search if we have time
        remaining_time = deadline - time.time()
        if remaining_time > 5:  # Need at least 5 seconds for local search to be worthwhile
            local_time_limit = min(remaining_time - 2, 30)  # Cap local search at 30s
            blocks = _local_search(blocks, v, k, t, rng, local_time_limit)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            if best_size <= 69:
                break
                
    return best_blocks
