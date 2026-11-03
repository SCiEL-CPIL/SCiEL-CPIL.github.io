import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy passes with different random seeds
    for seed in range(20):  # Try up to 20 restarts
        if time.time() > deadline:
            break
        
        random.seed(seed)
        blocks = greedy_cover(v, k, t, all_triples)
        
        # Remove redundant blocks
        blocks = remove_redundant_blocks(blocks, v, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
            # If we hit the known best or lower, we can stop early if confident
            if best_len <= 76:
                break

    if best_blocks is None:
        # Fallback to a simple deterministic greedy if something went wrong
        best_blocks = greedy_cover(v, k, t, all_triples, deterministic=True)
        best_blocks = remove_redundant_blocks(best_blocks, v, t)
        
    return best_blocks

def greedy_cover(v, k, t, all_triples, deterministic=False):
    uncovered = set(all_triples)
    blocks = []
    
    # Precompute blocks containing each triple? 
    # No, generate on the fly.
    
    while uncovered:
        if deterministic:
            target_triple = min(uncovered)
        else:
            target_triple = random.choice(list(uncovered))
        
        # Generate all blocks containing target_triple
        # target_triple is a tuple of t points
        # We need to choose k-t points from the remaining v-t points
        remaining_points = [p for p in range(v) if p not in target_triple]
        
        best_block = None
        best_gain = -1
        
        # To speed up, we can just iterate through combinations
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target_triple + combo))
            
            # Calculate gain: how many uncovered triples are in this block?
            gain = 0
            for triple in itertools.combinations(block, t):
                if triple in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
                # Early exit if we found a block that covers ALL uncovered triples? Unlikely.
                # But if gain is very high, we might break? No, just find max.
        
        blocks.append(best_block)
        
        # Remove covered triples
        for triple in itertools.combinations(best_block, t):
            if triple in uncovered:
                uncovered.discard(triple)
                
    return [list(b) for b in blocks]

def remove_redundant_blocks(blocks, v, t):
    # Convert to sets for faster intersection? Or just check coverage.
    # A block is redundant if every triple in it is covered by at least one other block.
    blocks_set = [set(b) for b in blocks]
    
    # To make it efficient, we can maintain a count of how many blocks cover each triple.
    # But blocks are large.
    # Simpler: Iterate through blocks. For each block, check if it can be removed.
    # To check if block B can be removed:
    # For each triple in B, is there another block in blocks (excluding B) that contains that triple?
    
    # This is O(N * 35 * N) which is too slow if N is 100. 100*35*100 = 350,000. This is fine.
    
    removed = True
    while removed:
        removed = False
        new_blocks = []
        for i, b in enumerate(blocks):
            # Check if b is redundant
            redundant = True
            for triple in itertools.combinations(b, t):
                # Check if any other block covers this triple
                covered_by_other = False
                for j, other_b in enumerate(blocks):
                    if i == j:
                        continue
                    if all(p in other_b for p in triple):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                removed = True
            else:
                new_blocks.append(b)
        
        blocks = new_blocks
        
    return blocks
