"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets for faster membership checks
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsub_set = set(all_tsubs)
    
    # Greedy covering with randomized restarts and local search
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    for restart in range(200):  # cap restarts
        if time.time() > deadline:
            break
        
        # Randomized greedy: pick a random uncovered t-subset to start each block
        uncovered = set(all_tsub_set)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple to start
            start_triple = rng.choice(list(uncovered))
            block = list(start_triple)
            block_set = set(block)
            
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets this point would cover
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        
        if not uncovered and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # If we've found a good solution, do local search to improve it
        if best_blocks is not None and len(best_blocks) <= 80:
            improved = _local_search(best_blocks, v, k, t, all_tsub_set, rng, deadline)
            if improved is not None and len(improved) < best_count:
                best_blocks = improved
                best_count = len(best_blocks)
    
    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = set(all_tsub_set)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks


def _local_search(blocks, v, k, t, all_tsub_set, rng, deadline):
    """Try to reduce the number of blocks by removing redundant ones."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # For each block, check if it's redundant (all its t-subsets are covered by other blocks)
    for i in range(n - 1):
        if time.time() > deadline:
            return None
        
        # Temporarily remove block i
        remaining = blocks[:i] + blocks[i+1:]
        
        # Check if all t-subsets of block i are still covered
        redundant = True
        for tsub in itertools.combinations(blocks[i], t):
            covered = False
            for j, block in enumerate(remaining):
                if set(tsub).issubset(set(block)):
                    covered = True
                    break
            if not covered:
                redundant = False
                break
        
        if redundant:
            blocks = remaining
            n -= 1
    
    return blocks
# EVOLVE-BLOCK-END
