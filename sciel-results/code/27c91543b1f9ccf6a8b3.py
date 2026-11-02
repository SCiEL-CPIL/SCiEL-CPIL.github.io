import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    all_points = list(range(v))
    all_triples = list(itertools.combinations(all_points, t))
    num_triples = len(all_triples)
    
    # Map triple to index for O(1) lookups
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Uncovered as a boolean numpy array (True = uncovered)
    uncovered = np.ones(num_triples, dtype=bool)
    uncovered_count = num_triples
    
    blocks = []
    triples_per_block = comb(k, t)
    
    # Precompute complement for each triple
    triple_complement = {}
    for tri in all_triples:
        tri_set = set(tri)
        triple_complement[tri] = tuple(p for p in all_points if p not in tri_set)
    
    # Greedy Construction
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered triple
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        target_triple_idx = int(random.choice(uncovered_indices))
        target_triple = all_triples[target_triple_idx]
        
        # Generate all blocks containing target_triple
        remaining = triple_complement[target_triple]
        
        best_block = None
        max_covered = -1
        
        for combo in itertools.combinations(remaining, k - t):
            block = tuple(sorted(target_triple + combo))
            
            # Count uncovered triples in this block
            covered_count = 0
            for tri in itertools.combinations(block, t):
                if uncovered[triple_to_idx[tri]]:
                    covered_count += 1
            
            if covered_count > max_covered:
                max_covered = covered_count
                best_block = block
        
        if best_block is None or max_covered == 0:
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        for tri in itertools.combinations(best_block, t):
            idx = triple_to_idx[tri]
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
    
    # Redundancy Removal with triple-to-blocks mapping
    if time.time() < deadline - 5.0:
        # Build triple-to-blocks mapping
        triple_to_blocks = {i: [] for i in range(num_triples)}
        for b_idx, block in enumerate(blocks):
            for tri in itertools.combinations(block, t):
                tri_idx = triple_to_idx[tri]
                triple_to_blocks[tri_idx].append(b_idx)
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                redundant = True
                for tri in itertools.combinations(block, t):
                    tri_idx = triple_to_idx[tri]
                    # Check if tri is in any other block
                    other_blocks = triple_to_blocks[tri_idx]
                    if len(other_blocks) <= 1:
                        # Only this block covers tri, so not redundant
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(block)
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                changed = True
                # Rebuild triple_to_blocks
                triple_to_blocks = {i: [] for i in range(num_triples)}
                for b_idx, block in enumerate(blocks):
                    for tri in itertools.combinations(block, t):
                        tri_idx = triple_to_idx[tri]
                        triple_to_blocks[tri_idx].append(b_idx)
    
    return blocks
