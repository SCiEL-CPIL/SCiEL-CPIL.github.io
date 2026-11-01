import itertools
import random
import time
import numpy as np
from collections import Counter

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a combination of:
    1. Randomized greedy construction with multiple restarts
    2. Local search / hill climbing to remove redundant blocks
    3. Deadline guard to ensure we return within time limit
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave 5s safety margin
    
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Precompute which triples each potential block covers
    # For efficiency, we'll work with triple indices
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    def greedy_construct(seed, max_blocks=None):
        """Randomized greedy covering construction."""
        rng = random.Random(seed)
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start
            start_triple = list(uncovered)[rng.randrange(len(uncovered))]
            block = set(start_triple)
            
            # Greedily add points that cover most uncovered triples
            while len(block) < k:
                best_point = None
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_triple = tuple(sorted(rest + (x,)))
                        if new_triple in triple_to_idx:
                            idx = triple_to_idx[new_triple]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain == 0:
                    # Pick a random point
                    best_point = rng.choice(candidates)
                
                block.add(best_point)
            
            block = sorted(block)
            blocks.append(block)
            
            # Remove covered triples
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                uncovered.discard(idx)
            
            if max_blocks and len(blocks) >= max_blocks:
                break
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove blocks that don't cover any unique triple."""
        if not blocks:
            return blocks
        
        # Count how many times each triple is covered
        triple_counts = Counter()
        for block in blocks:
            for triple in itertools.combinations(block, t):
                triple_counts[triple] += 1
        
        # Keep only blocks that cover at least one triple exclusively
        kept = []
        for block in blocks:
            unique = False
            for triple in itertools.combinations(block, t):
                if triple_counts[triple] == 1:
                    unique = True
                    break
            if unique:
                kept.append(block)
        
        # If we removed blocks, some triples might now be uncovered
        # Re-add necessary blocks
        if len(kept) < len(blocks):
            covered = set()
            for block in kept:
                for triple in itertools.combinations(block, t):
                    covered.add(triple)
            
            for block in blocks:
                if block in kept:
                    continue
                for triple in itertools.combinations(block, t):
                    if triple not in covered:
                        kept.append(block)
                        covered.update(itertools.combinations(block, t))
                        break
        
        return kept
    
    def local_search(blocks, max_iter=50):
        """Try to improve by swapping points in blocks."""
        blocks = list(blocks)
        for _ in range(max_iter):
            improved = False
            for i, block in enumerate(blocks):
                # Try replacing each point with another
                for j, pt in enumerate(block):
                    for new_pt in range(v):
                        if new_pt in block:
                            continue
                        new_block = block[:j] + [new_pt] + block[j+1:]
                        new_block = sorted(new_block)
                        
                        # Check if this swap doesn't uncover any triple
                        old_triples = set(itertools.combinations(block, t))
                        new_triples = set(itertools.combinations(new_block, t))
                        
                        # Check if all old triples are still covered by other blocks or new block
                        ok = True
                        for triple in old_triples - new_triples:
                            # This triple is no longer covered by this block
                            # Check if another block covers it
                            found = False
                            for other in blocks:
                                if other == block:
                                    continue
                                if triple in set(itertools.combinations(other, t)):
                                    found = True
                                    break
                            if not found:
                                ok = False
                                break
                        
                        if ok:
                            blocks[i] = new_block
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return blocks
    
    # Try multiple random seeds and keep the best
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        blocks = greedy_construct(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # Try local search on the best so far
        if time.time() < deadline - 5:
            improved = local_search(best_blocks, max_iter=20)
            improved = remove_redundant(improved)
            if len(improved) < best_len:
                best_len = len(improved)
                best_blocks = improved
        
        seed += 1
    
    # Final cleanup
    if best_blocks:
        best_blocks = remove_redundant(best_blocks)
        # Remove duplicates
        seen = set()
        unique_blocks = []
        for b in best_blocks:
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                unique_blocks.append(b)
        best_blocks = unique_blocks
    
    return best_blocks if best_blocks else []
