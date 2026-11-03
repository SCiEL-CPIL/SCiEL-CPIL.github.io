"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples
       to maximize the reduction in the entropy of the uncovered triple distribution.
    2. Local Search: Optimize the initial solution by removing redundant blocks
       and replacing blocks with more efficient ones.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy(v, k, t, all_triples, triple_to_idx)
    
    # Phase 2: Local Search Optimization
    blocks = _local_search(v, k, t, all_triples, triple_to_idx, blocks)
    
    return blocks

def _info_greedy(v, k, t, all_triples, triple_to_idx):
    """
    Greedy covering that maximizes entropy reduction.
    We approximate entropy reduction by prioritizing triples with 'rare' points
    (points that appear in few uncovered triples).
    """
    uncovered = [1] * len(all_triples)
    point_uncovered_count = [0] * v
    
    # Initialize point counts
    for idx, triple in enumerate(all_triples):
        for p in triple:
            point_uncovered_count[p] += 1
            
    blocks = []
    
    while any(uncovered):
        # Find the "most scarce" uncovered triple
        # Scarcity is defined by the sum of 1/point_count for its points.
        # Higher score = more scarce/isolated.
        best_triple_idx = -1
        best_score = -1.0
        
        # Optimization: We can't iterate all triples every time if it's slow.
        # But for v=24, t=3, there are 2024 triples. 76 blocks * 2024 * 3 is ~450k ops.
        # This is fast enough in Python if we keep it tight.
        
        for idx in range(len(all_triples)):
            if uncovered[idx]:
                triple = all_triples[idx]
                score = 0.0
                for p in triple:
                    score += 1.0 / (point_uncovered_count[p] + 1)
                if score > best_score:
                    best_score = score
                    best_triple_idx = idx
                    
        if best_triple_idx == -1:
            break
            
        best_triple = all_triples[best_triple_idx]
        block = list(best_triple)
        
        # Extend the block by picking points that maximize the weighted gain
        # Weighted gain = sum of scores of newly covered triples
        while len(block) < k:
            best_point = -1
            best_gain = -1.0
            
            # Candidate points: those not in block
            # To speed up, we only consider points that appear in uncovered triples intersecting current block?
            # No, we just iterate all v points, it's small (24).
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
                # Calculate weighted gain of adding x
                # New triples formed: {block[i], block[j], x}
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        idx = triple_to_idx.get(triple, -1)
                        if idx != -1 and uncovered[idx]:
                            # Weight this triple by rarity of its points
                            w = 0.0
                            for p in triple:
                                w += 1.0 / (point_uncovered_count[p] + 1)
                            gain += w
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered and point counts
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            if uncovered[idx]:
                uncovered[idx] = 0
                for p in triple:
                    point_uncovered_count[p] -= 1
            # If already covered, do nothing (block is distinct, but triple might be covered by previous blocks? 
            # No, greedy ensures we pick uncovered. But wait, if a triple was covered by a PREVIOUS block, 
            # it would be 0. So we only decrement if it was 1.
            # Actually, we should just decrement if it was uncovered.
            
    return blocks

def _local_search(v, k, t, all_triples, triple_to_idx, blocks, max_iters=2000):
    """
    Local search to reduce the number of blocks.
    1. Remove redundant blocks using coverage counts.
    2. Try to replace blocks with more efficient ones.
    """
    blocks = [list(b) for b in blocks]
    n_triples = len(all_triples)
    
    # Compute initial coverage counts
    cover_count = [0] * n_triples
    for b in blocks:
        for triple in itertools.combinations(b, t):
            cover_count[triple_to_idx[triple]] += 1
            
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            # Check if removing blocks[i] leaves any triple with count 0
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                idx = triple_to_idx[triple]
                if cover_count[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    cover_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break # Restart scan as indices shifted
                
    # Pass 2: Replace blocks to cover "hard" triples more efficiently
    # We try to find blocks that cover triples not covered by others (critical triples).
    random.seed(42)
    for _ in range(max_iters):
        idx = random.randint(0, len(blocks) - 1)
        
        # Find triples covered ONLY by blocks[idx]
        must_cover = []
        for triple in itertools.combinations(blocks[idx], t):
            t_idx = triple_to_idx[triple]
            if cover_count[t_idx] == 1:
                must_cover.append(triple)
                
        if not must_cover:
            # Block is redundant, remove it
            for triple in itertools.combinations(blocks[idx], t):
                t_idx = triple_to_idx[triple]
                cover_count[t_idx] -= 1
            blocks.pop(idx)
            continue
            
        # Try to construct a new block that covers must_cover
        # If must_cover is too large, skip
        if len(must_cover) > 10: 
            continue
            
        # Candidate points: union of points in must_cover
        candidate_points = set()
        for triple in must_cover:
            candidate_points.update(triple)
            
        if len(candidate_points) > k:
            continue
            
        new_block = list(candidate_points)
        remaining_must = set(must_cover)
        
        # Greedily add points to cover remaining must_cover
        while len(new_block) < k and remaining_must:
            best_point = -1
            best_gain = -1
            
            # Only consider points that are in some remaining_must triple or could form a triple with new_block
            # For simplicity, iterate all v, filter by validity
            for x in range(v):
                if x in new_block:
                    continue
                
                gain = 0
                for i in range(len(new_block)):
                    for j in range(i+1, len(new_block)):
                        triple = tuple(sorted([new_block[i], new_block[j], x]))
                        if triple in remaining_must:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point == -1 or best_gain == 0:
                break
            new_block.append(best_point)
            # Update remaining_must? 
            # For speed in this small loop, we just check against the original set at the end.
            
        new_block.sort()
        if len(new_block) == k and len(set(new_block)) == k:
            new_covered = set(itertools.combinations(new_block, t))
            if remaining_must.issubset(new_covered):
                # Replace blocks[idx] with new_block
                # Decrement counts for old block
                for triple in itertools.combinations(blocks[idx], t):
                    cover_count[triple_to_idx[triple]] -= 1
                # Increment counts for new block
                for triple in itertools.combinations(new_block, t):
                    cover_count[triple_to_idx[triple]] += 1
                blocks[idx] = new_block
                
    return blocks
