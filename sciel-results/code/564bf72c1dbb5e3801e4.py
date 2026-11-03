"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy with State Pruning."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Precompute all blocks and their triple coverage.
    2. Group triples by the number of blocks that cover them (scarcity).
    3. Greedy selection prioritizing blocks that cover the most 'scarce' (low frequency) 
       uncovered triples. This maximizes entropy reduction by resolving rare states first.
    4. Local search to optimize the final solution.
    """
    # Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    
    # Map each block to the set of triple indices it covers
    block_to_triples = {}
    for b in all_blocks:
        covered = set()
        for tr in itertools.combinations(b, t):
            covered.add(triple_to_idx[tr])
        block_to_triples[b] = covered
        
    # Calculate frequency of each triple (how many blocks cover it)
    triple_freq = [0] * num_triples
    for b, covered in block_to_triples.items():
        for idx in covered:
            triple_freq[idx] += 1
            
    # Sort triples by frequency (scarce first)
    # We will use this to weight the gain of a block
    # Weight of a triple = 1 / (freq + 1)
    triple_weights = [1.0 / (f + 1.0) for f in triple_freq]
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy_pruned(v, k, t, all_blocks, block_to_triples, triple_weights, num_triples)
    
    # Phase 2: Local Search Optimization
    blocks = _local_search(v, k, t, all_triples, triple_to_idx, block_to_triples, blocks)
    
    return blocks

def _info_greedy_pruned(v, k, t, all_blocks, block_to_triples, triple_weights, num_triples):
    """
    Greedy covering that maximizes entropy reduction by prioritizing scarce triples.
    We maintain a set of uncovered triple indices.
    For each step, we evaluate all blocks to find the one that covers the most
    'weighted' uncovered triples.
    """
    uncovered = set(range(num_triples))
    blocks = []
    
    # For speed, we can pre-sort blocks or just iterate. 
    # With ~200k blocks and ~2000 steps, iterating all blocks each step is O(2000*200k*7) ~ 30M ops, which is fine in Python if optimized.
    # However, we can prune blocks that are subsets of already selected blocks? No, blocks are same size.
    # We can skip blocks that cover no new triples.
    
    # Optimization: Instead of checking all blocks every time, we can maintain a list of "active" blocks.
    # But since the gain changes as uncovered changes, we must re-evaluate.
    
    # To speed up gain calculation:
    # gain(block) = sum(triple_weights[i] for i in block_to_triples[block] if i in uncovered)
    
    while uncovered:
        best_block = None
        best_gain = -1
        
        # Iterate through all blocks
        for b in all_blocks:
            covered = block_to_triples[b]
            # Calculate gain
            gain = 0.0
            for idx in covered:
                if idx in uncovered:
                    gain += triple_weights[idx]
                    
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is None or best_gain == 0:
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        for idx in block_to_triples[best_block]:
            uncovered.discard(idx)
            
    return blocks

def _local_search(v, k, t, all_triples, triple_to_idx, block_to_triples, blocks, max_iters=500):
    """
    Local search to reduce the number of blocks.
    1. Remove redundant blocks.
    2. Try to replace blocks with more efficient ones.
    """
    blocks = list(blocks)
    
    # Precompute coverage sets for current blocks
    # We need to check if removing a block leaves any triple uncovered.
    
    def get_coverage(block_list):
        cov = [0] * len(all_triples)
        for b in block_list:
            for idx in block_to_triples[tuple(b)]:
                cov[idx] += 1
        return cov
        
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        cov = get_coverage(blocks)
        for i in range(len(blocks)):
            # If any triple has cov 1 and is covered by blocks[i], it's not redundant
            # Check if removing blocks[i] leaves any triple with cov 0
            is_redundant = True
            for idx in block_to_triples[tuple(blocks[i])]:
                if cov[idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                blocks.pop(i)
                changed = True
                break # Restart loop as coverage changed
                
    # Pass 2: Local search swap
    random.seed(42)
    for _ in range(max_iters):
        if not blocks:
            break
        idx = random.randint(0, len(blocks) - 1)
        b_to_remove = blocks[idx]
        
        # Find triples covered ONLY by b_to_remove
        cov = get_coverage(blocks)
        must_cover = []
        for tr_idx in block_to_triples[tuple(b_to_remove)]:
            if cov[tr_idx] == 1:
                must_cover.append(tr_idx)
                
        if not must_cover:
            blocks.pop(idx)
            continue
            
        # Try to find a new block that covers all must_cover and adds some extra coverage
        # Filter must_cover triples to find candidate points
        candidate_points = set()
        for tr_idx in must_cover:
            candidate_points.update(all_triples[tr_idx])
            
        if len(candidate_points) > k:
            continue
            
        # Greedily build a block covering must_cover
        new_block_set = set(candidate_points)
        remaining_must = set(must_cover)
        
        while len(new_block_set) < k and remaining_must:
            best_point = None
            best_gain = -1
            for p in range(v):
                if p in new_block_set:
                    continue
                # Check how many remaining_must triples are covered by adding p
                # A triple is covered if its other 2 points are in new_block_set
                gain = 0
                for tr_idx in remaining_must:
                    tr = all_triples[tr_idx]
                    if p in tr:
                        others = [x for x in tr if x != p]
                        if all(x in new_block_set for x in others):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point is None or best_gain == 0:
                break
            new_block_set.add(best_point)
            
        if len(new_block_set) == k:
            new_block = tuple(sorted(new_block_set))
            # Check if new block covers all must_cover
            new_cov = block_to_triples.get(new_block, set())
            if set(must_cover).issubset(new_cov):
                # Check if this swap is beneficial (covers more total triples or same count)
                # For now, just accept if it covers must_cover
                blocks[idx] = list(new_block)
                
    return blocks
