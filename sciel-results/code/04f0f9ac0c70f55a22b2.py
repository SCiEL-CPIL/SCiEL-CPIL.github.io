import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Allow up to 110 seconds to be safe (limit is 150s, keep margin)
    timeout = 110.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their covered triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(all_blocks):
        covered = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        block_triples.append(covered)
        for tr_idx in covered:
            triple_to_blocks[tr_idx].append(i)
    
    # Precompute triple degree (number of blocks covering each triple)
    triple_degree = [len(triple_to_blocks[j]) for j in range(num_triples)]
    
    # Bitmask representation: each block's covered triples as a Python int bitmask
    block_bitmasks = []
    for tr_list in block_triples:
        mask = 0
        for tr_idx in tr_list:
            mask |= (1 << tr_idx)
        block_bitmasks.append(mask)
    
    # Static spectral-like score: weight triples by inverse degree
    # This is fast and captures the "rarity" of triples
    block_static_score = np.zeros(num_blocks, dtype=np.float64)
    for i in range(num_blocks):
        score = 0.0
        for tr_idx in block_triples[i]:
            score += 1.0 / triple_degree[tr_idx]
        block_static_score[i] = score
        
    # Sort blocks by static score for tie-breaking
    static_order = np.argsort(-block_static_score)
    static_rank = np.zeros(num_blocks, dtype=np.int64)
    for rank, bi in enumerate(static_order):
        static_rank[bi] = rank
        
    # Full mask for all triples
    full_mask = (1 << num_triples) - 1

    def greedy_cover_from(start_blocks, random_seed=None):
        """
        Greedy cover starting from a set of blocks.
        start_blocks: list of block indices already selected.
        Returns list of block indices.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        current_blocks = list(start_blocks)
        
        # Calculate current coverage
        coverage_mask = 0
        for bi in current_blocks:
            coverage_mask |= block_bitmasks[bi]
            
        uncovered_mask = full_mask & ~coverage_mask
        
        while uncovered_mask:
            # Find uncovered triples
            # We pick a random uncovered triple to drive the greedy choice
            # To be fast, we pick the first few bits? No, random is better for diversity.
            # But iterating all bits is slow.
            # We can pick a random triple index and check if it's uncovered, retry if not.
            # Since coverage is dense, most triples are covered.
            # Better: maintain a list of uncovered indices? 
            # Or: just pick a random triple from the whole set, if covered, pick another.
            # With 2024 triples, and ~76 blocks covering 35 each, 
            # coverage is ~2660 triple-slots. Total 2024. 
            # So almost all are covered. 
            # We need to find the *uncovered* ones.
            
            # Efficient way: 
            # 1. Get list of set bits in uncovered_mask.
            # 2. Pick one at random.
            
            uncovered_indices = []
            temp = uncovered_mask
            idx = 0
            while temp:
                if temp & 1:
                    uncovered_indices.append(idx)
                temp >>= 1
                idx += 1
            
            if not uncovered_indices:
                break
                
            # Weighted random selection: prefer triples with lower degree (rarer)
            # This helps break ties and focus on hard parts
            weights = [1.0 / triple_degree[tr] for tr in uncovered_indices]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            target_triple_idx = uncovered_indices[0]
            for tr, w in zip(uncovered_indices, weights):
                cum += w
                if r <= cum:
                    target_triple_idx = tr
                    break
            
            # Get candidate blocks covering this triple
            candidates = triple_to_blocks[target_triple_idx]
            
            best_gain = -1
            best_block_idx = -1
            best_rank = float('inf')
            
            # Evaluate gain
            for bi in candidates:
                # Skip if already in current_blocks? 
                # If we add a block already in the set, gain is 0 for new triples.
                # But we only call this if uncovered_mask != 0.
                # If bi is in current_blocks, its triples are already in coverage_mask, 
                # so they are NOT in uncovered_mask. Thus gain will be 0.
                # So we can just evaluate all.
                
                gain = (block_bitmasks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_rank = static_rank[bi]
                    if best_gain == 35: # Max possible
                        break
                elif gain == best_gain and gain > 0:
                    if static_rank[bi] < best_rank:
                        best_block_idx = bi
                        best_rank = static_rank[bi]
            
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                coverage_mask |= block_bitmasks[best_block_idx]
                uncovered_mask = full_mask & ~coverage_mask
            else:
                # Should not happen if problem is solvable
                break
                
        return current_blocks

    def peel(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return block_indices
            
        counts = [0] * num_triples
        for bi in block_indices:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
        
        new_indices = []
        for bi in block_indices:
            removable = True
            for tr_idx in block_triples[bi]:
                if counts[tr_idx] == 1:
                    removable = False
                    break
            if not removable:
                new_indices.append(bi)
            else:
                for tr_idx in block_triples[bi]:
                    counts[tr_idx] -= 1
                    
        return new_indices

    def local_search_remove(block_indices):
        """Iterative removal of redundant blocks."""
        if not block_indices:
            return block_indices
        
        coverage_mask = 0
        for bi in block_indices:
            coverage_mask |= block_bitmasks[bi]
        
        improved = True
        while improved:
            improved = False
            for i, bi in enumerate(block_indices):
                without_bi = coverage_mask & ~block_bitmasks[bi]
                if (without_bi & block_bitmasks[bi]) == block_bitmasks[bi]:
                    coverage_mask = without_bi
                    block_indices.pop(i)
                    improved = True
                    break
        return block_indices

    best_block_indices = None
    best_count = float('inf')
    attempts = 0
    
    while time.time() - start_time < timeout:
        seed = attempts
        
        if best_block_indices is None or attempts % 2 == 0:
            # Fresh restart
            current = greedy_cover_from([], seed)
        else:
            # Perturb and Repair
            # Remove 1 or 2 random blocks from best
            if len(best_block_indices) > 2:
                k_remove = random.randint(1, 2)
                remove_set = set(random.sample(best_block_indices, k_remove))
                start_set = [b for b in best_block_indices if b not in remove_set]
                current = greedy_cover_from(start_set, seed)
            else:
                current = greedy_cover_from([], seed)
                
        current = peel(current)
        current = local_search_remove(current)
        
        if len(current) < best_count:
            best_count = len(current)
            best_block_indices = list(current)
            
        attempts += 1
        
        if time.time() - start_time > timeout - 8.0:
            break

    if best_block_indices is None:
        best_block_indices = greedy_cover_from([], 0)
        best_block_indices = peel(best_block_indices)
        
    return [list(all_blocks[bi]) for bi in best_block_indices]
