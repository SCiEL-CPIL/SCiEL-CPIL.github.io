import itertools
import time
import random
from collections import defaultdict
import numpy as np

# Global cache for precomputation
_PRECOMPUTED = {}

def _precompute_bitsets(v, k, t):
    """
    Precompute bitsets for all k-blocks and mappings for fast greedy.
    v=24, k=7, t=3 -> 2024 triples, 346104 blocks.
    """
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
        
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute mask for each block
    # Mask is an integer where bit i is 1 if the block covers triple i
    block_masks = [0] * num_blocks
    # Map triple index to list of block indices that cover it
    blocks_by_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
        block_masks[b_idx] = mask
        
    _PRECOMPUTED[(v, k, t)] = {
        'all_blocks': all_blocks,
        'block_masks': block_masks,
        'blocks_by_triple': blocks_by_triple,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx
    }
    return _PRECOMPUTED[(v, k, t)]

def _greedy_bitset(v, k, t, data, seed):
    """
    Greedy set cover using bitsets.
    1. Pick random uncovered triple.
    2. Among blocks covering it, pick one with max gain (new triples covered).
    3. Update uncovered mask.
    """
    rng = random.Random(seed)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    uncovered_mask = (1 << num_triples) - 1
    selected_blocks = []
    selected_indices = set()
    
    while uncovered_mask:
        # Find an uncovered triple
        # Instead of picking a random bit from the mask (which is hard for large ints in pure python without bit ops tricks),
        # we can pick a random triple index and check if it's uncovered.
        # Since coverage is dense, this might require a few retries.
        # Alternatively, we can iterate through blocks_by_triple for a random triple?
        # Let's just pick a random triple index.
        
        while True:
            t_idx = rng.randint(0, num_triples - 1)
            if (uncovered_mask >> t_idx) & 1:
                break
        
        candidates = blocks_by_triple[t_idx]
        
        best_gain = -1
        best_block_idx = -1
        
        # Evaluate candidates
        # We can optimize this by only checking blocks not already selected?
        # Actually, if a block is already selected, its mask is 0 in uncovered_mask, so gain is 0.
        # But we don't want to select the same block twice.
        # The problem statement says "blocks must be distinct".
        # So we should avoid selecting already selected blocks.
        
        for b_idx in candidates:
            if b_idx in selected_indices:
                continue
            mask = block_masks[b_idx]
            gain = (mask & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            # Should not happen if covering is possible, but fallback
            # Pick any candidate not selected
            for b_idx in candidates:
                if b_idx not in selected_indices:
                    best_block_idx = b_idx
                    break
            if best_block_idx == -1:
                break # Stuck?

        selected_blocks.append(all_blocks[best_block_idx])
        selected_indices.add(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return selected_blocks

def _reduce_blocks(blocks, v, t, triple_to_idx):
    """
    Remove redundant blocks.
    A block is redundant if every triple it covers is also covered by another selected block.
    """
    if not blocks:
        return []
        
    n_blocks = len(blocks)
    active = [True] * n_blocks
    
    # Map triple to list of active block indices covering it
    # We need to update this as we remove blocks.
    # To keep it fast, we can just re-scan or use counts.
    # Let's use counts.
    
    # Precompute triples for each block
    block_triples = []
    for b in blocks:
        tris = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
        
    cov_count = [0] * len(triple_to_idx)
    for tris in block_triples:
        for ti in tris:
            cov_count[ti] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(n_blocks):
            if not active[i]:
                continue
            redundant = True
            for ti in block_triples[i]:
                if cov_count[ti] == 1:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True
                for ti in block_triples[i]:
                    cov_count[ti] -= 1
                # Note: We don't remove from the list yet, just mark inactive.
                # We need to be careful: if we mark i inactive, cov_count for its triples decreases.
                # Other blocks might become redundant.
                
    return [blocks[i] for i in range(n_blocks) if active[i]]

def _local_search_swap(v, k, t, blocks, triple_to_idx, deadline):
    """
    Local search with point swaps to create redundancy, enabling block removals.
    """
    blocks = [list(b) for b in blocks]
    n_triples = len(triple_to_idx)
    rng = random.Random(999)
    
    # Build coverage count array
    cov_count = [0] * n_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
            
    improved = True
    max_iters = 0
    
    while improved and time.time() < deadline:
        improved = False
        max_iters += 1
        if max_iters > 200:
            break
        
        # Try to remove blocks first
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() >= deadline:
                break
            b = blocks[i]
            block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
            if all(cov_count[ti] > 1 for ti in block_triples):
                for ti in block_triples:
                    cov_count[ti] -= 1
                blocks.pop(i)
                improved = True
        
        if not improved:
            # Try point swaps to create redundancy
            for _ in range(20):
                if time.time() >= deadline:
                    break
                
                if len(blocks) == 0:
                    break
                    
                idx = rng.randint(0, len(blocks) - 1)
                old_x = rng.choice(blocks[idx])
                new_x = rng.randint(0, v - 1)
                
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                
                # Compute the effect of the swap
                old_block = blocks[idx]
                new_block = sorted([new_x] + [p for p in old_block if p != old_x])
                
                old_tris = set(itertools.combinations(old_block, t))
                new_tris = set(itertools.combinations(new_block, t))
                
                # Check if swap is valid (no triple becomes uncovered)
                valid = True
                for tri in old_tris:
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] <= 1 and tri not in new_tris:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Apply swap
                for tri in old_tris:
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] -= 1
                for tri in new_tris:
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] += 1
                blocks[idx] = new_block
                improved = True
                break
    
    # Final cleanup pass
    for i in range(len(blocks) - 1, -1, -1):
        if time.time() >= deadline:
            break
        b = blocks[i]
        block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
        if all(cov_count[ti] > 1 for ti in block_triples):
            for ti in block_triples:
                cov_count[ti] -= 1
            blocks.pop(i)
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Pre-compute
    data = _precompute_bitsets(v, k, t)
    triple_to_idx = data['triple_to_idx']
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # Ensure we have enough time for greedy + reduction + local search
        if time.time() + 2.0 > deadline:
            break
            
        blocks = _greedy_bitset(v, k, t, data, seed)
        
        # Apply reduction
        if time.time() + 1.0 < deadline:
            blocks = _reduce_blocks(blocks, v, t, triple_to_idx)
            
        # Try swap-based local search if we have more time
        if time.time() + 10.0 < deadline:
            blocks = _local_search_swap(v, k, t, blocks, triple_to_idx, deadline)
            
        # Final reduction after local search
        if time.time() + 0.5 < deadline:
            blocks = _reduce_blocks(blocks, v, t, triple_to_idx)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if best_len == 69:
            break
            
        seed += 1
        
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_bitset(v, k, t, data, 0)
        best_blocks = _reduce_blocks(best_blocks, v, t, triple_to_idx)
        
    return best_blocks
