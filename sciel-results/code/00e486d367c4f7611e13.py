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
    
    # Also precompute the list of triple indices for each block (for local search)
    block_triple_indices = []
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tris = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
            tris.append(t_idx)
        block_masks[b_idx] = mask
        block_triple_indices.append(tris)
        
    _PRECOMPUTED[(v, k, t)] = {
        'all_blocks': all_blocks,
        'block_masks': block_masks,
        'blocks_by_triple': blocks_by_triple,
        'block_triple_indices': block_triple_indices,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx
    }
    return _PRECOMPUTED[(v, k, t)]

def _weighted_greedy(v, k, t, data, seed):
    """
    Weighted greedy: assign random weights to points, then iteratively select
    the block that maximizes the weighted sum of newly covered triples.
    This escapes deterministic local minima.
    """
    rng = random.Random(seed)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    # Assign random weights to points
    point_weights = [rng.random() for _ in range(v)]
    
    # Compute triple weights: weight of a triple = sum of its point weights
    triple_weights = np.zeros(num_triples)
    all_triples_list = list(itertools.combinations(range(v), t))
    for i, tri in enumerate(all_triples_list):
        triple_weights[i] = sum(point_weights[p] for p in tri)
    
    uncovered_mask = (1 << num_triples) - 1
    selected_blocks = []
    selected_indices = set()
    
    while uncovered_mask:
        # Find uncovered triple indices (sample a few to keep it fast)
        # Instead of scanning all, sample candidates
        # Pick a random uncovered triple to get candidate blocks
        while True:
            t_idx = rng.randint(0, num_triples - 1)
            if (uncovered_mask >> t_idx) & 1:
                break
        
        candidates = blocks_by_triple[t_idx]
        
        best_score = -1
        best_block_idx = -1
        
        # Evaluate candidates using weighted score
        for b_idx in candidates:
            if b_idx in selected_indices:
                continue
            mask = block_masks[b_idx]
            # Compute weighted gain: sum of triple_weights for newly covered triples
            # We need to find which triples in this block are still uncovered
            # Use bit operations to find the intersection
            new_covered = mask & uncovered_mask
            if new_covered == 0:
                continue
            # Compute weighted sum of newly covered triples
            # This is expensive in pure Python, so we approximate by counting
            # and using a random sample of the new triples
            # For speed, use bit_count as a proxy and add randomness
            gain_count = new_covered.bit_count()
            # Add a small random perturbation to break ties
            score = gain_count + rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            # Fallback: pick any candidate not selected
            for b_idx in candidates:
                if b_idx not in selected_indices:
                    best_block_idx = b_idx
                    break
            if best_block_idx == -1:
                break  # Stuck
        
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
        if max_iters > 300:
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
            for _ in range(50):
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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Pre-compute
    data = _precompute_bitsets(v, k, t)
    triple_to_idx = data['triple_to_idx']
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # Ensure we have enough time for greedy + reduction + local search
        if time.time() + 3.0 > deadline:
            break
            
        # Use weighted greedy with different seeds
        blocks = _weighted_greedy(v, k, t, data, seed)
        
        # Apply reduction
        if time.time() + 1.0 < deadline:
            blocks = _reduce_blocks(blocks, v, t, triple_to_idx)
            
        # Try swap-based local search if we have more time
        if time.time() + 15.0 < deadline:
            blocks = _local_search_swap(v, k, t, blocks, triple_to_idx, deadline)
            
        # Final reduction after local search
        if time.time() + 0.5 < deadline:
            blocks = _reduce_blocks(blocks, v, t, triple_to_idx)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if best_len <= 76:
            # We've matched or beaten the known best, try a few more seeds
            if best_len == 69:
                break
            seed += 1
            continue
            
        seed += 1
        
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _weighted_greedy(v, k, t, data, 0)
        best_blocks = _reduce_blocks(best_blocks, v, t, triple_to_idx)
        
    return best_blocks
