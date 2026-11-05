import numpy as np
import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # 1. Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)
    
    # 2. Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Precompute block_indices: shape (num_blocks, C(k, t))
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
    
    # 4. Precompute frequency and inverse frequency
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    inv_freq = 1.0 / (freq.astype(np.float64) + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multi-strategy greedy restarts
    seed = 0
    while time.time() < deadline - 30.0:  # Leave 30s for local search
        strategy = seed % 3
        
        if strategy == 0:
            blocks = _greedy_max_gain(all_blocks, block_indices, num_tsubsets, deadline)
        elif strategy == 1:
            blocks = _greedy_weighted(all_blocks, block_indices, num_tsubsets, deadline, inv_freq)
        else:
            blocks = _greedy_entropy(all_blocks, block_indices, num_tsubsets, deadline, seed)
        
        if blocks is None:
            seed += 1
            continue
        
        blocks = _remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        seed += 1
        if best_len <= 73:
            break
    
    # Phase 2: Local search on best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        best_blocks = _local_search(best_blocks, all_blocks, deadline)
    
    if best_blocks is None:
        # Fallback
        best_blocks = _greedy_max_gain(all_blocks, block_indices, num_tsubsets, deadline)
        if best_blocks is None:
            best_blocks = []
    
    return best_blocks


def _greedy_max_gain(all_blocks, block_indices, num_tsubsets, deadline):
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
        gains = uncovered[block_indices].sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered[block_indices[best_idx]] = False
    
    if np.any(uncovered):
        return None
    return [list(all_blocks[i]) for i in selected_indices]


def _greedy_weighted(all_blocks, block_indices, num_tsubsets, deadline, inv_freq):
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
        w = np.where(uncovered, inv_freq, 0.0)
        gains = w[block_indices].sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered[block_indices[best_idx]] = False
    
    if np.any(uncovered):
        return None
    return [list(all_blocks[i]) for i in selected_indices]


def _greedy_entropy(all_blocks, block_indices, num_tsubsets, deadline, seed):
    """
    Entropy-inspired greedy: dynamically weight triples based on how often they've
    been covered so far, favoring blocks that cover under-covered triples.
    """
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    cover_count = np.zeros(num_tsubsets, dtype=np.float64)
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
        
        gains = uncovered[block_indices].sum(axis=1)
        
        # Dynamic weights: 1/(cover_count + 1) for uncovered triples
        w = np.zeros(num_tsubsets, dtype=np.float64)
        w[uncovered] = 1.0 / (cover_count[uncovered] + 1.0)
        diversity = w[block_indices].sum(axis=1)
        
        max_div = diversity.max() if diversity.max() > 0 else 1.0
        scores = gains * 0.8 + (diversity / max_div) * 0.2
        
        best_idx = np.argmax(scores)
        if gains[best_idx] == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered[block_indices[best_idx]] = False
        cover_count[block_indices[best_idx]] += 1.0
    
    if np.any(uncovered):
        return None
    return [list(all_blocks[i]) for i in selected_indices]


def _remove_redundant(blocks):
    if not blocks or len(blocks) <= 1:
        return blocks
    
    # Build triple coverage count
    triple_count = {}
    for b in blocks:
        for tr in itertools.combinations(b, 3):
            triple_count[tr] = triple_count.get(tr, 0) + 1
    
    kept = []
    for b in blocks:
        redundant = True
        for tr in itertools.combinations(b, 3):
            if triple_count[tr] == 1:
                redundant = False
                break
        if not redundant:
            kept.append(b)
    
    return kept


def _local_search(blocks, all_blocks, deadline):
    if not blocks:
        return blocks
    
    current = list(blocks)
    
    def build_coverage(block_list):
        tc = {}
        for b in block_list:
            for tr in itertools.combinations(b, 3):
                tc[tr] = tc.get(tr, 0) + 1
        return tc
    
    # Iteratively remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        tc = build_coverage(current)
        for i in range(len(current)):
            redundant = True
            for tr in itertools.combinations(current[i], 3):
                if tc[tr] == 1:
                    redundant = False
                    break
            if redundant:
                del current[i]
                changed = True
                break
    
    # Try swaps
    rng = random.Random(123)
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(current) <= 72:
            break
        
        i = rng.randrange(len(current))
        j = rng.randrange(len(all_blocks))
        candidate = list(all_blocks[j])
        if candidate in current:
            continue
        
        new_current = current[:i] + [candidate] + current[i+1:]
        tc = build_coverage(new_current)
        
        # Check validity
        valid = all(cnt >= 1 for cnt in tc.values())
        if not valid:
            continue
        
        # Try to remove redundant
        changed2 = True
        while changed2 and time.time() < deadline:
            changed2 = False
            tc = build_coverage(new_current)
            for idx in range(len(new_current)):
                redundant = True
                for tr in itertools.combinations(new_current[idx], 3):
                    if tc[tr] == 1:
                        redundant = False
                        break
                if redundant:
                    del new_current[idx]
                    changed2 = True
                    break
        
        if len(new_current) < len(current):
            current = new_current
    
    return current
