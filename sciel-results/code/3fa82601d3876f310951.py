import itertools
import random
import time
import numpy as np

# Global cache for precomputed data
_PRECOMPUTED = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
    
    # Map each t-subset to an index
    triple_to_idx = {}
    triples = list(itertools.combinations(range(v), t))
    for i, triple in enumerate(triples):
        triple_to_idx[triple] = i
        
    # Map each k-subset (block) to a list of triple indices
    block_to_triples = []
    blocks = list(itertools.combinations(range(v), k))
    for block in blocks:
        t_indices = []
        for sub in itertools.combinations(block, t):
            t_indices.append(triple_to_idx[sub])
        block_to_triples.append(t_indices)
        
    _PRECOMPUTED[(v, k, t)] = {
        'blocks': blocks,
        'block_to_triples': block_to_triples,
        'num_blocks': len(blocks),
        'num_triples': len(triples)
    }
    return _PRECOMPUTED[(v, k, t)]

def _greedy_construct(v, k, t):
    """
    Fast greedy construction to get a valid initial covering.
    Returns a list of block indices.
    """
    data = _precompute(v, k, t)
    num_triples = data['num_triples']
    block_to_triples = data['block_to_triples']
    
    # Inverse mapping for greedy
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx, t_indices in enumerate(block_to_triples):
        for t_idx in t_indices:
            triple_to_blocks[t_idx].append(b_idx)
            
    uncovered = set(range(num_triples))
    active_blocks = []
    
    while uncovered:
        # Pick an uncovered triple
        target_triple = next(iter(uncovered))
        
        # Find a block covering this triple that covers the most uncovered triples
        best_block = -1
        max_gain = -1
        
        candidates = triple_to_blocks[target_triple]
        
        for b_idx in candidates:
            gain = 0
            for t_idx in block_to_triples[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
                
        active_blocks.append(best_block)
        # Update uncovered
        for t_idx in block_to_triples[best_block]:
            uncovered.discard(t_idx)
            
    return active_blocks

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for safety
    
    if (v, k, t) != (24, 7, 3):
        return _simple_greedy(v, k, t)
        
    data = _precompute(v, k, t)
    num_blocks = data['num_blocks']
    block_to_triples = data['block_to_triples']
    blocks = data['blocks']
    num_triples = data['num_triples']
    
    # 1. Initial Solution
    initial_blocks = _greedy_construct(v, k, t)
    
    # State
    active = set(initial_blocks)
    cover_counts = np.zeros(num_triples, dtype=np.int32)
    for b_idx in active:
        for t_idx in block_to_triples[b_idx]:
            cover_counts[t_idx] += 1
            
    best_blocks = list(active)
    best_len = len(active)
    
    # 2. Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.0
            
        # Propose move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a block
            if not active:
                continue
            b_idx = random.choice(list(active))
            
            # Calculate Delta
            delta_blocks = -1
            new_uncovered = 0
            for t_idx in block_to_triples[b_idx]:
                if cover_counts[t_idx] == 1:
                    new_uncovered += 1
            delta_uncovered = new_uncovered
            delta_energy = delta_blocks + 100 * delta_uncovered
            
            if delta_energy < 0 or (delta_energy > 0 and random.random() < np.exp(-delta_energy / T)):
                # Accept
                active.remove(b_idx)
                for t_idx in block_to_triples[b_idx]:
                    cover_counts[t_idx] -= 1
                if len(active) < best_len and np.sum(cover_counts == 0) == 0:
                    best_len = len(active)
                    best_blocks = list(active)
        elif move_type < 0.6:
            # Add a block
            b_idx = random.randint(0, num_blocks - 1)
            if b_idx in active:
                continue
                
            delta_blocks = 1
            new_covered = 0
            for t_idx in block_to_triples[b_idx]:
                if cover_counts[t_idx] == 0:
                    new_covered += 1
            delta_uncovered = -new_covered
            delta_energy = delta_blocks + 100 * delta_uncovered
            
            if delta_energy < 0 or (delta_energy > 0 and random.random() < np.exp(-delta_energy / T)):
                # Accept
                active.add(b_idx)
                for t_idx in block_to_triples[b_idx]:
                    cover_counts[t_idx] += 1
                if len(active) < best_len and np.sum(cover_counts == 0) == 0:
                    best_len = len(active)
                    best_blocks = list(active)
        else:
            # Swap: Remove A, Add B
            if not active:
                continue
            b_idx_remove = random.choice(list(active))
            b_idx_add = random.randint(0, num_blocks - 1)
            if b_idx_add == b_idx_remove:
                continue
            if b_idx_add in active:
                continue
                
            old_counts = cover_counts.copy()
            
            for t_idx in block_to_triples[b_idx_remove]:
                cover_counts[t_idx] -= 1
            for t_idx in block_to_triples[b_idx_add]:
                cover_counts[t_idx] += 1
                
            new_uncovered = np.sum(cover_counts == 0)
            old_uncovered = np.sum(old_counts == 0)
            
            delta_blocks = 0
            delta_uncovered = new_uncovered - old_uncovered
            delta_energy = delta_blocks + 100 * delta_uncovered
            
            if delta_energy < 0 or (delta_energy > 0 and random.random() < np.exp(-delta_energy / T)):
                # Accept
                active.remove(b_idx_remove)
                active.add(b_idx_add)
                # cover_counts is already updated
                if len(active) < best_len and new_uncovered == 0:
                    best_len = len(active)
                    best_blocks = list(active)
            else:
                # Reject
                cover_counts = old_counts
                
        T *= cooling_rate
        
    # Final cleanup: try to delete redundant blocks from best_blocks
    final_active = set(best_blocks)
    final_counts = np.zeros(num_triples, dtype=np.int32)
    for b_idx in final_active:
        for t_idx in block_to_triples[b_idx]:
            final_counts[t_idx] += 1
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for b_idx in list(final_active):
            redundant = True
            for t_idx in block_to_triples[b_idx]:
                if final_counts[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                final_active.remove(b_idx)
                for t_idx in block_to_triples[b_idx]:
                    final_counts[t_idx] -= 1
                changed = True
                
    result = [blocks[i] for i in final_active]
    return result
