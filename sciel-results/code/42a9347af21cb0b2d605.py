import itertools
import time
import random

def construct(v, k, t):
    if v == 24 and k == 7 and t == 3:
        return _construct_24_7_3()
    
    # Fallback
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    start_time = time.time()
    while uncovered:
        if time.time() - start_time > 100:
            break
        pivot = random.choice(list(uncovered))
        block = list(pivot)
        while len(block) < k:
            best_point = -1
            max_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_point = x
            block.append(best_point)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

def _construct_24_7_3():
    start_time = time.time()
    deadline = 110.0
    
    # 1. Pre-computation
    triples = list(itertools.combinations(range(24), 3))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    blocks_list = []
    block_masks = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(range(24), 7):
        mask = 0
        block_idx = len(blocks_list)
        
        # Compute mask and update reverse map
        for i in range(7):
            for j in range(i+1, 7):
                for l in range(j+1, 7):
                    triple = (block[i], block[j], block[l])
                    idx = triple_to_idx[triple]
                    mask |= 1 << idx
                    triple_to_blocks[idx].append(block_idx)
                    
        blocks_list.append(block)
        block_masks.append(mask)
        
    best_blocks = None
    best_count = float('inf')
    
    # 2. Randomized Greedy
    while time.time() - start_time < deadline:
        uncovered_mask = (1 << num_triples) - 1
        selected_indices = []
        
        while uncovered_mask:
            # Select pivot
            # If uncovered_mask is dense, random guess is fast.
            # If sparse, we might loop. 
            # To be safe, if we fail 100 times, do linear scan.
            pivot_idx = -1
            attempts = 0
            while True:
                idx = random.randint(0, num_triples - 1)
                if (uncovered_mask >> idx) & 1:
                    pivot_idx = idx
                    break
                attempts += 1
                if attempts > 100:
                    # Linear scan fallback
                    for i in range(num_triples):
                        if (uncovered_mask >> i) & 1:
                            pivot_idx = i
                            break
                    break
            
            # Find best block
            best_b_idx = -1
            max_gain = -1
            
            for b_idx in triple_to_blocks[pivot_idx]:
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_b_idx = b_idx
            
            selected_indices.append(best_b_idx)
            uncovered_mask &= ~block_masks[best_b_idx]
            
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_blocks = selected_indices[:]
            
    # 3. Return result
    return [list(blocks_list[i]) for i in best_blocks]
