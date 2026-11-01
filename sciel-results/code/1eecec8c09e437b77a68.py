import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    num_triples = len(triple_set)
    
    best_blocks = None
    best_size = float('inf')
    
    def get_covered_triples(blocks):
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        return covered

    def greedy_init(seed):
        random.seed(seed)
        uncovered = set(all_triples)
        blocks = []
        points = list(range(v))
        
        while uncovered:
            # Pick a random uncovered triple to start
            start_triple = random.choice(list(uncovered))
            block = list(start_triple)
            
            while len(block) < k:
                best_point = None
                best_gain = -1
                candidates = [x for x in points if x not in block]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        tri = tuple(sorted(rest + (x,)))
                        if tri in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain == 0:
                    best_point = random.choice(candidates) if candidates else 0
                    if best_point in block:
                         for p in points:
                             if p not in block:
                                 best_point = p
                                 break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            for tri in itertools.combinations(block, t):
                uncovered.discard(tri)
                
        return blocks

    def local_search(blocks, max_iterations=5000, seed=0):
        random.seed(seed)
        blocks = [tuple(b) for b in blocks]
        current_size = len(blocks)
        
        count = {tri: 0 for tri in all_triples}
        for b in blocks:
            for tri in itertools.combinations(b, t):
                count[tri] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5:
                break
            
            # 1. Try to delete a block
            for i, b in enumerate(blocks):
                can_delete = True
                for tri in itertools.combinations(b, t):
                    if count[tri] == 1:
                        can_delete = False
                        break
                
                if can_delete:
                    for tri in itertools.combinations(b, t):
                        count[tri] -= 1
                    blocks.pop(i)
                    current_size -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try to swap a point in a block
            # Only consider swaps that keep the covering valid (no count becomes 0)
            # We prioritize swaps that might lead to a deletion.
            
            # Pick a block with critical triples
            b_idx = random.randint(0, len(blocks)-1)
            b = list(blocks[b_idx])
            
            # Find a point x in b such that all triples containing x in b have count > 1
            # This ensures we can remove x without uncovering anything
            valid_x_indices = []
            for idx, x in enumerate(b):
                safe = True
                for tri in itertools.combinations(b, t):
                    if x in tri and count[tri] == 1:
                        safe = False
                        break
                if safe:
                    valid_x_indices.append(idx)
            
            if not valid_x_indices:
                # No safe swap in this block, try another block or skip
                if random.random() < 0.1:
                    continue
                else:
                    # Try a different block
                    b_idx = random.randint(0, len(blocks)-1)
                    b = list(blocks[b_idx])
                    valid_x_indices = []
                    for idx, x in enumerate(b):
                        safe = True
                        for tri in itertools.combinations(b, t):
                            if x in tri and count[tri] == 1:
                                safe = False
                                break
                        if safe:
                            valid_x_indices.append(idx)
                    if not valid_x_indices:
                        continue

            x_idx = random.choice(valid_x_indices)
            x = b[x_idx]
            
            candidates = [y for y in range(v) if y not in b]
            if not candidates:
                continue
                
            y = random.choice(candidates)
            
            # Perform swap
            # Decrease counts for triples with x in old b
            for tri in itertools.combinations(b, t):
                if x in tri:
                    count[tri] -= 1
            # Increase counts for triples with y in new b
            new_b = b.copy()
            new_b[x_idx] = y
            new_b.sort()
            for tri in itertools.combinations(new_b, t):
                if y in tri:
                    count[tri] += 1
            
            blocks[b_idx] = tuple(new_b)
            improved = True # Restart deletion scan

        return blocks

    attempts = 0
    while time.time() < deadline:
        blocks = greedy_init(attempts)
        if len(blocks) < best_size:
            best_blocks = blocks
            best_size = len(blocks)
        
        # Local search
        blocks = local_search(blocks, max_iterations=2000, seed=attempts)
        
        if len(blocks) < best_size:
            best_blocks = blocks
            best_size = len(blocks)
            
        attempts += 1
        if best_size <= 76:
             break
             
    if best_blocks is None:
        best_blocks = greedy_init(0)
        
    return [list(b) for b in best_blocks]
