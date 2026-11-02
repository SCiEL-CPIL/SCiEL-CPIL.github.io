import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Precompute for each point, the list of (pair, triple_idx) for fast gain calc
    # For a given pair (a,b) and point x, the triple is (x,a,b) sorted
    # We need: for each x, and for each pair in current block, compute gain
    
    def random_greedy_cover():
        # uncovered[i] is 1 if triple i is uncovered
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        while True:
            # Check if all covered
            if not np.any(uncovered):
                break
            
            # Pick a random uncovered triple
            seed_idx = None
            while True:
                candidate = random.randint(0, num_triples - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_points = []
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for pair in block_pairs:
                        triple = tuple(sorted((x,) + pair))
                        idx = triple_to_idx[triple]
                        if uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    block.append(random.choice(best_points))
                else:
                    available = [x for x in all_points if x not in block_set]
                    if available:
                        block.append(random.choice(available))
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        
        return blocks

    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # triple_count[idx] = number of blocks covering triple idx
        triple_count = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                # Update counts
                for i in to_remove:
                    block = blocks[i]
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                
                blocks = new_blocks
                changed = True
                
        return blocks

    def local_search_swap(blocks):
        """Try to swap points in blocks to enable more deletions."""
        if not blocks:
            return blocks
        
        triple_count = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        improved = True
        while improved:
            improved = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    return blocks
                block = blocks[idx]
                # Try swapping each point in the block
                for pos in range(len(block)):
                    old_x = block[pos]
                    other_points = [p for i, p in enumerate(block) if i != pos]
                    
                    # Try a few random new points
                    for _ in range(10):
                        new_x = random.randint(0, v - 1)
                        if new_x == old_x or new_x in other_points:
                            continue
                        
                        # Check if swap is valid (no triple loses coverage)
                        valid = True
                        old_triples = set()
                        new_triples = set()
                        
                        for triple in itertools.combinations(block, t):
                            old_triples.add(triple)
                        
                        new_block = sorted([new_x] + other_points)
                        for triple in itertools.combinations(new_block, t):
                            new_triples.add(triple)
                        
                        # Check all old triples that are not in new_block
                        for triple in old_triples:
                            if triple not in new_triples:
                                idx_t = triple_to_idx[tuple(triple)]
                                if triple_count[idx_t] <= 1:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        # Apply swap
                        for triple in old_triples:
                            idx_t = triple_to_idx[tuple(triple)]
                            triple_count[idx_t] -= 1
                        for triple in new_triples:
                            idx_t = triple_to_idx[tuple(triple)]
                            triple_count[idx_t] += 1
                        
                        blocks[idx] = new_block
                        improved = True
                        break  # Move to next block after one successful swap
        
        return blocks

    def affine_initial_pool():
        """Generate blocks from affine geometry structure for v=24, k=7, t=3."""
        if v != 24:
            return None
        
        # Use structure: 24 points as 6 groups of 4, or as 24 elements
        # Simple structured approach: create blocks with high overlap
        blocks = []
        seen = set()
        
        # Create blocks based on modular arithmetic patterns
        for start in range(v):
            for step in [1, 2, 3, 4, 5, 6]:
                block = []
                for i in range(k):
                    x = (start + i * step) % v
                    if x not in block:
                        block.append(x)
                if len(block) == k:
                    block.sort()
                    key = tuple(block)
                    if key not in seen:
                        seen.add(key)
                        blocks.append(block)
        
        return blocks

    # Main loop
    while time.time() < deadline:
        # Occasionally use affine initial pool
        if random.random() < 0.2:
            pool = affine_initial_pool()
            if pool and len(pool) > 0:
                # Start with affine blocks, then fill gaps
                blocks = list(pool)
                block_set = set(tuple(b) for b in blocks)
                
                # Check coverage
                uncovered = np.ones(num_triples, dtype=np.uint8)
                for block in blocks:
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        uncovered[idx] = 0
                
                # Add more blocks to cover remaining
                while np.any(uncovered):
                    if time.time() > deadline:
                        break
                    seed_idx = None
                    while True:
                        candidate = random.randint(0, num_triples - 1)
                        if uncovered[candidate]:
                            seed_idx = candidate
                            break
                    
                    seed_triple = all_t_subsets[seed_idx]
                    block = list(seed_triple)
                    
                    while len(block) < k:
                        best_gain = -1
                        best_points = []
                        block_set_cur = set(block)
                        block_pairs = list(itertools.combinations(block, t - 1))
                        
                        for x in all_points:
                            if x in block_set_cur:
                                continue
                            
                            gain = 0
                            for pair in block_pairs:
                                triple = tuple(sorted((x,) + pair))
                                idx = triple_to_idx[triple]
                                if uncovered[idx]:
                                    gain += 1
                            
                            if gain > best_gain:
                                best_gain = gain
                                best_points = [x]
                            elif gain == best_gain:
                                best_points.append(x)
                        
                        if best_points:
                            block.append(random.choice(best_points))
                        else:
                            available = [x for x in all_points if x not in block_set_cur]
                            if available:
                                block.append(random.choice(available))
                            else:
                                break
                    
                    block.sort()
                    key = tuple(block)
                    if key not in block_set:
                        block_set.add(key)
                        blocks.append(block)
                    
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        uncovered[idx] = 0
                
                blocks = local_search_remove_redundant(blocks)
                blocks = local_search_swap(blocks)
                blocks = local_search_remove_redundant(blocks)
            else:
                blocks = random_greedy_cover()
                blocks = local_search_remove_redundant(blocks)
                blocks = local_search_swap(blocks)
                blocks = local_search_remove_redundant(blocks)
        else:
            blocks = random_greedy_cover()
            blocks = local_search_remove_redundant(blocks)
            blocks = local_search_swap(blocks)
            blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    return best_blocks if best_blocks is not None else []
