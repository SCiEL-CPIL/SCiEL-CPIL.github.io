import itertools
import time
import random
import numpy as np
from heapq import heappush, heappop

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = all_triplets
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of triplet IDs it covers
    block_triplet_ids = []
    for block in all_blocks:
        tids = [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        block_triplet_ids.append(tids)
    
    # Build a lookup: for each triplet, which blocks contain it
    triplet_blocks = [[] for _ in range(n_triplets)]
    for bi, tids in enumerate(block_triplet_ids):
        for tid in tids:
            triplet_blocks[tid].append(bi)
    
    # Precompute block_triplet_ids as numpy arrays for vectorized gain computation
    block_triplet_ids_np = [np.array(tids, dtype=np.int32) for tids in block_triplet_ids]
    
    # Helper to enumerate blocks containing a set of points
    def blocks_containing(points):
        points_set = frozenset(points)
        if len(points_set) > k:
            return
        remaining = [p for p in range(v) if p not in points_set]
        need = k - len(points_set)
        if need < 0:
            return
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points_set | set(comb)))
    
    # Vectorized greedy using numpy
    def fast_greedy(seed):
        rng = random.Random(seed)
        counts = np.zeros(n_triplets, dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 35:
                break
            
            uncovered_mask = (counts == 0).astype(np.int32)
            n_uncovered = int(uncovered_mask.sum())
            if n_uncovered == 0:
                break
            
            # Compute gain for each block using vectorized operations
            # For each block, count how many of its triplets are uncovered
            best_gain = -1
            best_block_idx = -1
            
            # Process blocks in chunks to manage memory
            chunk_size = 10000
            for start in range(0, n_blocks, chunk_size):
                end = min(start + chunk_size, n_blocks)
                # Compute gains for this chunk
                for bi in range(start, end):
                    tids = block_triplet_ids_np[bi]
                    gain = int(np.sum(uncovered_mask[tids]))
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
            
            if best_block_idx < 0:
                break
            
            selected.append(best_block_idx)
            tids = block_triplet_ids_np[best_block_idx]
            counts[tids] += 1
        
        return [all_blocks[i] for i in selected]
    
    # Optimized local search with multiple move types
    def optimize(blocks, time_budget=30):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        
        def get_triplet_ids(block):
            return [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        
        def is_redundant(block):
            for trip in itertools.combinations(block, t):
                tid = triplet_to_id[trip]
                if counts[tid] == 1:
                    return False
            return True
        
        def try_remove_redundant():
            n = len(blocks)
            for i in range(n):
                if is_redundant(blocks[i]):
                    b = blocks.pop(i)
                    for tid in get_triplet_ids(b):
                        counts[tid] -= 1
                    return True
            return False
        
        def try_2_for_1():
            n = len(blocks)
            if n < 2:
                return False
            
            # Systematically try pairs, but limit total work
            max_pairs = 200
            pairs_tried = 0
            
            # Use random sampling of pairs
            for _ in range(max_pairs):
                if time.time() > start_time + time_budget:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                pairs_tried += 1
                
                b1 = blocks[i]
                b2 = blocks[j]
                ids1 = set(get_triplet_ids(b1))
                ids2 = set(get_triplet_ids(b2))
                
                # Find triplets that would become uncovered
                needed = []
                for tid in (ids1 | ids2):
                    if counts[tid] == 1:
                        needed.append(tid)
                
                if not needed:
                    # Both blocks are redundant
                    for idx in sorted([i, j], reverse=True):
                        b_rem = blocks.pop(idx)
                        for tid in get_triplet_ids(b_rem):
                            counts[tid] -= 1
                    return True
                
                # Find points needed
                points_needed = set()
                for tid in needed:
                    points_needed.update(id_to_triplet[tid])
                
                if len(points_needed) > k:
                    continue
                
                # Try to find a replacement block
                for new_block in blocks_containing(points_needed):
                    if set(new_block) == set(b1) or set(new_block) == set(b2):
                        continue
                    new_ids = set(get_triplet_ids(new_block))
                    if all(tid in new_ids for tid in needed):
                        # Perform swap
                        for idx in sorted([i, j], reverse=True):
                            b_rem = blocks.pop(idx)
                            for tid in get_triplet_ids(b_rem):
                                counts[tid] -= 1
                        blocks.append(new_block)
                        for tid in get_triplet_ids(new_block):
                            counts[tid] += 1
                        return True
            
            return False
        
        def try_1_for_1():
            n = len(blocks)
            if n == 0:
                return False
            
            # Try several random 1-for-1 swaps
            for _ in range(50):
                if time.time() > start_time + time_budget:
                    break
                i = random.randint(0, n - 1)
                b1 = blocks[i]
                ids1 = set(get_triplet_ids(b1))
                needed = [tid for tid in ids1 if counts[tid] == 1]
                
                if not needed:
                    # Block is redundant, just remove it
                    blocks.pop(i)
                    for tid in ids1:
                        counts[tid] -= 1
                    return True
                
                points_needed = set()
                for tid in needed:
                    points_needed.update(id_to_triplet[tid])
                
                if len(points_needed) > k:
                    continue
                
                for new_block in blocks_containing(points_needed):
                    if set(new_block) == set(b1):
                        continue
                    new_ids = set(get_triplet_ids(new_block))
                    if all(tid in new_ids for tid in needed):
                        # Replace
                        for tid in ids1:
                            counts[tid] -= 1
                        blocks[i] = new_block
                        for tid in new_ids:
                            counts[tid] += 1
                        return True
            
            return False
        
        # Main optimization loop
        no_improve_count = 0
        max_no_improve = 5
        
        while time.time() < start_time + time_budget:
            improved = False
            
            # 1. Remove redundant blocks
            if try_remove_redundant():
                improved = True
                no_improve_count = 0
                continue
            
            # 2. Try 2-for-1 replacement
            if try_2_for_1():
                improved = True
                no_improve_count = 0
                continue
            
            # 3. Try 1-for-1 replacement
            if try_1_for_1():
                improved = True
                no_improve_count = 0
                continue
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
        
        return blocks
    
    # Ensure validity: fill in any missing triplets
    def ensure_valid(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        
        uncovered = [i for i in range(n_triplets) if counts[i] == 0]
        if not uncovered:
            return blocks
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            tid = uncovered[0]
            
            best_block = None
            best_gain = -1
            for bi in triplet_blocks[tid]:
                tids = block_triplet_ids_np[bi]
                gain = int(np.sum((counts[tids] == 0).astype(np.int32)))
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block is not None:
                blocks.append(all_blocks[best_block])
                for tid2 in block_triplet_ids_np[best_block]:
                    counts[tid2] += 1
                uncovered = [i for i in range(n_triplets) if counts[i] == 0]
            else:
                break
        
        return blocks
    
    # Verify a set of blocks is valid
    def is_valid(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for trip in itertools.combinations(b, t):
                counts[triplet_to_id[trip]] += 1
        return np.min(counts) > 0
    
    best_blocks = None
    best_len = float('inf')
    
    # Adaptive number of greedy runs based on remaining time
    n_greedy_runs = 6
    for seed in range(n_greedy_runs):
        if time.time() > deadline - 50:
            break
        
        blocks = fast_greedy(seed)
        if not blocks:
            continue
        
        # Budget local search time adaptively
        remaining = deadline - time.time() - 10
        ls_budget = max(5, min(25, remaining / (n_greedy_runs - seed)))
        opt_blocks = optimize(blocks, time_budget=ls_budget)
        
        # Ensure validity
        opt_blocks = ensure_valid(opt_blocks)
        
        if is_valid(opt_blocks):
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
    
    # If no valid solution found, do a pure greedy
    if best_blocks is None:
        blocks = fast_greedy(0)
        best_blocks = ensure_valid(blocks)
    
    return best_blocks

if __name__ == "__main__":
    pass
