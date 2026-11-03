import itertools
import time
import random
from collections import defaultdict

def _greedy_cover_exhaustive(v, k, t, rng, time_limit):
    """
    Constructs a covering using a greedy heuristic.
    For each step, picks a pivot triple and finds the best block containing it
    by evaluating all possible completions.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > time_limit:
            break
            
        # Pick a random uncovered triple to start the block
        pivot_triple = rng.choice(list(uncovered))
        pivot_set = set(pivot_triple)
        
        other_points = [p for p in range(v) if p not in pivot_set]
        
        best_gain = -1
        best_block = None
        
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            
            gain = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is not None:
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(tri)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    A block is redundant if every triple it covers is also covered by at least one other block.
    """
    if not blocks:
        return blocks
    
    triple_cover_count = defaultdict(int)
    block_triples = []
    
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        
        for i, block in enumerate(blocks):
            is_essential = False
            for tri in block_triples[i]:
                if triple_cover_count[tri] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
                new_block_triples.append(block_triples[i])
            else:
                changed = True
                for tri in block_triples[i]:
                    triple_cover_count[tri] -= 1
        
        blocks = new_blocks
        block_triples = new_block_triples
    
    return blocks

def _local_search_sa(blocks, v, k, t, rng, time_limit):
    """
    Local search with swap and merge operations.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    def get_state(blocks):
        triple_cover_count = defaultdict(int)
        block_triples = []
        for block in blocks:
            tris = list(itertools.combinations(block, t))
            block_triples.append(tris)
            for tri in tris:
                triple_cover_count[tri] += 1
        return triple_cover_count, block_triples

    triple_cover_count, block_triples = get_state(blocks)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        
        # Shuffle indices
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            original_block = blocks[i]
            original_triples = block_triples[i]
            
            # Identify "must" triples (count == 1)
            must_triples = [tri for tri in original_triples if triple_cover_count[tri] == 1]
            
            must_points = set()
            for tri in must_triples:
                must_points.update(tri)
                
            if len(must_points) > k:
                continue
                
            found_better = False
            max_candidates = 10
            for _ in range(max_candidates):
                available_points = [p for p in range(v) if p not in must_points]
                if len(available_points) < k - len(must_points):
                    continue
                extra_points = rng.sample(available_points, k - len(must_points))
                candidate = sorted(must_points | set(extra_points))
                
                if tuple(candidate) == tuple(original_block):
                    continue
                
                new_triples = list(itertools.combinations(candidate, t))
                old_triples_set = set(original_triples)
                new_triples_set = set(new_triples)
                
                valid = True
                for tri in old_triples_set:
                    if tri not in new_triples_set:
                        if triple_cover_count[tri] == 1:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                # Perform Swap
                for tri in original_triples:
                    triple_cover_count[tri] -= 1
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = candidate
                block_triples[i] = new_triples
                
                # Try to reduce
                new_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    found_better = True
                    triple_cover_count, block_triples = get_state(blocks)
                    break
                else:
                    # Revert
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
                    
            if found_better:
                break

        # Try merge operation: replace two blocks with one if possible
        if not improved and time.time() < time_limit and n_blocks >= 2:
            pair_indices = list(range(n_blocks))
            rng.shuffle(pair_indices)
            merge_found = False
            
            for a in range(min(n_blocks, 5)):
                if time.time() > time_limit:
                    break
                i = pair_indices[a]
                for b in range(a + 1, min(n_blocks, 5)):
                    j = pair_indices[b]
                    
                    # Check if union of must-points from blocks i and j fits in k
                    must_i = [tri for tri in block_triples[i] if triple_cover_count[tri] == 1]
                    must_j = [tri for tri in block_triples[j] if triple_cover_count[tri] == 1]
                    
                    must_points_i = set()
                    for tri in must_i:
                        must_points_i.update(tri)
                    must_points_j = set()
                    for tri in must_j:
                        must_points_j.update(tri)
                    
                    all_must = must_points_i | must_points_j
                    if len(all_must) > k:
                        continue
                    
                    # Try to find a block containing all_must that covers all triples from both blocks
                    # that are not covered by other blocks
                    needed_triples = set()
                    for tri in block_triples[i]:
                        if triple_cover_count[tri] == 1:
                            needed_triples.add(tri)
                    for tri in block_triples[j]:
                        if triple_cover_count[tri] == 1:
                            needed_triples.add(tri)
                    
                    # Check if needed_triples can fit in a k-block
                    needed_points = set()
                    for tri in needed_triples:
                        needed_points.update(tri)
                    if len(needed_points) > k:
                        continue
                    
                    # Find all triples that must be covered by the new block
                    # (those with count == 1 from either block i or j)
                    # And check they all fit
                    candidate_extra = [p for p in range(v) if p not in needed_points]
                    if len(candidate_extra) < k - len(needed_points):
                        continue
                    
                    # Try a few random completions
                    for _ in range(3):
                        extra = rng.sample(candidate_extra, k - len(needed_points))
                        candidate = sorted(needed_points | set(extra))
                        
                        cand_triples = set(itertools.combinations(candidate, t))
                        
                        # Check all needed triples are covered
                        if not needed_triples.issubset(cand_triples):
                            continue
                        
                        # Check: removing blocks i and j, adding candidate, is valid?
                        # For each triple in block i or j with count == 1, it must be in candidate
                        # For each triple in block i or j with count > 1, it's fine to drop
                        valid = True
                        for tri in block_triples[i]:
                            if tri not in cand_triples and triple_cover_count[tri] == 1:
                                valid = False
                                break
                        if valid:
                            for tri in block_triples[j]:
                                if tri not in cand_triples and triple_cover_count[tri] == 1:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        # Apply: remove i and j, add candidate
                        for tri in block_triples[i]:
                            triple_cover_count[tri] -= 1
                        for tri in block_triples[j]:
                            triple_cover_count[tri] -= 1
                        for tri in cand_triples:
                            triple_cover_count[tri] += 1
                        
                        new_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:] + [candidate]
                        new_blocks = _reduce_covering(new_blocks, v, k, t)
                        
                        if len(new_blocks) < len(blocks):
                            blocks = new_blocks
                            improved = True
                            merge_found = True
                            triple_cover_count, block_triples = get_state(blocks)
                            break
                        else:
                            # Revert
                            for tri in cand_triples:
                                triple_cover_count[tri] -= 1
                            for tri in block_triples[j]:
                                triple_cover_count[tri] += 1
                            for tri in block_triples[i]:
                                triple_cover_count[tri] += 1
                        
                        if merge_found:
                            break
                    if merge_found:
                        break
                if merge_found:
                    break
            
            if merge_found:
                break

    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Initial Greedy Construction
    rng = random.Random(42)
    blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Local Search Refinement
    if time.time() < deadline:
        rng_ls = random.Random(123)
        improved_blocks = _local_search_sa(best_blocks, v, k, t, rng_ls, deadline)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)
            
    # 3. Multiple Randomized Greedy Runs
    seed_counter = 1
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
        
        if time.time() > deadline - 5:
            break
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
            if time.time() < deadline - 15:
                rng_ls = random.Random(seed_counter + 1000)
                improved_blocks = _local_search_sa(best_blocks, v, k, t, rng_ls, deadline - 5)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks
