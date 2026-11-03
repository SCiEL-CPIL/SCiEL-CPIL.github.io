"""
Optimized covering design C(24,7,3) using precomputed triple mappings and
randomized local search with simulated annealing.
"""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute_triples(v, t):
    """Precompute all t-subsets as frozensets and map to indices."""
    all_triples = []
    triple_to_idx = {}
    for i, combo in enumerate(itertools.combinations(range(v), t)):
        fs = frozenset(combo)
        all_triples.append(fs)
        triple_to_idx[fs] = i
    return all_triples, triple_to_idx


def _block_triple_indices(block, triple_to_idx):
    """Get list of triple indices covered by a block."""
    return [triple_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]


def _greedy_cover(v, k, t, rng, all_triples, triple_to_idx):
    """Fast greedy construction with precomputed triple indices."""
    n_triples = len(all_triples)
    uncovered = [1] * n_triples  # 1 = uncovered, 0 = covered
    n_uncovered = n_triples
    blocks = []
    block_set = set()
    
    while n_uncovered > 0:
        # Pick a random uncovered triple as seed
        # Find an uncovered triple
        while True:
            idx = rng.randrange(n_triples)
            if uncovered[idx]:
                break
        
        seed = list(all_triples[idx])
        rng.shuffle(seed)
        block = list(seed)
        
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in triple_to_idx:
                        idx = triple_to_idx[fs]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            # Update uncovered
            for c in itertools.combinations(block, t):
                fs = frozenset(c)
                if fs in triple_to_idx:
                    idx = triple_to_idx[fs]
                    if uncovered[idx]:
                        uncovered[idx] = 0
                        n_uncovered -= 1
                
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 145.0  # leave 5s buffer for safety
    
    # Precompute triples
    all_triples, triple_to_idx = _precompute_triples(v, t)
    n_triples = len(all_triples)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_triples, triple_to_idx)
    
    # Precompute triple indices for each block
    block_triple_indices = [_block_triple_indices(b, triple_to_idx) for b in blocks]
    
    # Build coverage counts
    counts = [0] * n_triples
    for idxs in block_triple_indices:
        for idx in idxs:
            counts[idx] += 1
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Helper to check validity: all counts >= 1
    def is_valid():
        return all(c >= 1 for c in counts)
    
    # Simulated Annealing / Local Search Loop
    iteration = 0
    temperature = 100.0
    cooling = 0.9995
    no_improvement = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            idxs = block_triple_indices[i]
            redundant = True
            for idx in idxs:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                b = blocks[i]
                idxs = block_triple_indices[i]
                blocks.pop(i)
                block_triple_indices.pop(i)
                for idx in idxs:
                    counts[idx] -= 1
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    no_improvement = 0
                else:
                    no_improvement += 1
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_idxs = block_triple_indices[i]
            
            # Identify critical triples (count == 1) in old_block
            critical_idxs = [idx for idx in old_idxs if counts[idx] == 1]
            
            if not critical_idxs:
                # No critical triples, try random removal (should have been caught above)
                continue
            
            # Collect points needed to cover critical triples
            points_needed = set()
            for idx in critical_idxs:
                points_needed.update(all_triples[idx])
            
            if len(points_needed) > k:
                # Cannot cover all critical triples in one block
                continue
            
            # Construct new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Greedy fill to maximize coverage of uncovered or singly-covered triples
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                for rest in itertools.combinations(new_block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in triple_to_idx:
                        idx = triple_to_idx[fs]
                        if counts[idx] == 0:
                            gain += 100
                        elif counts[idx] == 1:
                            gain += 10
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)
            
            if len(new_block_set) != k:
                continue
            
            new_block = tuple(sorted(new_block_set))
            if new_block == old_block:
                continue
            
            new_idxs = _block_triple_indices(new_block, triple_to_idx)
            
            # Simulate swap: remove old, add new
            # Check if valid
            valid = True
            temp_counts = counts[:]  # Copy for checking
            
            for idx in old_idxs:
                temp_counts[idx] -= 1
            for idx in new_idxs:
                temp_counts[idx] += 1
            
            for idx in range(n_triples):
                if temp_counts[idx] < 1:
                    valid = False
                    break
            
            if valid:
                # Accept swap
                blocks[i] = new_block
                block_triple_indices[i] = new_idxs
                counts = temp_counts
                no_improvement += 1
            else:
                no_improvement += 1
        
        # 3. Random restart if stuck
        if no_improvement >= 500 or iteration % 200 == 0:
            new_cover = _greedy_cover(v, k, t, rng, all_triples, triple_to_idx)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                block_triple_indices = [_block_triple_indices(b, triple_to_idx) for b in blocks]
                counts = [0] * n_triples
                for idxs in block_triple_indices:
                    for idx in idxs:
                        counts[idx] += 1
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    no_improvement = 0
            else:
                no_improvement = 0
        
        # Cool down
        temperature *= cooling
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
