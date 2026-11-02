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
    
    # Precompute all possible blocks? No, too many C(24,7) = 346104.
    # But for a fixed pivot, C(21,4) = 5985 is manageable.
    
    # Optimization: Precompute the triples for a block given its elements?
    # We will generate triples on the fly for the candidate blocks.
    
    while uncovered:
        if time.time() > time_limit:
            break
            
        # Pick a random uncovered triple to start the block
        # Using a random choice helps escape local optima in the greedy path
        pivot_triple = rng.choice(list(uncovered))
        pivot_set = set(pivot_triple)
        
        # Candidates are all k-subsets containing pivot_triple
        # This is equivalent to choosing (k-t) points from the remaining (v-t) points
        other_points = [p for p in range(v) if p not in pivot_set]
        
        best_gain = -1
        best_block = None
        
        # Iterate over all combinations of (k-t) points from other_points
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            
            # Calculate gain: number of uncovered triples in this block
            gain = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is not None:
            blocks.append(list(best_block))
            # Remove covered triples
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
    
    # Count how many times each triple is covered
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
                # Decrement counts for this block's triples
                for tri in block_triples[i]:
                    triple_cover_count[tri] -= 1
        
        blocks = new_blocks
        block_triples = new_block_triples
    
    return blocks

def _local_search(blocks, v, k, t, rng, time_limit):
    """
    Attempts to improve the covering by swapping blocks and reducing.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts and block triples
    triple_cover_count = defaultdict(int)
    block_triples = []
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        
        # Shuffle indices to try different orders
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            # Try to replace block i with a random candidate
            # Or a candidate biased towards covering "rare" triples?
            # For simplicity and speed, random is often sufficient if we have many iterations.
            candidate = sorted(rng.sample(range(v), k))
            
            if tuple(candidate) == tuple(blocks[i]):
                continue
                
            old_triples = set(block_triples[i])
            new_triples = set(itertools.combinations(candidate, t))
            
            # Check if swapping maintains coverage
            # We need to ensure that any triple in old_triples but not in new_triples
            # is covered by at least one OTHER block.
            # i.e., count[tri] > 1 for all tri in (old_triples - new_triples)
            
            lost_triples = old_triples - new_triples
            
            valid_swap = True
            for tri in lost_triples:
                if triple_cover_count[tri] == 1:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform swap
                # Update counts
                for tri in old_triples:
                    triple_cover_count[tri] -= 1
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = candidate
                block_triples[i] = list(new_triples)
                
                # Try to reduce
                # We need to check if any block is now redundant.
                # Instead of calling full _reduce_covering which re-calculates everything,
                # we can just check if the new block made others redundant, or if it itself is redundant.
                # But _reduce_covering is robust. Let's call it, but it's O(N * C(k,t)).
                # With N~70 and C(7,3)=35, it's fast.
                
                new_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    # Recalculate state for the new smaller set
                    triple_cover_count = defaultdict(int)
                    block_triples = []
                    for block in blocks:
                        tris = list(itertools.combinations(block, t))
                        block_triples.append(tris)
                        for tri in tris:
                            triple_cover_count[tri] += 1
                    break # Restart loop with new size
                else:
                    # Revert swap
                    for tri in old_triples:
                        triple_cover_count[tri] += 1
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    blocks[i] = sorted(rng.sample(range(v), k)) # Placeholder, need original
                    # We lost the original block! We need to store it before swapping.
                    
                    # Let's fix the revert logic.
                    # Store original block and triples before swapping.
                    # I will refactor the loop to store originals.
                    pass
                    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Initial Greedy Construction (Exhaustive per step)
    # This is the primary driver.
    rng = random.Random(42)
    blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Local Search Refinement
    # Try to improve the best solution found so far
    if time.time() < deadline:
        # Use a separate RNG for local search
        rng_ls = random.Random(123)
        improved_blocks = _local_search_safe(best_blocks, v, k, t, rng_ls, deadline)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)
            
    # 3. Multiple Randomized Greedy Runs
    # If we have time left, run more greedy constructions with different seeds
    seed_counter = 1
    while time.time() < deadline - 10: # Leave 10s for final reductions/checks
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
        
        if time.time() > deadline - 5:
            break
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
            # If we improved, maybe run local search on this new best?
            if time.time() < deadline - 15:
                rng_ls = random.Random(seed_counter + 1000)
                improved_blocks = _local_search_safe(best_blocks, v, k, t, rng_ls, deadline - 5)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks

def _local_search_safe(blocks, v, k, t, rng, time_limit):
    """
    Safe version of local search that properly handles reverts.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts and block triples
    triple_cover_count = defaultdict(int)
    block_triples = []
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        
        # Shuffle indices to try different orders
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            # Store original state to allow revert
            original_block = blocks[i]
            original_triples = block_triples[i]
            
            # Generate candidate
            candidate = sorted(rng.sample(range(v), k))
            
            if tuple(candidate) == tuple(original_block):
                continue
                
            new_triples = list(itertools.combinations(candidate, t))
            old_triples_set = set(original_triples)
            new_triples_set = set(new_triples)
            
            # Check validity: lost triples must be covered by others
            lost_triples = old_triples_set - new_triples_set
            
            valid_swap = True
            for tri in lost_triples:
                if triple_cover_count[tri] == 1:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform swap
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
                    # Recalculate state
                    triple_cover_count = defaultdict(int)
                    block_triples = []
                    for block in blocks:
                        tris = list(itertools.combinations(block, t))
                        block_triples.append(tris)
                        for tri in tris:
                            triple_cover_count[tri] += 1
                    break # Restart loop
                else:
                    # Revert swap
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
                    
    return blocks

# Note: The _local_search function defined earlier was flawed. 
# We replace its usage with _local_search_safe in construct.
# We can remove the flawed _local_search to avoid confusion, but keeping it doesn't hurt 
# as long as construct doesn't call it. I will remove it to be clean.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Initial Greedy Construction (Exhaustive per step)
    rng = random.Random(42)
    blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Local Search Refinement
    if time.time() < deadline:
        rng_ls = random.Random(123)
        improved_blocks = _local_search_safe(best_blocks, v, k, t, rng_ls, deadline)
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
                improved_blocks = _local_search_safe(best_blocks, v, k, t, rng_ls, deadline - 5)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks
