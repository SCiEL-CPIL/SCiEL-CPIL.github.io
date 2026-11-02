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
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(tri)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
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

def _local_search_directed(blocks, v, k, t, rng, time_limit):
    """
    Directed local search:
    1. Remove redundant blocks.
    2. Try to replace a block with critical triples with a new block that covers those critical triples,
       hoping to make other blocks redundant.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    if not blocks:
        return blocks

    # Precompute
    triple_cover_count = defaultdict(int)
    block_triples = []
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    improved = True
    stall_count = 0
    
    while improved and time.time() < time_limit:
        improved = False
        stall_count += 1
        
        # 1. Reduce
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
            stall_count = 0
            continue
            
        # If no reduction, try to find a swap that enables reduction
        # Pick a random block to potentially replace
        n_blocks = len(blocks)
        if n_blocks == 0:
            break
            
        # Identify blocks with critical triples
        candidates_for_swap = []
        for i in range(n_blocks):
            has_critical = False
            for tri in block_triples[i]:
                if triple_cover_count[tri] == 1:
                    has_critical = True
                    break
            if has_critical:
                candidates_for_swap.append(i)
                
        if not candidates_for_swap:
            # No critical blocks? Then all blocks are redundant?
            # _reduce_covering should have removed them.
            # If we are here, it means no block is redundant, but no block has critical triples?
            # That implies all counts are > 1. Then all blocks are redundant.
            # So this case shouldn't happen if _reduce_covering is correct.
            break
            
        # Try a few random candidates
        rng.shuffle(candidates_for_swap)
        swap_found = False
        
        for i in candidates_for_swap[:5]: # Try up to 5 swaps per iteration
            if time.time() > time_limit:
                break
                
            # Get critical triples in block i
            critical_triples = [tri for tri in block_triples[i] if triple_cover_count[tri] == 1]
            
            # We need a new block that contains ALL critical_triples
            # Union of points in critical_triples
            points_in_critical = set()
            for tri in critical_triples:
                points_in_critical.update(tri)
                
            if len(points_in_critical) > k:
                continue # Impossible to cover all critical triples in one block
                
            # Generate candidates containing points_in_critical
            # We need to choose k - len(points_in_critical) points from the remaining v - len(points_in_critical) points
            remaining_points = [p for p in range(v) if p not in points_in_critical]
            
            # To avoid generating too many, we can limit the number of candidates tested
            # Or just pick one random candidate?
            # If we pick random, we might miss the good one.
            # Let's try a few random completions.
            found_valid_swap = False
            best_candidate = None
            
            # If the set of points is small, we can be more exhaustive?
            # For now, try 100 random completions
            for _ in range(100):
                if not remaining_points:
                    candidate = sorted(points_in_critical)
                else:
                    fill = rng.sample(remaining_points, k - len(points_in_critical))
                    candidate = sorted(points_in_critical | set(fill))
                    
                # Check if this candidate is the same as the current block
                if tuple(candidate) == tuple(blocks[i]):
                    continue
                    
                # Check if this candidate is already in the list of blocks?
                # It doesn't strictly need to be distinct if we are replacing, 
                # but if it's identical to another block, it's redundant.
                # _reduce_covering will handle redundancy.
                
                # Check validity:
                # 1. Must contain all critical_triples (ensured by construction)
                # 2. Must not "break" any other triple?
                #    Removing block i decreases count of all tris in block i by 1.
                #    Adding candidate increases count of all tris in candidate by 1.
                #    We already know critical_triples (count 1) are in candidate, so their count goes 1 -> 0 (remove i) -> 1 (add candidate). Safe.
                #    What about other tris in block i?
                #    They had count >= 2 (since they were not critical).
                #    After removing i, count >= 1. Safe.
                #    What about tris in candidate but not in block i?
                #    They gain count. Safe.
                #    So any candidate containing critical_triples is a VALID swap.
                
                # We just want to find one that might help reduction.
                # How to evaluate "help"?
                # Ideally, the new block covers triples that are "scarce" (count 1 or 2) in OTHER blocks,
                # making those other blocks non-critical?
                # Or simply, it changes the distribution.
                
                # Let's just accept the first valid candidate and see if reduction happens.
                # If not, revert.
                
                # Perform Swap
                old_block = blocks[i]
                old_triples = block_triples[i]
                
                # Update counts
                for tri in old_triples:
                    triple_cover_count[tri] -= 1
                new_triples = list(itertools.combinations(candidate, t))
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = candidate
                block_triples[i] = new_triples
                
                # Try to reduce
                reduced_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(reduced_blocks) < len(blocks):
                    # Success!
                    blocks = reduced_blocks
                    improved = True
                    swap_found = True
                    stall_count = 0
                    # Recalculate state
                    triple_cover_count = defaultdict(int)
                    block_triples = []
                    for block in blocks:
                        tris = list(itertools.combinations(block, t))
                        block_triples.append(tris)
                        for tri in tris:
                            triple_cover_count[tri] += 1
                    break # Restart the while loop
                else:
                    # Revert
                    for tri in old_triples:
                        triple_cover_count[tri] += 1
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    blocks[i] = old_block
                    block_triples[i] = old_triples
                    
            if swap_found:
                break
                
        if not improved:
            # If we stalled for too long, maybe we are stuck in a local minimum.
            # We can try a random perturbation?
            # For now, just break to allow other strategies (like new greedy) to run.
            if stall_count > 50:
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
    
    # 2. Directed Local Search
    if time.time() < deadline:
        rng_ls = random.Random(123)
        improved_blocks = _local_search_directed(best_blocks, v, k, t, rng_ls, deadline)
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
                improved_blocks = _local_search_directed(best_blocks, v, k, t, rng_ls, deadline - 5)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks
