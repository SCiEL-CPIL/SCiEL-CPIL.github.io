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
    Simulated Annealing style local search.
    We maintain a valid covering. Moves are:
    1. Remove a redundant block (if any exist, though we usually start reduced).
    2. Swap a block with a random candidate.
    3. Swap a block with a candidate that covers "rare" triples (heuristic).
    
    To allow escaping local minima, we might temporarily accept a "worse" state?
    No, the constraint is strict validity. We cannot have an invalid state.
    So "SA" here means we explore the space of valid coverings.
    The "energy" is the number of blocks. We only move if size decreases or stays same (to explore).
    Actually, since we can only remove blocks to improve, and swaps keep size constant,
    the only way to improve is to find a swap that creates a redundancy, allowing a removal.
    
    Strategy:
    - Try to swap block i with a candidate.
    - If valid, perform swap.
    - Try to remove any now-redundant block.
    - If size decreased, keep.
    - If size same, we might keep the swap to explore, or revert. 
      Keeping the swap allows different configurations.
      We will accept swaps that maintain validity, and then aggressively reduce.
      If reduction fails, we can revert or keep. Keeping might help future swaps.
      Let's keep valid swaps that don't increase size (impossible) and try to reduce.
      If reduction doesn't reduce size, we can revert to avoid random walk, OR keep to diversify.
      Given the small search space of valid swaps, random walk is okay if we eventually reduce.
      However, to be safe and efficient, let's only accept swaps that lead to a reduction,
      OR perform a "neutral" swap with some probability to escape plateaus.
      
    Let's refine:
    1. Try swap.
    2. If valid, apply.
    3. Reduce.
    4. If reduced, accept.
    5. If not reduced, revert swap (backtrack) to maintain "best effort" trajectory,
       UNLESS we are stuck and want to explore. 
       Since we don't have a cost function other than size, and size doesn't change on swap,
       the only signal is "did it enable a removal?".
       If not, reverting is safer for convergence, but keeping helps diversity.
       Let's try: Accept swap if it reduces size. If not, revert.
       To escape local minima, we rely on the random selection of candidates.
       
    Additional Idea: "Directed" swaps.
    Instead of random candidate, pick a candidate that covers the triple with the lowest coverage count (among those covered by block i).
    This tries to increase redundancy of rare triples? No, that decreases their count if we remove block i?
    Wait. If we swap block i (which covers rare triple X) with block j (which does not cover X),
    X becomes uncovered -> Invalid.
    So we can only swap if all triples in block i are covered by others (count > 1).
    If a triple has count 1, block i is "essential" for that triple.
    
    So, valid swaps are only possible for blocks that are NOT essential for any of their triples.
    If a block is essential for a triple, we cannot remove it (or swap it out) unless the new block also covers that triple.
    
    So, for block i:
    - Find triples in block i with count == 1. These MUST be in the new candidate.
    - Find triples in block i with count > 1. These CAN be dropped.
    - The new candidate must contain all "must" triples.
    - If "must" triples > t, and they don't fit in a k-block, no swap is possible for this block (unless we change the definition).
      Actually, if a block has multiple count-1 triples, the new block must contain all of them.
      If the union of "must" triples has size > k, no valid swap exists for this block.
    
    This is a much more constrained and directed search.
    Let's implement this logic.
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
            
            # If no must triples, any valid block that doesn't break other counts?
            # Actually, if no must triples, we can swap with ANY block.
            # But we want to find a swap that creates redundancy.
            
            # The new block MUST contain all points in the union of must_triples.
            # Let M be the set of points in must_triples.
            must_points = set()
            for tri in must_triples:
                must_points.update(tri)
                
            if len(must_points) > k:
                # Cannot form a block containing all must points
                continue
                
            # We need to choose (k - len(must_points)) points from the remaining (v - len(must_points)) points.
            # To maximize chance of creating redundancy, we should prefer points that are in "dense" areas?
            # Or just random?
            # Let's try a few random completions.
            
            found_better = False
            # Limit number of candidates to try per block to save time
            max_candidates = 5
            for _ in range(max_candidates):
                # Randomly select the remaining points
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
                
                # Check validity:
                # 1. All must_triples must be in new_triples (guaranteed by construction)
                # 2. All other old_triples (count > 1) that are NOT in new_triples must have count > 1
                #    (i.e., covered by others)
                
                valid = True
                # Check lost triples
                for tri in old_triples_set:
                    if tri not in new_triples_set:
                        if triple_cover_count[tri] == 1:
                            # This triple is only covered by block i, and we are removing it.
                            # But wait, if it's in must_triples, it's in new_triples.
                            # So if it's lost, it wasn't a must triple, so count was > 1.
                            # So it should be covered by others.
                            # The check `triple_cover_count[tri] == 1` would be false here if logic holds.
                            # Let's be safe.
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
                    # Recalculate state
                    triple_cover_count, block_triples = get_state(blocks)
                    break # Restart loop with new size
                else:
                    # Revert
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
                    
            if found_better:
                break # Restart outer loop

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
