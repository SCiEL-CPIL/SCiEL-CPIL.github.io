import itertools
import time
import random
from collections import defaultdict

def _fast_greedy(v, k, t, rng, time_limit):
    """
    Fast greedy: pick random uncovered triple, pick random completion.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    start_time = time.time()
    
    # Precompute points for fast exclusion
    points = list(range(v))
    
    while uncovered:
        if time.time() > time_limit:
            break
            
        # Pick a random uncovered triple
        pivot_triple = rng.choice(list(uncovered))
        pivot_set = set(pivot_triple)
        
        # Find valid completions
        # We need k-t more points from v-t available points
        available = [p for p in points if p not in pivot_set]
        
        if len(available) < k - t:
            # Should not happen if v >= k
            break
            
        # Pick a random completion
        # To be slightly smarter, we could pick points that are in many uncovered triples?
        # For "fast" greedy, pure random is fastest.
        completion = rng.sample(available, k - t)
        
        block = tuple(sorted(pivot_set | set(completion)))
        blocks.append(list(block))
        
        # Remove covered triples
        for tri in itertools.combinations(block, t):
            uncovered.discard(tri)
            
    return blocks

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

def _local_search_sa(blocks, v, k, t, rng, time_limit):
    """
    Simulated Annealing / Random Walk local search.
    Maintains validity. Accepts valid swaps.
    Tries to reduce size after each swap.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    triple_cover_count = defaultdict(int)
    block_triples = []
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    best_blocks = [list(b) for b in blocks]
    best_size = len(blocks)
    
    # Temperature for accepting "neutral" moves?
    # We accept ALL valid moves. This is a random walk on the graph of valid coverings of fixed size.
    # The goal is to find a "hole" where a block can be removed.
    
    # Precompute set of all points for sampling
    points = list(range(v))
    
    while time.time() < time_limit:
        n_blocks = len(blocks)
        if n_blocks < best_size:
            best_size = n_blocks
            best_blocks = [list(b) for b in blocks]
            
        # Pick a random block to swap
        i = rng.randint(0, n_blocks - 1)
        original_block = blocks[i]
        original_triples = block_triples[i]
        original_triples_set = set(original_triples)
        
        # Generate candidate
        # Bias: try to include points that are in "rare" triples?
        # For now, random sample.
        candidate = sorted(rng.sample(points, k))
        
        if tuple(candidate) == tuple(original_block):
            continue
            
        new_triples = list(itertools.combinations(candidate, t))
        new_triples_set = set(new_triples)
        
        # Check validity of swap
        # Lost triples must be covered by others
        lost_triples = original_triples_set - new_triples_set
        
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
            # We can do a quick check: is the new block redundant? Or any other block?
            # Full reduction is expensive. Let's do a targeted reduction.
            # Check if block i is now redundant.
            is_i_redundant = True
            for tri in new_triples:
                if triple_cover_count[tri] == 1:
                    is_i_redundant = False
                    break
                    
            if is_i_redundant:
                # Remove block i
                for tri in new_triples:
                    triple_cover_count[tri] -= 1
                blocks.pop(i)
                block_triples.pop(i)
                # After removal, other blocks might become redundant?
                # Unlikely if we just removed a redundant one, but possible if it was "shielding" others?
                # No, removing a block only decreases counts. If a block was essential before, it remains essential.
                # If a block was non-essential, it remains non-essential or becomes more non-essential.
                # So we should check for other redundancies.
                # Let's run a full reduction pass since size changed.
                blocks = _reduce_covering(blocks, v, k, t)
                
                # Recalculate state
                triple_cover_count = defaultdict(int)
                block_triples = []
                for block in blocks:
                    tris = list(itertools.combinations(block, t))
                    block_triples.append(tris)
                    for tri in tris:
                        triple_cover_count[tri] += 1
                
                # We reduced size, so this is a good move.
                # We can continue searching from this new state.
                continue
            else:
                # Swap was valid but didn't lead to immediate reduction.
                # We keep the swap (Random Walk).
                # This allows us to explore the space.
                pass
        # If invalid, we just skip (don't revert, since we didn't apply)
        
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Multiple Fast Greedy Runs
    # Fast greedy is very fast, so we can run many.
    seed_counter = 0
    while time.time() < deadline - 20:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _fast_greedy(v, k, t, rng, deadline)
        if not blocks:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        if best_size <= 69:
            break
            
    if not best_blocks:
        # Fallback to slow greedy if fast failed (shouldn't happen)
        rng = random.Random(42)
        best_blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
        best_blocks = _reduce_covering(best_blocks, v, k, t)
        best_size = len(best_blocks)

    # 2. Local Search Refinement on Best Solution
    if time.time() < deadline - 5:
        rng_ls = random.Random(123 + seed_counter)
        improved_blocks = _local_search_sa(best_blocks, v, k, t, rng_ls, deadline)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)
            
    # 3. If time remains, run local search on other good solutions?
    # Or just more greedy?
    # Let's do a few more greedy + local search combos if time allows.
    
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _fast_greedy(v, k, t, rng, deadline)
        if not blocks:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        # Only do local search if this greedy result is competitive
        if len(blocks) <= best_size + 2:
            if time.time() < deadline - 5:
                rng_ls = random.Random(seed_counter + 1000)
                improved_blocks = _local_search_sa(blocks, v, k, t, rng_ls, deadline)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks
