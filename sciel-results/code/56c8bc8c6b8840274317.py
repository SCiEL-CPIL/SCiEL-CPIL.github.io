import time
import random
import itertools

def _greedy_construct(v, k, t, deadline, rng):
    """
    Randomized greedy set cover for the covering design problem.
    Precomputes all candidate blocks and their covered triples.
    Iteratively selects the block covering the most uncovered triples,
    with random tie-breaking to explore different basins.
    """
    # Precompute all candidate blocks
    all_blocks = [frozenset(b) for b in itertools.combinations(range(v), k)]
    # Precompute which triples each block covers
    block_triples = [set(itertools.combinations(b, t)) for b in all_blocks]
    # Total number of triples
    all_triples = set(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    selected = []
    
    while uncovered and time.time() < deadline:
        # Find the block with maximum gain (covers most uncovered triples)
        best_gain = -1
        best_candidates = []
        for i, bt in enumerate(block_triples):
            gain = len(uncovered & bt)
            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain and gain > 0:
                best_candidates.append(i)
        
        if not best_candidates or best_gain <= 0:
            break
        
        # Pick randomly among tied best candidates to add randomness
        idx = rng.choice(best_candidates)
        selected.append(list(all_blocks[idx]))
        uncovered.difference_update(block_triples[idx])
    
    return selected

def _local_search(blocks, v, k, t, deadline, rng):
    """
    Local search: try to remove redundant blocks.
    A block is redundant if every triple it covers is also covered by another selected block.
    """
    if not blocks:
        return blocks
    
    # Build a map: triple -> list of block indices that cover it
    triple_to_blocks = {}
    for bi, block in enumerate(blocks):
        for tri in itertools.combinations(block, t):
            triple_to_blocks.setdefault(tri, []).append(bi)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing blocks in random order
        order = list(range(len(blocks)))
        rng.shuffle(order)
        for bi in order:
            if time.time() > deadline:
                break
            block = blocks[bi]
            # Check if block is redundant: for every triple in block, is there another block covering it?
            redundant = True
            for tri in itertools.combinations(block, t):
                covering = triple_to_blocks.get(tri, [])
                # Need at least one other block (not bi) covering this triple
                if len(covering) < 2:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                blocks.pop(bi)
                # Update triple_to_blocks
                for tri in itertools.combinations(block, t):
                    if tri in triple_to_blocks:
                        triple_to_blocks[tri] = [x for x in triple_to_blocks[tri] if x != bi]
                        # Fix indices for blocks after bi
                        # Actually this is expensive; let's rebuild more carefully
                        # For simplicity, just mark as changed and break to restart
                        changed = True
                        break
                if changed:
                    # Rebuild triple_to_blocks from scratch after removal
                    triple_to_blocks = {}
                    for j, b in enumerate(blocks):
                        for tri in itertools.combinations(b, t):
                            triple_to_blocks.setdefault(tri, []).append(j)
                    break
    
    return blocks

def _refine(blocks, v, k, t, deadline, rng):
    """
    Additional refinement: try replacing blocks with alternatives that cover the same 
    critical triples but perhaps allow further pruning.
    For now, just do repeated local search passes.
    """
    # Do multiple passes of local search
    for _ in range(5):
        if time.time() >= deadline:
            break
        blocks = _local_search(blocks, v, k, t, deadline, rng)
    return blocks

def construct(v, k, t):
    """
    Main entry point: construct a covering design C(v, k, t).
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Seed for reproducibility but also randomness
    seed = int(time.time() * 1000) % (2**31)
    rng = random.Random(seed)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy passes
    max_passes = 20
    for pass_num in range(max_passes):
        if time.time() >= deadline:
            break
        
        # Create a fresh RNG with different seed for each pass
        pass_rng = random.Random(seed + pass_num * 7919)
        
        # Run greedy
        blocks = _greedy_construct(v, k, t, deadline, pass_rng)
        
        if not blocks:
            continue
        
        # Local search refinement
        blocks = _local_search(blocks, v, k, t, deadline, pass_rng)
        blocks = _refine(blocks, v, k, t, deadline, pass_rng)
        
        # Verify coverage (quick check)
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        all_triples = set(itertools.combinations(range(v), t))
        if covered == all_triples:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(b) for b in blocks]
        else:
            # If greedy didn't fully cover (shouldn't happen), try to complete
            uncovered = all_triples - covered
            # Add blocks until covered
            while uncovered and time.time() < deadline:
                # Find a block covering the most uncovered
                best_gain = -1
                best_block = None
                for b in itertools.combinations(range(v), k):
                    b_set = set(b)
                    gain = len(uncovered & set(itertools.combinations(b, t)))
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
                if best_block is None or best_gain <= 0:
                    break
                blocks.append(list(best_block))
                covered.update(itertools.combinations(best_block, t))
                uncovered = all_triples - covered
            
            if covered == all_triples and len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(b) for b in blocks]
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        pass_rng = random.Random(0)
        best_blocks = _greedy_construct(v, k, t, deadline, pass_rng)
        best_blocks = _local_search(best_blocks, v, k, t, deadline, pass_rng)
    
    # Ensure blocks are sorted and distinct
    if best_blocks:
        # Remove duplicates
        seen = set()
        unique_blocks = []
        for b in best_blocks:
            key = tuple(sorted(b))
            if key not in seen:
                seen.add(key)
                unique_blocks.append(list(key))
        best_blocks = unique_blocks
    
    return best_blocks
