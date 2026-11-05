import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_k_blocks:
        tids = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triples.append(tids)
    
    # Precompute numpy arrays for block_triples for faster indexing
    block_triples_np = [np.array(ts, dtype=np.int32) for ts in block_triples]
    
    # Map each triple to the list of block indices that cover it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)
    
    # Convert triple_blocks to numpy arrays for vectorized updates
    triple_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_blocks]

    def greedy_construct_fast(seed):
        rng = random.Random(seed)
        # gain[b] = number of currently uncovered triples that block b covers
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > deadline - 3:
                # Emergency: finish with a fast greedy ignoring tie-breaking randomness
                # Just pick first max-gain block each time
                while is_uncovered.any():
                    max_gain = np.max(gain)
                    if max_gain == 0:
                        break
                    candidates = np.where(gain == max_gain)[0]
                    b_idx = int(candidates[0])
                    selected_blocks.append(b_idx)
                    for t_idx in block_triples[b_idx]:
                        if is_uncovered[t_idx]:
                            is_uncovered[t_idx] = False
                            gain[triple_blocks_np[t_idx]] -= 1
                break
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            b_idx = int(rng.choice(candidates))
            
            selected_blocks.append(b_idx)
            
            # Update: for each newly-covered triple, decrement gain of all blocks covering it
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    gain[triple_blocks_np[t_idx]] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        """Remove redundant blocks: a block is removable if every triple it covers
        is also covered by at least one other selected block."""
        if not selected_blocks:
            return []
        
        n = len(selected_blocks)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            cover_count[block_triples_np[b_idx]] += 1
        
        active = np.ones(n, dtype=np.bool_)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not active[i]:
                    continue
                tids = block_triples_np[selected_blocks[i]]
                # If all triples in this block have count > 1, it's redundant
                if np.min(cover_count[tids]) > 1:
                    active[i] = False
                    cover_count[tids] -= 1
                    changed = True
        
        return [selected_blocks[i] for i in range(n) if active[i]]

    def local_search_swap(cover, max_time=5.0):
        """Try to swap one block for another to reduce total block count or improve structure.
        Strategy: for each block in the cover, try removing it and re-covering the exposed
        triples with the fewest additional blocks. If the net change is negative, accept."""
        if len(cover) <= 1:
            return cover
        
        start_ls = time.time()
        cover = list(cover)
        
        # Compute cover count
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            cover_count[block_triples_np[b_idx]] += 1
        
        improved = True
        while improved:
            improved = False
            if time.time() > start_ls + max_time:
                break
            
            for i in range(len(cover)):
                if time.time() > start_ls + max_time:
                    break
                
                b_idx = cover[i]
                tids = block_triples_np[b_idx]
                
                # Which triples become uncovered if we remove this block?
                exposed = tids[cover_count[tids] == 1]
                
                if len(exposed) == 0:
                    # Block is redundant, remove it
                    cover_count[tids] -= 1
                    cover.pop(i)
                    improved = True
                    break
                
                # Try to cover exposed triples with a single new block
                # Find blocks that cover all exposed triples
                # exposed has at most 35 elements
                # A block covers all exposed if all exposed triples are in the block
                # This is expensive to check for all ~346k blocks.
                # Instead: find candidate blocks that contain at least one point from each exposed triple.
                # Simpler: just check if there's a block that contains all points in the union of exposed triples.
                # Actually, a block covers a triple if it contains all 3 points.
                # So we need a block that contains all points appearing in any exposed triple.
                
                # Get the set of points in exposed triples
                exposed_points = set()
                for tr in exposed:
                    exposed_points.update(triples[tr])
                
                if len(exposed_points) > k:
                    continue  # No single block can cover all exposed points
                
                # Find blocks containing all exposed_points
                # This is still expensive. Skip for now.
                # Instead, try a simpler heuristic: if removing one block and adding two is beneficial, do it.
                # But that's more complex. For now, just do the redundant removal pass.
        
        return cover

    best_cover = None
    seed = 0
    
    # Run multiple greedy attempts
    while time.time() < deadline - 10:
        # Run greedy
        cover = greedy_construct_fast(seed)
        
        # Reduce
        cover = reduce_cover(cover)
        
        # Local search: try to find more reductions
        if time.time() < deadline - 15:
            cover = local_search_swap(cover, max_time=3.0)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
        
        seed += 1

    if best_cover is None:
        best_cover = greedy_construct_fast(0)
        best_cover = reduce_cover(best_cover)
        
    return [list(all_k_blocks[i]) for i in best_cover]
