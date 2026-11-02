import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Precompute all t-subsets (triples)
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Helper to get triples covered by a block
    # We will not precompute all blocks to save memory/time, 
    # but we will generate candidates on the fly.
    
    def get_triples_in_block(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to anchor the candidate search
            anchor_idx = rng.choice(list(uncovered))
            anchor_triple = all_triples[anchor_idx]
            
            # Generate all blocks containing the anchor triple
            # The block must contain the t points of the anchor, 
            # plus k-t additional points from the remaining v-t points.
            remaining_points = [p for p in points if p not in anchor_triple]
            candidates = []
            
            # We want to pick a block that covers the most uncovered triples.
            # Instead of checking all candidates, we can sample or check all.
            # Number of candidates: C(21, 4) = 5985. This is small.
            for add_4 in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor_triple + add_4))
                # Calculate gain: how many uncovered triples are in this block?
                # We need to check which of the C(7,3)=35 triples in this block are in 'uncovered'
                gain = 0
                # Optimization: only count if it's potentially useful
                # We can pre-compute triples for this specific block structure? No, too slow.
                # Just iterate.
                for tri in itertools.combinations(block, t):
                    if triple_to_idx[tri] in uncovered:
                        gain += 1
                
                candidates.append((gain, block))
            
            # Pick the best candidate
            if not candidates:
                break
            best_gain, best_block = max(candidates, key=lambda x: (x[0], rng.random()))
            
            blocks.append(best_block)
            
            # Update uncovered
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(triple_to_idx[tri])
                
        return blocks

    def local_search_deletion(blocks):
        # Compute coverage counts
        cover_count = [0] * n_triples
        for block in blocks:
            for tri in itertools.combinations(block, t):
                cover_count[triple_to_idx[tri]] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                block = blocks[i]
                # Check if any triple in this block has count == 1
                # If min count > 1, we can remove the block
                can_remove = True
                for tri in itertools.combinations(block, t):
                    if cover_count[triple_to_idx[tri]] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove block
                    for tri in itertools.combinations(block, t):
                        cover_count[triple_to_idx[tri]] -= 1
                    blocks.pop(i)
                    improved = True
                    break # Restart the loop as indices shifted
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy passes with different seeds
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        blocks = greedy_construct(i)
        blocks = local_search_deletion(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit the theoretical lower bound or known best, we can stop early
        # Known best for C(24,7,3) is 76.
        if best_len <= 76:
            break

    if best_blocks is None:
        # Fallback: simple greedy if something went wrong or time ran out immediately
        blocks = greedy_construct(0)
        best_blocks = blocks

    return [list(b) for b in best_blocks]
