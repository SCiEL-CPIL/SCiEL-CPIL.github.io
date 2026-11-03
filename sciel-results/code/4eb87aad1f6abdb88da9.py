"""
Covering design C(24,7,3) using a bit-mask greedy algorithm.
"""
import itertools
import time


def _precompute_point_masks(v, t):
    """
    For each point, precompute a bitmask of all t-subsets that contain that point.
    """
    triples = list(itertools.combinations(range(v), t))
    triple_index = {triple: i for i, triple in enumerate(triples)}
    n_triples = len(triples)
    
    point_masks = [0] * v
    for i in range(v):
        mask = 0
        for j in range(n_triples):
            if triples[j][0] == i or triples[j][1] == i or triples[j][2] == i:
                mask |= (1 << j)
        point_masks[i] = mask
    
    return point_masks, triple_index


def _precompute_block_masks(v, k, t, point_masks, triple_index):
    """
    Precompute the triple mask for every possible k-block.
    This is expensive but allows very fast greedy selection.
    """
    n_blocks = 0
    for combo in itertools.combinations(range(v), k):
        n_blocks += 1
    
    # Instead of storing all block masks (which can be huge), we can compute them on the fly
    # or use a smarter approach. Let's compute them lazily if needed.
    # For C(24,7,3), there are C(24,7) = 346104 possible blocks.
    # Storing 346104 ints of 2024 bits each is about 346104 * 253 bytes = ~87 MB.
    # This is feasible.
    
    all_block_masks = {}
    for combo in itertools.combinations(range(v), k):
        mask = 0
        for point in combo:
            mask |= point_masks[point]
        all_block_masks[combo] = mask
    
    return all_block_masks


def _greedy_cover_bitmask(v, k, t, deadline):
    """
    Bit-mask based greedy covering.
    """
    point_masks, triple_index = _precompute_point_masks(v, t)
    n_triples = len(triple_index)
    all_triples_mask = (1 << n_triples) - 1
    
    # Precompute block masks
    all_block_masks = _precompute_block_masks(v, k, t, point_masks, triple_index)
    
    uncovered_mask = all_triples_mask
    blocks = []
    block_masks = []
    
    while uncovered_mask and time.time() < deadline:
        # Find the block that covers the most uncovered triples
        best_block = None
        best_gain = -1
        
        for block, mask in all_block_masks.items():
            gain = bin(uncovered_mask & mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None or best_gain == 0:
            break
        
        blocks.append(list(best_block))
        block_masks.append(all_block_masks[tuple(best_block)])
        uncovered_mask &= ~all_block_masks[tuple(best_block)]
    
    return blocks


def _local_search_sa(blocks, v, k, t, time_limit):
    """
    Simple local search to try to reduce the number of blocks.
    """
    # This is a placeholder for now; the greedy algorithm should be strong enough.
    return blocks


def construct(v, k, t):
    """
    Construct a covering design for the given parameters.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds buffer
    
    # For small v, we can use exact methods, but for v=24, we need heuristics
    blocks = _greedy_cover_bitmask(v, k, t, deadline)
    
    # Simple local search to try to improve
    if time.time() < deadline - 5:
        blocks = _local_search_sa(blocks, v, k, t, deadline)
    
    return blocks
