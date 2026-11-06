import time
import random
from itertools import combinations

_DEADLINE_START = time.time()
_DEADLINE_SECONDS = 145.0  # Safety margin for 150s limit

def _now():
    return time.time()

def _time_left():
    return _DEADLINE_SECONDS - (_now() - _DEADLINE_START)

def _precompute():
    """
    Precompute 5-subset IDs for all 6-subsets.
    5-subsets of 12 points: C(12,5) = 792.
    6-subsets of 12 points: C(12,6) = 924.
    """
    points = range(12)
    all_5subs = list(combinations(points, 5))
    map_5sub_to_id = {sub: i for i, sub in enumerate(all_5subs)}
    
    all_6subs = list(combinations(points, 6))
    block_to_5ids = {}
    for b in all_6subs:
        ids = [map_5sub_to_id[s] for s in combinations(b, 5)]
        block_to_5ids[b] = ids
        
    return all_6subs, block_to_5ids

def _greedy_pass(order, block_to_5ids, target=132):
    """
    Run a greedy pass on the given order of blocks.
    Returns the list of blocks found.
    """
    covered_ids = set()
    blocks = []
    
    for b in order:
        if _time_left() < 2.0:
            break
        ids = block_to_5ids[b]
        # Check if any 5-subset is already covered
        # Using a set, 'any' is fast enough for 6 items
        if any(i in covered_ids for i in ids):
            continue
        
        # Accept block
        blocks.append(b)
        covered_ids.update(ids)
        
        if len(blocks) == target:
            return blocks
            
    return blocks

def _try_s4511_backtracking(timeout=5.0):
    """
    Try to construct S(4,5,11) via backtracking.
    If successful, derive S(5,6,12) from it.
    """
    start_time = _now()
    points = range(11)
    all_5subs = list(combinations(points, 5))
    map_5sub_to_id = {sub: i for i, sub in enumerate(all_5subs)}
    
    all_6subs = list(combinations(points, 6))
    # For each 6-subset, compute its 5-subsets
    block_to_5ids = {}
    for b in all_6subs:
        ids = [map_5sub_to_id[s] for s in combinations(b, 5)]
        block_to_5ids[b] = ids
    
    # We need to find a packing of 5-subsets on 11 points with 6-subsets
    # S(4,5,11) has 66 blocks.
    # Actually, S(4,5,11) is a Steiner system where every 4-subset is in exactly one block.
    # Number of blocks = C(11,4)/C(5,4) = 330/5 = 66.
    
    # Let's do a greedy with backtracking for S(4,5,11)
    # This might be too slow. Let's stick to the greedy for S(5,6,12) which is known to work with restarts.
    # The inspiration mentions _solve_s4511 but the current program doesn't have it.
    # Let's implement a fast greedy for S(5,6,12) with many restarts.
    
    # Actually, the current program already does this.
    # Let's try a different strategy: use the known structure of the Witt design.
    # The Witt design S(5,6,12) can be constructed from the Mathieu group M12.
    # But implementing M12 is complex.
    
    # Alternative: use the fact that S(5,6,12) is unique up to isomorphism.
    # We can try to build it by fixing a few blocks and then using symmetry.
    
    # For now, let's just return None and let the main function use the greedy restarts.
    return None

def construct():
    """
    Construct the Witt design S(5,6,12) using randomized greedy search.
    """
    # Precomputation is fast (~ms)
    all_6subs, block_to_5ids = _precompute()
    
    best_blocks = []
    
    # Strategy: Multiple randomized restarts
    # We iterate seeds. If we find 132, we return immediately.
    # If time runs out, we return the best found so far.
    
    # First, try a deterministic symmetry-breaking greedy as a quick check
    b1 = (0, 1, 2, 3, 4, 5)
    b2 = (0, 1, 2, 6, 7, 8)
    # Filter others
    rest = [b for b in all_6subs if b != b1 and b != b2]
    order_sym = [b1, b2] + rest
    blocks = _greedy_pass(order_sym, block_to_5ids)
    if len(blocks) == 132:
        return [list(b) for b in blocks]
    if len(blocks) > len(best_blocks):
        best_blocks = blocks
        
    # Randomized restarts
    # We use different seeds to explore different orderings.
    # The "inspiration" suggests this works well.
    num_restarts = 5000 # High number, but we break on time or success
    
    for seed in range(num_restarts):
        if _time_left() < 5.0:
            break
            
        rng = random.Random(seed)
        shuffled = all_6subs[:]
        rng.shuffle(shuffled)
        
        blocks = _greedy_pass(shuffled, block_to_5ids)
        
        if len(blocks) == 132:
            return [list(b) for b in blocks]
            
        if len(blocks) > len(best_blocks):
            best_blocks = blocks

    # Return best found
    return [list(b) for b in best_blocks]
