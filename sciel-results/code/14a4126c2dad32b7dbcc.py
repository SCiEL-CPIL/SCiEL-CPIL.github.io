import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using randomized greedy search 
    followed by a redundancy removal pass.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    # Precompute all t-subsets for redundancy check efficiency if needed, 
    # but for now we compute on the fly or use sets.
    
    while time.time() < deadline:
        blocks = randomized_greedy(v, k, t)
        
        # Refine by removing redundant blocks
        blocks = remove_redundant_blocks(blocks, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # If we reach a very low number, we might stop early, 
            # but let's keep searching for a bit to see if we can improve.
            # The lower bound is 264. If we get close, we might want to 
            # spend more time on local search, but for now, simple iteration is safe.
            
    if best_blocks is None:
        # Fallback to a single deterministic run if something went wrong or time was too short
        best_blocks = randomized_greedy(v, k, t)
        best_blocks = remove_redundant_blocks(best_blocks, v, k, t)
        
    return best_blocks

def randomized_greedy(v, k, t):
    """
    Greedy covering algorithm with randomization in the selection of 
    the first uncovered subset and tie-breaking.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to start the new block
        # Converting to list is O(N), N=11628, which is fast.
        start_subset = random.choice(tuple(uncovered))
        block = list(start_subset)
        
        while len(block) < k:
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block:
                    continue
                
                # Calculate gain: how many uncovered t-subsets would be covered
                # by adding x to the current block?
                # The new subsets covered are those of size t that include x 
                # and t-1 elements from the current block.
                gain = 0
                # t-1 combinations from current block
                for rest in itertools.combinations(block, t - 1):
                    # Create the new subset
                    # rest is a tuple of t-1 elements
                    # x is an int
                    # We need to check if the set of rest + {x} is in uncovered
                    # To avoid sorting every time, we can use a frozenset or tuple sorted
                    # Since v is small, sorting is cheap, but we can optimize.
                    # rest is sorted? combinations() yields sorted tuples.
                    # x might not be in sorted position.
                    # Let's just make a tuple and sort it.
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            # Pick randomly among the best candidates to introduce diversity
            if best_candidates:
                block.append(random.choice(best_candidates))
            else:
                # Should not happen if uncovered is not empty and we pick from uncovered
                # But just in case, pick any unused point
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
        
        block.sort()
        blocks.append(block)
        
        # Remove covered subsets from uncovered
        for sub in itertools.combinations(block, t):
            if sub in uncovered:
                uncovered.remove(sub)
                
    return blocks

def remove_redundant_blocks(blocks, v, k, t):
    """
    Removes blocks that do not cover any unique t-subset.
    """
    if not blocks:
        return blocks
    
    # Count how many times each t-subset is covered
    cover_counts = {}
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_counts[sub] = cover_counts.get(sub, 0) + 1
            
    changed = True
    current_blocks = blocks
    while changed:
        changed = False
        new_blocks = []
        for block in current_blocks:
            # Check if this block is essential
            # It is essential if there is at least one subset in it 
            # that is covered exactly once (by this block)
            is_essential = False
            for sub in itertools.combinations(block, t):
                if cover_counts.get(sub, 0) == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
            else:
                # Redundant, remove it
                changed = True
                for sub in itertools.combinations(block, t):
                    cover_counts[sub] -= 1
                    # Optional: clean up dict if 0, but not necessary for correctness
                    # if cover_counts[sub] == 0:
                    #     del cover_counts[sub]
                    
        current_blocks = new_blocks
        
    return current_blocks
