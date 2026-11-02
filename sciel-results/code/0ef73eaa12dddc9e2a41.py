import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using randomized greedy restarts.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Pre-compute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    total_subsets = len(all_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Set a limit on restarts to avoid infinite loops if it's too fast, 
    # but primarily rely on the time deadline.
    max_restarts = 10000 
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # Using list(uncovered)[random_index] is O(N) to create list.
            # Better: use next(iter(uncovered)) for deterministic start, 
            # but we want randomization. 
            # For small sets, list conversion is fine.
            # Or we can maintain a list of uncovered items for random access.
            # Let's stick to set for O(1) removal, but pick random start.
            
            # Optimization: If we pick a random start, we need a random element from the set.
            # random.choice(list(uncovered)) is expensive if uncovered is large.
            # However, the size decreases. 
            # Alternative: Always start with the first uncovered one (sorted), 
            # but randomize the *addition* of points.
            # Actually, randomizing the start point is crucial for diversity.
            # Let's use a list for uncovered to allow O(1) random access and removal?
            # Removal from list is O(N). 
            # Given N=2024, O(N) removal is 2024 ops. 76 blocks * 35 removals = 2660 removals * 2024 = 5M ops.
            # This might be slower than set.
            
            # Let's stick to set for uncovered.
            # To pick a random element:
            # random.choice is not available for sets.
            # We can do: next(iter(uncovered)) which is arbitrary but fast.
            # To make it random, we can hash or use a random index into a sorted list?
            # Let's just use next(iter(uncovered)) for the start, and randomize the greedy choice.
            # This is simpler and often sufficient.
            
            start_sub = next(iter(uncovered))
            block = list(start_sub)
            
            # Greedily fill the block
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check all pairs in current block
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            triple = tuple(sorted((block[i], block[j], x)))
                            if triple in uncovered:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if uncovered is not empty and v > k
                    break
                
                # Pick randomly from best candidates to introduce diversity
                next_point = random.choice(best_candidates)
                block.append(next_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            # Remove all t-subsets of the new block
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
            
            # Check if we improved the best result
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
                
                # Early termination if we hit the theoretical lower bound?
                # Lower bound for C(24,7,3) is 69. If we reach 69, we can stop.
                if best_len <= 69:
                    return best_blocks

    if best_blocks is None:
        # Fallback to deterministic greedy if no restarts completed (should not happen)
        uncovered = set(all_subsets)
        blocks = []
        while uncovered:
            start_sub = min(uncovered)
            block = list(start_sub)
            while len(block) < k:
                best_gain = -1
                best_x = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            triple = tuple(sorted((block[i], block[j], x)))
                            if triple in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        best_blocks = blocks
        
    return best_blocks
