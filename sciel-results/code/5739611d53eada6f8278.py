import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def time_left():
        return deadline - time.time()
        
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    attempts = 0
    while time_left() > 10:
        attempts += 1
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time_left() < 5:
                break
            
            # Pick a random uncovered t-subset to be the seed of the new block
            # Picking the one with the highest frequency in remaining uncovered might be better?
            # For now, random helps escape local minima.
            seed_subset = random.choice(list(uncovered))
            
            block = list(seed_subset)
            
            # Greedily fill the block
            while len(block) < k:
                best_x = -1
                max_gain = -1
                # Candidate points
                candidates = [x for x in range(v) if x not in block]
                
                # Calculate gain for each candidate
                # Gain is the number of uncovered t-subsets that would be covered by adding x
                # A t-subset is covered if it is a subset of block + {x}
                # Specifically, the new t-subsets are those containing x and t-1 elements from block
                for x in candidates:
                    gain = 0
                    # Generate all combinations of t-1 elements from block
                    # Check if (comb + (x,)) is in uncovered
                    # Optimization: precompute combinations of block of size t-1?
                    # Block size grows, so this is O(k * C(m, t-1))
                    for comb in itertools.combinations(block, t - 1):
                        t_sub = tuple(sorted(comb + (x,)))
                        if t_sub in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for t_sub in itertools.combinations(block, t):
                uncovered.discard(t_sub)
        
        # Reduction phase: remove redundant blocks
        # A block is redundant if all its t-subsets are covered by other blocks
        # This is O(B * C(k, t) * B) which might be slow.
        # Simplified: just check if any t-subset in the block is NOT covered by other blocks.
        # If all are covered, it's redundant.
        
        # Let's do a simple pass:
        # For each block, check if it's necessary.
        # To be safe and fast, we might skip complex reduction if time is tight.
        # But reduction is key to lowering the count.
        
        # Efficient reduction:
        # 1. Count how many times each t-subset is covered.
        # 2. If a t-subset is covered only once, the block covering it is essential.
        # 3. If a t-subset is covered multiple times, we might be able to remove a block.
        
        # Let's implement a simple iterative removal:
        # While True:
        #   found_removal = False
        #   For each block i:
        #     Check if all t-subsets in block i are covered by blocks j != i.
        #     If yes, remove block i, found_removal = True, break.
        #   If not found_removal, break.
        
        # This is O(B^2 * C(k, t)). B is ~300. C(8,5)=56. 300*300*56 = 5M ops. Feasible.
        
        changed = True
        while changed and time_left() > 2:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                redundant = True
                for t_sub in itertools.combinations(blocks[i], t):
                    covered = False
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if t_sub in set(itertools.combinations(blocks[j], t)):
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
            # If we reached the theoretical lower bound or a very good number, we can stop early?
            # Lower bound is 264. If we get close, we can stop.
            if best_count <= 265:
                break

    if best_blocks is None:
        # Fallback to a simple greedy if something went wrong
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            seed_subset = min(uncovered)
            block = list(seed_subset)
            while len(block) < k:
                best_x = -1
                max_gain = -1
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for comb in itertools.combinations(block, t - 1):
                        t_sub = tuple(sorted(comb + (x,)))
                        if t_sub in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for t_sub in itertools.combinations(block, t):
                uncovered.discard(t_sub)
        best_blocks = blocks

    return best_blocks
