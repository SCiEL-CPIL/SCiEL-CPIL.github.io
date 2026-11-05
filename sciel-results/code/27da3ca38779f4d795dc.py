import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of attempts based on time
    # Each attempt is fast (<0.1s), so we can do many.
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        random.seed(attempts) # Deterministic randomization for reproducibility in eval? 
        # Actually, just using random without seed is fine, but deterministic is safer for debugging.
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 10:
                break
                
            # Pick random seed from uncovered
            # Using random.choice on a set is O(n). 
            # To make it O(1), we can keep a list of uncovered and shuffle?
            # Or just pick the first one, but shuffle the list of all subsets initially?
            # Simpler: convert to list, pick random index.
            # But uncovered changes.
            # Let's just pick the first one for now, but vary the order of points considered?
            # Or, pick a random subset from the current uncovered set.
            
            # Efficient random selection from set:
            # Since we don't know the size, and it changes, 
            # let's just iterate and pick the first one, but we can randomize the order of points in gain calculation?
            # No, the seed selection is the key variation.
            
            # To pick a random element from a set in Python:
            # random.choice(list(uncovered)) is O(n).
            # Given n=3876, this is fine.
            
            if not uncovered:
                break
                
            # Optimization: If uncovered is small, just pick one.
            # For better diversity, pick randomly.
            seed = random.choice(list(uncovered))
            
            block = list(seed)
            
            while len(block) < k:
                if time.time() > deadline - 10:
                    break
                    
                best_x = -1
                best_gain = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                block.append(best_x)
                
            block.sort()
            
            for s in itertools.combinations(block, t):
                uncovered.discard(s)
                
            blocks.append(block)
            
        # Cleanup
        if time.time() < deadline - 5:
            block_subsets = [set(itertools.combinations(b, t)) for b in blocks]
            for _ in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                removed = False
                for i in range(len(blocks)):
                    if time.time() > deadline - 5:
                        break
                    is_redundant = True
                    for s in block_subsets[i]:
                        covered = False
                        for j in range(len(blocks)):
                            if i == j:
                                continue
                            if s in block_subsets[j]:
                                covered = True
                                break
                        if not covered:
                            is_redundant = False
                            break
                    if is_redundant:
                        blocks.pop(i)
                        block_subsets.pop(i)
                        removed = True
                        break
                if not removed:
                    break

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
        # Early stop if we hit the lower bound (131) or known best (151)
        if best_count <= 151:
            break

    return best_blocks if best_blocks else []
