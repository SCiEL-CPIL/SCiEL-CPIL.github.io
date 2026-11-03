import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 110
    
    all_triples = list(itertools.combinations(range(v), t))
    # Map triple to index for faster array access?
    # Or just use dict. Dict is fine for 2024 items.
    
    best_blocks = None
    best_len = float('inf')
    
    max_attempts = 30 # Increase attempts since it's faster
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        trip_count = {trip: 0 for trip in all_triples}
        uncovered_count = len(all_triples)
        current_blocks = []
        current_block_triplets = [] # List of sets of triples for each block
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            # Find an uncovered triple
            # To add randomness, we can pick randomly from the uncovered ones
            # But finding them takes O(N).
            # Let's just pick the first one found, or randomize the order of all_triples?
            # Randomizing 2024 items is fast.
            random.shuffle(all_triples) # This modifies the list, so we need a copy or re-shuffle each time?
            # Better: pick a random index, check if uncovered. If not, try another.
            # Or just:
            target = None
            for trip in all_triples:
                if trip_count[trip] == 0:
                    target = trip
                    break
            
            if target is None:
                break
                
            target_set = set(target)
            remaining_points = [x for x in range(v) if x not in target_set]
            need = k - t
            
            best_gain = -1
            best_block = None
            best_block_triplets_set = None
            
            for comb in itertools.combinations(remaining_points, need):
                block = tuple(sorted(target + comb))
                block_triplets_set = set(itertools.combinations(block, t))
                
                # Count uncovered triples in this block
                gain = 0
                for trip in block_triplets_set:
                    if trip_count[trip] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_block_triplets_set = block_triplets_set
                    # Early break if gain is max possible? Max is 35.
                    if gain == 35:
                        break
            
            if best_block is None:
                break
                
            current_blocks.append(best_block)
            current_block_triplets.append(best_block_triplets_set)
            
            # Update counts
            for trip in best_block_triplets_set:
                trip_count[trip] += 1
                if trip_count[trip] == 1:
                    uncovered_count -= 1
            
        # Local Search
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(current_blocks)):
                block_triplets_set = current_block_triplets[i]
                # Check if all triples are covered by others
                # i.e., trip_count[trip] > 1 for all trip in block_triplets_set
                is_redundant = True
                for trip in block_triplets_set:
                    if trip_count[trip] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block i
                    current_blocks.pop(i)
                    current_block_triplets.pop(i)
                    for trip in block_triplets_set:
                        trip_count[trip] -= 1
                    changed = True
                    break
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks
            
    return best_blocks
