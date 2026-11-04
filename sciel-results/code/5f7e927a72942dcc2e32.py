import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute triple IDs for faster access?
    # id_cache[x][a][b] is too big. 
    # Let's just use subset_to_id.
    
    def get_id(subset):
        return subset_to_id[tuple(subset)]
        
    best_blocks = None
    best_count = float('inf')
    
    # We will run many restarts
    # Estimate: 1 restart takes ~5-10ms. 115s / 10ms = 11500 restarts.
    # Let's cap at 5000 to be safe.
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Randomize the process
        # 1. Randomize point order for tie-breaking
        # 2. Randomize first point selection among top candidates
        
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered:
            # Select first point
            # Count involvement in uncovered triples
            point_counts = [0] * v
            for uid in uncovered:
                s = all_t_subsets[uid]
                for p in s:
                    point_counts[p] += 1
            
            max_count = max(point_counts)
            if max_count == 0:
                break # Should not happen if uncovered is not empty
            
            candidates = [i for i in range(v) if point_counts[i] == max_count]
            first_point = random.choice(candidates)
            
            block = [first_point]
            block_set = {first_point}
            
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                # Current pairs in block
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for pair in block_pairs:
                        # Construct triple
                        # Sort to match dictionary key
                        triple = tuple(sorted((x, pair[0], pair[1])))
                        uid = subset_to_id[triple]
                        if uid in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_gain <= 0:
                    # No point adds coverage? This implies the block cannot cover any new triples?
                    # But we started with a point in uncovered triples.
                    # If we are stuck, we might need to add points even if gain is 0?
                    # No, if gain is 0, adding x doesn't cover any *new* triples involving x and existing block.
                    # However, x might be needed to form triples with *future* points?
                    # The greedy method usually assumes we build the block to maximize immediate coverage.
                    # If immediate gain is 0, we should probably pick a point that has high potential for future?
                    # Or just pick randomly to complete the block.
                    # Given the structure, if we started with a good point, gain should be > 0 until block is full or coverage is done.
                    # If gain is 0, we pick a random point to fill the block.
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    x = random.choice(remaining)
                else:
                    x = random.choice(best_points)
                
                block.append(x)
                block_set.add(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uid = subset_to_id[tuple(sub)]
                if uid in uncovered:
                    uncovered.remove(uid)
        
        # Local Search: Remove redundant blocks
        # Repeat until no change
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, b in enumerate(blocks):
                # Check if b covers any triple not covered by other blocks
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    uid = subset_to_id[tuple(sub)]
                    # Check if any other block covers this sub
                    covered_by_other = False
                    for j, other_b in enumerate(blocks):
                        if i == j:
                            continue
                        # Check if sub is in other_b
                        # Efficient check: set intersection or subset
                        # Since t is small, just check if all elements of sub are in other_b
                        if all(p in other_b for p in sub):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                
                if not is_redundant:
                    new_blocks.append(b)
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                changed = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [b[:] for b in blocks]
        
        if best_blocks is None or len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
            if best_count <= 76:
                break

    if best_blocks is None:
        # Fallback: generate a simple greedy without restarts
        # This should not happen
        uncovered = set(range(n_t_subsets))
        blocks = []
        while uncovered:
            block = [0]
            block_set = {0}
            while len(block) < k:
                best_gain = -1
                best_point = 1
                block_pairs = list(itertools.combinations(block, t - 1))
                for x in range(1, v):
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in block_pairs:
                        triple = tuple(sorted((x, pair[0], pair[1])))
                        if subset_to_id[triple] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uid = subset_to_id[tuple(sub)]
                if uid in uncovered:
                    uncovered.remove(uid)
        best_blocks = blocks
        
    return best_blocks
