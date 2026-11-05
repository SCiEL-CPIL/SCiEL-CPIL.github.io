import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    points = range(v)
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Helper to get index of a sorted triple
    def get_idx(triple):
        # triple is a tuple of 3 ints
        # They might not be sorted, so sort them first
        # But in our usage, we generate them sorted or handle it
        pass

    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # 1. Greedy Construction
        counts = [0] * num_triples
        blocks = []
        
        # To speed up finding uncovered triples, we can maintain a set of uncovered indices
        # But scanning 2024 items is fast.
        
        while True:
            # Find an uncovered triple
            uncovered_idx = -1
            for i in range(num_triples):
                if counts[i] == 0:
                    uncovered_idx = i
                    break
            
            if uncovered_idx == -1:
                break
            
            # Get the triple
            target_triple = triples[uncovered_idx]
            
            # Start block with this triple
            block = list(target_triple)
            
            # Add remaining points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                best_points_list = []
                
                for x in points:
                    if x in block:
                        continue
                    
                    # Calculate gain
                    gain = 0
                    # Pairs in current block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            u = block[i]
                            v = block[j]
                            # Triple (u, v, x)
                            # Sort to get canonical form
                            tri = tuple(sorted((u, v, x)))
                            idx = triple_to_idx[tri]
                            if counts[idx] == 0:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_points_list = [x]
                    elif gain == best_gain and gain > 0:
                        best_points_list.append(x)
                    
                if best_points_list:
                    # Pick randomly from best candidates
                    block.append(random.choice(best_points_list))
                else:
                    # No point increases coverage? This shouldn't happen if we just started with an uncovered triple
                    # But if it does, pick any point to fill the block
                    # Actually, if gain is 0 for all, it means adding any point doesn't cover new triples.
                    # We still need to fill the block to size k.
                    # Pick a random point not in block.
                    candidates = [x for x in points if x not in block]
                    if candidates:
                        block.append(random.choice(candidates))
                    else:
                        break # Should not happen if k <= v

            # Finalize block
            block.sort()
            blocks.append(block)
            
            # Update counts
            for tri in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(tri)]
                counts[idx] += 1

        # 2. Local Search (Deletion)
        # Repeat until no change
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow removal
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                is_essential = False
                for tri in itertools.combinations(b, t):
                    idx = triple_to_idx[tuple(tri)]
                    if counts[idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove block
                    for tri in itertools.combinations(b, t):
                        idx = triple_to_idx[tuple(tri)]
                        counts[idx] -= 1
                    blocks.pop(i)
                    changed = True
        
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks] # Deep copy
            
        iteration += 1
        
    return best_blocks
