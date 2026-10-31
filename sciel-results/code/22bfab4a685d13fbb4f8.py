import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for faster counting? 
    # For v=19, t=4, C(19,4) = 3876. This is small enough to use a dict or set.
    
    def greedy_construct():
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to start the block
            # This adds randomness to the greedy phase
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            # Fill the block with k points
            while len(block) < k:
                best_x = -1
                max_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain: how many uncovered t-subsets are covered by adding x
                    # A t-subset is covered if it is a subset of block + {x}
                    # And it must be in uncovered
                    gain = 0
                    # We need to check all t-subsets of (block U {x})
                    # But only those that include x? No, any t-subset of the new block.
                    # However, t-subsets not containing x were already subsets of the old block.
                    # Since we only add x, the new subsets covered MUST contain x.
                    # So we only check t-subsets of (block U {x}) that contain x.
                    # These are formed by choosing t-1 points from block and adding x.
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                        if max_gain == 0:
                            # If no gain, break early? No, we must pick something.
                            # But if all gains are 0, we are stuck? 
                            # Actually, if uncovered is not empty, there must be some gain eventually?
                            # Not necessarily for a specific x. But we pick max.
                            pass
                if best_x == -1:
                    break # Should not happen if uncovered is not empty and k <= v
                block.append(best_x)
                # Update uncovered
                for sub in itertools.combinations(block, t):
                    uncovered.discard(tuple(sub))
            
            block.sort()
            blocks.append(block)
        return blocks

    def local_optimize(blocks):
        # Use cover counts to efficiently check redundancy
        # cover_count[s] = number of blocks containing s
        cover_count = {s: 0 for s in all_t_subsets}
        for b in blocks:
            for s in itertools.combinations(b, t):
                cover_count[tuple(s)] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for s in itertools.combinations(b, t):
                    if cover_count[tuple(s)] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in itertools.combinations(b, t):
                        cover_count[tuple(s)] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks

    # Initial solution
    best_blocks = greedy_construct()
    best_blocks = local_optimize(best_blocks)
    best_size = len(best_blocks)
    
    current_blocks = list(best_blocks)
    current_size = best_size
    
    # ILS Loop
    no_improve_count = 0
    max_no_improve = 500 # Restart if stuck
    
    while time.time() - start_time < time_limit:
        # Perturb current_blocks
        # Pick a random block and swap a point with a random outside point
        if current_size == 0:
            break
            
        new_blocks = [list(b) for b in current_blocks]
        
        # Try a few perturbations if the first one doesn't help? 
        # Or just one per iteration.
        b_idx = random.randint(0, len(new_blocks) - 1)
        b = new_blocks[b_idx]
        
        # Pick a point to remove
        p_idx = random.randint(0, k - 1)
        p_out = b[p_idx]
        
        # Pick a point to add
        candidates = [x for x in range(v) if x not in b]
        if not candidates:
            continue
        p_in = random.choice(candidates)
        
        b[p_idx] = p_in
        b.sort()
        
        # Local Optimize
        opt_blocks = local_optimize(new_blocks)
        opt_size = len(opt_blocks)
        
        if opt_size < current_size:
            current_blocks = opt_blocks
            current_size = opt_size
            no_improve_count = 0
            
            if current_size < best_size:
                best_blocks = current_blocks
                best_size = current_size
                
        elif opt_size == current_size:
            # Accept with some probability to explore
            if random.random() < 0.1:
                current_blocks = opt_blocks
                # current_size stays same
                no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart from greedy
            current_blocks = greedy_construct()
            current_blocks = local_optimize(current_blocks)
            current_size = len(current_blocks)
            no_improve_count = 0
            
            if current_size < best_size:
                best_blocks = current_blocks
                best_size = current_size

    return [list(b) for b in best_blocks]
