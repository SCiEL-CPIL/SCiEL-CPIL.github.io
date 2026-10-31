import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: Greedy construction
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered and time_left() > 5:
        # Pick the first uncovered t-subset to anchor the block
        anchor = min(uncovered)
        block = list(anchor)
        
        while len(block) < k:
            # For each candidate point, compute gain
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
        
        block.sort()
        block_set = set(block)
        blocks.append(block)
        # Remove covered t-subsets
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    if not uncovered:
        # Verify and return
        return blocks
    
    # Phase 2: Local improvement - try to improve blocks
    # Build coverage count for each t-subset
    def build_coverage(blocks):
        cov = {}
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                cov[ts] = cov.get(ts, 0) + 1
        return cov
    
    # Try to remove redundant blocks
    def try_remove_redundant(blocks):
        changed = True
        while changed:
            changed = False
            cov = build_coverage(blocks)
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cov.get(ts, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
        return blocks
    
    blocks = try_remove_redundant(blocks)
    
    # Try to improve individual blocks by swapping points
    def try_improve_blocks(blocks, time_limit=5):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            uncovered = set(all_tsubs)
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    uncovered.discard(ts)
            
            if not uncovered:
                break
            
            for i in range(len(blocks)):
                if time.time() >= time_limit:
                    break
                b = blocks[i]
                b_set = set(b)
                # Try swapping each point in b with each point not in b
                best_swap = None
                best_gain = 0
                for j, old_x in enumerate(b):
                    for new_x in range(v):
                        if new_x in b_set:
                            continue
                        # Compute new block
                        new_b = b[:]
                        new_b[j] = new_x
                        new_b.sort()
                        new_b_set = set(new_b)
                        # Count how many currently uncovered t-subs this new block covers
                        gain = 0
                        for ts in itertools.combinations(new_b, t):
                            if ts in uncovered:
                                gain += 1
                        # Also count how many t-subs the old block covered that the new block doesn't
                        # (i.e., t-subs unique to old block that would become uncovered)
                        loss = 0
                        cov = {}
                        for bb in blocks:
                            for ts in itertools.combinations(bb, t):
                                cov[ts] = cov.get(ts, 0) + 1
                        for ts in itertools.combinations(b, t):
                            if cov.get(ts, 0) == 1 and ts not in set(itertools.combinations(new_b, t)):
                                loss += 1
                        net = gain - loss
                        if net > best_gain:
                            best_gain = net
                            best_swap = (j, old_x, new_x)
                
                if best_swap and best_gain > 0:
                    j, old_x, new_x = best_swap
                    b[j] = new_x
                    b.sort()
                    improved = True
    
    if time_left() > 10:
        try_improve_blocks(blocks, time_limit=start_time + time_left() - 10)
    
    blocks = try_remove_redundant(blocks)
    
    # Final check: ensure all t-subsets are covered
    # If not, we need to add blocks back
    uncovered = set(all_tsubs)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            uncovered.discard(ts)
    
    if uncovered:
        # Add greedy blocks for remaining uncovered
        while uncovered and time_left() > 1:
            anchor = min(uncovered)
            block = list(anchor)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
    
    return blocks
