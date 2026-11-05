import itertools
import random
import time

# Precomputation for v=19, k=7, t=4
# These are specific to the task parameters, but construct takes v,k,t.
# We should probably handle generic v,k,t or at least v=19,k=7,t=4 efficiently.
# The task says "evaluator passes this task's parameters v, k and t".
# So we must handle v=19, k=7, t=4.
# We can check if v==19, k==7, t==4 and use the optimized path.
# Otherwise, fallback to a simpler greedy?
# The task contract implies v=19, k=7, t=4 is the specific instance.
# I will assume v=19, k=7, t=4 for the optimization, but keep a generic fallback.

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback: Simple Greedy
        return _generic_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Precompute for v=19, k=7, t=4
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets) # 3876
    
    all_blocks = list(itertools.combinations(points, k))
    block_to_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_subsets.append(subs)
        
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)

    best_blocks = None
    best_len = float('inf')
    
    def greedy_random():
        uncovered = set(range(num_subsets))
        blocks = []
        while uncovered:
            # Pick random uncovered subset
            s = random.choice(list(uncovered))
            # Pick random block covering it
            candidates = subset_to_blocks[s]
            b_idx = random.choice(candidates)
            blocks.append(b_idx)
            # Remove covered subsets
            for sub in block_to_subsets[b_idx]:
                if sub in uncovered:
                    uncovered.discard(sub)
        return blocks

    def local_search(blocks):
        # blocks: list of block indices
        # cover_count: array of size num_subsets
        cover_count = [0] * num_subsets
        for b_idx in blocks:
            for s_idx in block_to_subsets[b_idx]:
                cover_count[s_idx] += 1
        
        blocks_set = set(blocks)
        changed = True
        while changed:
            changed = False
            # Remove redundant blocks
            for b_idx in list(blocks):
                # Check if all subsets in b_idx have count > 1
                redundant = True
                for s_idx in block_to_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    for s_idx in block_to_subsets[b_idx]:
                        cover_count[s_idx] -= 1
                    blocks.remove(b_idx)
                    blocks_set.discard(b_idx)
                    changed = True
                    if time.time() > deadline:
                        return blocks
            
            # If no removals, try swaps
            if not changed:
                swapped = False
                for b_old in list(blocks):
                    # Find critical subsets (count == 1)
                    crit = [s_idx for s_idx in block_to_subsets[b_old] if cover_count[s_idx] == 1]
                    if not crit:
                        continue
                    
                    # Find candidate b_new that covers all crit
                    # b_new must contain all s in crit
                    # We can iterate through blocks covering the first crit subset
                    candidates = subset_to_blocks[crit[0]]
                    for b_new in candidates:
                        if b_new in blocks_set:
                            continue
                        # Check if b_new covers all crit
                        valid = True
                        b_new_subs = set(block_to_subsets[b_new])
                        for s in crit:
                            if s not in b_new_subs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Perform swap
                        # Remove b_old
                        for s_idx in block_to_subsets[b_old]:
                            cover_count[s_idx] -= 1
                        # Add b_new
                        for s_idx in block_to_subsets[b_new]:
                            cover_count[s_idx] += 1
                        
                        blocks_set.discard(b_old)
                        blocks_set.add(b_new)
                        blocks[blocks.index(b_old)] = b_new
                        
                        # Try to remove blocks
                        removed_any = False
                        for b_check in list(blocks):
                            redundant = True
                            for s_idx in block_to_subsets[b_check]:
                                if cover_count[s_idx] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                for s_idx in block_to_subsets[b_check]:
                                    cover_count[s_idx] -= 1
                                blocks.remove(b_check)
                                blocks_set.discard(b_check)
                                removed_any = True
                                if time.time() > deadline:
                                    return blocks
                        
                        if removed_any:
                            swapped = True
                            break # Restart swap search with new state
                        else:
                            # Revert swap
                            for s_idx in block_to_subsets[b_new]:
                                cover_count[s_idx] -= 1
                            for s_idx in block_to_subsets[b_old]:
                                cover_count[s_idx] += 1
                            blocks_set.discard(b_new)
                            blocks_set.add(b_old)
                            blocks[blocks.index(b_new)] = b_old
                            
                if swapped:
                    changed = True # Trigger outer loop again to check for more removals
                else:
                    # Stuck in local minimum. 
                    # Could try more aggressive moves, but for now break.
                    break
                    
        return blocks

    # Run multiple iterations
    while time.time() < deadline:
        blocks = greedy_random()
        blocks = local_search(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            # If we reached 151, we can stop? 151 is best known.
            if best_len <= 151:
                break

    if best_blocks is None:
        # Fallback
        best_blocks = greedy_random()
        best_blocks = local_search(best_blocks)

    # Convert indices to lists of ints
    result = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return result

def _generic_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        best_block = None
        max_covered = -1
        for b in itertools.combinations(range(v), k):
            covered = sum(1 for s in itertools.combinations(b, t) if s in uncovered)
            if covered > max_covered:
                max_covered = covered
                best_block = b
        if best_block is None:
            break
        block = list(best_block)
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
