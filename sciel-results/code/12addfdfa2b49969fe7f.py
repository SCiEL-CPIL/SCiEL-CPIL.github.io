import itertools
import random
import time

def construct(v, k, t):
    # Precompute t-subsets and their IDs
    t_subs = list(itertools.combinations(range(v), t))
    sub_id = {s: i for i, s in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # If t is 0 or v is 0, handle edge cases? 
    # v=19, k=8, t=5 is the target.
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    restarts = 0
    
    while time.time() < deadline:
        random.seed(restarts)
        covered = [0] * n_t_subs
        uncovered_count = n_t_subs
        blocks = []
        
        while uncovered_count > 0:
            B = []
            while len(B) < k:
                candidates = [x for x in range(v) if x not in B]
                if not candidates:
                    break # Should not happen if v >= k
                
                # Precompute (t-1)-subsets of current B
                if len(B) >= t - 1:
                    combs_B = list(itertools.combinations(B, t - 1))
                else:
                    combs_B = []
                
                best_gain = -1
                best_x = None
                # If no combs, gain is 0 for all. Pick randomly to vary path.
                if not combs_B:
                    best_x = random.choice(candidates)
                else:
                    for x in candidates:
                        gain = 0
                        # Check which t-subsets containing x are uncovered
                        # and are otherwise contained in B
                        for c in combs_B:
                            # Create t-subset s = c + (x,)
                            # c is a tuple of t-1 points
                            # x is a point
                            # We need to sort to match the key in sub_id
                            # Optimization: Since c is sorted and x is not in B (so not in c),
                            # we can merge insert x into c to get sorted tuple.
                            # Or just tuple(sorted(c + (x,)))
                            # sorted is O(t log t). t=5, so fast.
                            s = tuple(sorted(c + (x,)))
                            idx = sub_id[s]
                            if covered[idx] == 0:
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                        elif gain == best_gain and best_x is not None:
                            # Tie-breaking: random choice to diversify restarts
                            if random.random() < 0.5:
                                best_x = x
                
                if best_x is None:
                    best_x = candidates[0]
                B.append(best_x)
            
            # Sort block for consistency (though set logic doesn't care, good for output)
            B.sort()
            blocks.append(B)
            
            # Update covered counts
            for s in itertools.combinations(B, t):
                idx = sub_id[s]
                if covered[idx] == 0:
                    uncovered_count -= 1
                covered[idx] += 1
        
        # Post-processing: Remove redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate backwards to allow popping
            for i in range(len(blocks) - 1, -1, -1):
                B = blocks[i]
                essential = False
                # Check if any t-subset in B is covered ONLY by this block
                for s in itertools.combinations(B, t):
                    idx = sub_id[s]
                    if covered[idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block
                    blocks.pop(i)
                    # Update covered counts
                    for s in itertools.combinations(B, t):
                        covered[sub_id[s]] -= 1
                    changed = True
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
        
        restarts += 1
        
        # If we've done enough restarts or found a very good solution, we might stop early?
        # No, keep running to find better.
        
    if best_blocks is None:
        # Fallback: return an empty list or a trivial covering if something went wrong
        # But we should always have at least one restart.
        # If v=0, return []
        return []
        
    return best_blocks
