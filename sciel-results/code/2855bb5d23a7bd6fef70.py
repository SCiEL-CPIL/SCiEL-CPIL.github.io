import itertools
import time
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0

    # Total number of t-subsets
    total_t = list(itertools.combinations(range(v), t))
    n_t = len(total_t)
    
    # Map each t-subset to an index
    t_to_idx = {sub: i for i, sub in enumerate(total_t)}
    
    # covered: boolean array of size n_t
    covered = np.zeros(n_t, dtype=bool)
    
    # For each point x, and each (t-1)-subset S of the other points,
    # we want to know which t-subsets contain S ∪ {x}.
    # Instead, for each point x, precompute a list of (t-1)-subsets of range(v) not containing x,
    # and map each to the index of the t-subset formed by adding x.
    
    # Precompute for each point x: list of (t-1)-subsets of points != x, and the corresponding t-subset indices
    # This is expensive, so we do it lazily or use a smarter representation.
    
    # Alternative: For a given partial block B (list of points), the gain of adding x is:
    #   number of (t-1)-subsets S of B such that S ∪ {x} is an uncovered t-subset.
    # We can compute this on the fly.
    
    blocks = []
    
    def remaining_time():
        return deadline - time.time()
    
    while remaining_time() > 5.0:
        if not np.any(~covered):
            break
        
        # Find the first uncovered t-subset to start a new block
        first_uncovered = np.where(~covered)[0][0]
        t_sub = total_t[first_uncovered]
        
        block = list(t_sub)
        
        while len(block) < k and remaining_time() > 5.0:
            block_set = set(block)
            best_gain = -1
            best_x = None
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are covered by adding x
                # These are t-subsets of the form S ∪ {x} where S is a (t-1)-subset of block
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    t_candidate = tuple(sorted(S + (x,)))
                    idx = t_to_idx[t_candidate]
                    if not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            # Pad with arbitrary points
            block_set = set(block)
            for x in range(v):
                if x not in block_set and len(block) < k:
                    block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Update covered
        for t_sub in itertools.combinations(block, t):
            covered[t_to_idx[t_sub]] = True
    
    return blocks
