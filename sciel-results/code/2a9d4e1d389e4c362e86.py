import itertools
import time
import random

_cache = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_set = set(map(frozenset, all_t_subsets))
    
    # Map block tuple to its coverage set
    block_covs = {}
    for b in itertools.combinations(range(v), k):
        b_tuple = tuple(b)
        b_set = frozenset(b)
        cov = set(map(frozenset, itertools.combinations(b, t)))
        block_covs[b_tuple] = cov
        
    _cache[key] = (t_set, block_covs)
    return _cache[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    t_set, block_covs = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Symmetry breaking: Fix the first block to be {0, ..., k-1}
    first_block = tuple(range(k))
    first_cov = block_covs[first_block]
    
    num_restarts = 50 # Each run is fast
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = set(t_set)
        uncovered.difference_update(first_cov)
        current_blocks = [list(first_block)]
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_block = None
            best_cov = None
            candidates = []
            
            for b, cov in block_covs.items():
                # Skip the first block? It's already chosen.
                # Actually, we can include it, gain will be 0 if fully covered, or >0 if not?
                # We removed its coverage, so gain for first_block should be 0.
                # So it won't be picked again.
                
                gain = len(uncovered & cov)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    best_cov = cov
                    candidates = [(b, cov)]
                elif gain == best_gain and gain > 0:
                    candidates.append((b, cov))
                    
            if best_gain == 0:
                break
                
            if len(candidates) > 1:
                sel_block, sel_cov = random.choice(candidates)
            else:
                sel_block, sel_cov = best_block, best_cov
                
            current_blocks.append(list(sel_block))
            uncovered.difference_update(sel_cov)
            
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    if best_blocks is None:
        # Fallback
        best_blocks = [list(first_block)]
        
    return best_blocks
