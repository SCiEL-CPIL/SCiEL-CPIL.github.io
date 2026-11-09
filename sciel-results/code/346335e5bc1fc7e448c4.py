import time
import math
from math import gcd

# EVOLVE-BLOCK-START

_N = 512
_LIMIT = 511
_DEADLINE = time.time() + 140.0

# Precompute weights
_W = [0.0] * (_LIMIT + 1)
for a in range(1, _LIMIT + 1):
    _W[a] = 1.0 / (_N - a)

# Precompute neighbors as bitsets
# neighbors[a] will have bit b set if gcd(a,b) > 1
_NEIGHBORS = [0] * (_LIMIT + 1)
for a in range(1, _LIMIT + 1):
    mask = 0
    for b in range(1, _LIMIT + 1):
        if a != b and gcd(a, b) > 1:
            mask |= (1 << b)
    _NEIGHBORS[a] = mask

# Sort indices by weight descending
_SORTED = sorted(range(1, _LIMIT + 1), key=lambda x: -_W[x])

# Helper to sum weights of a mask
def _sum_mask(mask):
    s = 0.0
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        idx = lsb.bit_length() - 1
        s += _W[idx]
        m ^= lsb
    return s

_best_score = 0.0
_best_set = []
_current_set = []

def _dfs(idx, current_score, available_mask, sum_avail):
    global _best_score, _best_set, _current_set
    
    if time.time() > _DEADLINE:
        return

    # Pruning
    # Upper bound: current score + sum of all available weights
    ub = current_score + sum_avail
    if ub <= _best_score + 1e-12:
        return

    if idx == _LIMIT:
        if current_score > _best_score:
            _best_score = current_score
            _best_set = list(_current_set)
        return

    a = _SORTED[idx]
    
    # Option 1: Include a (if available)
    bit_a = 1 << a
    if available_mask & bit_a:
        # Include
        _current_set.append(a)
        
        # Remove a and its neighbors from available
        # neighbors[a] includes all b where gcd(a,b)>1
        # We need to remove a itself too, though a is not in neighbors[a] usually? 
        # gcd(a,a)=a>1, so a IS in neighbors[a] if we defined it that way.
        # My precomputation: if a != b. So a is NOT in neighbors[a].
        # So we must remove bit_a separately.
        removed = _NEIGHBORS[a] & available_mask
        new_mask = available_mask & ~(removed | bit_a)
        
        # Calculate sum of removed weights
        # sum_removed = sum_weights(removed)
        # We can compute this by iterating removed
        # But we already have sum_avail.
        # new_sum = sum_avail - _W[a] - sum_weights(removed)
        
        # Optimization: sum_weights(removed) is expensive if done by loop every time.
        # But removed is a subset of available.
        # Let's just compute it.
        s_rem = 0.0
        m = removed
        while m:
            lsb = m & -m
            idx2 = lsb.bit_length() - 1
            s_rem += _W[idx2]
            m ^= lsb
            
        new_sum = sum_avail - _W[a] - s_rem
        
        _dfs(idx + 1, current_score + _W[a], new_mask, new_sum)
        
        _current_set.pop()
        
        if time.time() > _DEADLINE:
            return

    # Option 2: Exclude a
    if available_mask & bit_a:
        # Exclude a, so it's no longer available
        new_mask = available_mask & ~bit_a
        new_sum = sum_avail - _W[a]
        
        _dfs(idx + 1, current_score, new_mask, new_sum)

def construct():
    global _best_score, _best_set, _current_set
    
    # Initial greedy to set a good lower bound
    # Greedy by weight
    chosen = []
    mask = (1 << (_LIMIT + 1)) - 2  # bits 1..511 set
    for a in _SORTED:
        if mask & (1 << a):
            chosen.append(a)
            mask &= ~(_NEIGHBORS[a] | (1 << a))
            
    _best_score = sum(_W[a] for a in chosen)
    _best_set = list(chosen)
    _current_set = []
    
    # Start DFS from the beginning
    # Initial mask: all bits 1..511
    init_mask = (1 << (_LIMIT + 1)) - 2
    init_sum = sum(_W[a] for a in range(1, _LIMIT + 1))
    
    # To avoid recursion depth issues or overhead, we could use iterative, 
    # but 511 is fine.
    _dfs(0, 0.0, init_mask, init_sum)
    
    return list(_best_set)

# EVOLVE-BLOCK-END
