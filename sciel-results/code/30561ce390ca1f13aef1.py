import sys
from itertools import combinations
from math import lcm

# Global variables for speed
BAD_TRIPLES = []
PAIR_TO_THIRDS = {}
_ORDER = []
_BEST_SIZE = 0
_BEST_SET = []

def _precompute():
    global BAD_TRIPLES, PAIR_TO_THIRDS
    BAD_TRIPLES = []
    PAIR_TO_THIRDS = {}
    n = 128
    # Generate all bad triples
    # a < b < c
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    BAD_TRIPLES.append((a, b, c))
                    # Map pairs to third elements
                    # Use sorted tuple as key
                    key_ab = (a, b)
                    key_ac = (a, c)
                    key_bc = (b, c)
                    PAIR_TO_THIRDS[key_ab] = PAIR_TO_THIRDS.get(key_ab, []) + [c]
                    PAIR_TO_THIRDS[key_ac] = PAIR_TO_THIRDS.get(key_ac, []) + [b]
                    PAIR_TO_THIRDS[key_bc] = PAIR_TO_THIRDS.get(key_bc, []) + [a]

def _get_packing(avail_mask):
    count = 0
    used = 0
    # Iterate through precomputed triple masks?
    # Let's precompute masks for triples to avoid bit shifting in loop.
    # We'll do this in _precompute.
    for mask in _TRIPLE_MASKS:
        if (mask & avail_mask) == mask:
            if (mask & used) == 0:
                count += 1
                used |= mask
    return count

_TRIPLE_MASKS = []

def _solve_bnb(initial_mask, best_set, best_size):
    global _BEST_SIZE, _BEST_SET
    
    def recurse(included_mask, avail_mask):
        # 1. Upper Bound
        if included_mask.bit_count() + avail_mask.bit_count() <= _BEST_SIZE:
            return
            
        # Greedy Packing Bound
        packing = _get_packing(avail_mask)
        ub = included_mask.bit_count() + avail_mask.bit_count() - packing
        
        if ub <= _BEST_SIZE:
            return
            
        # 2. Base Case
        if avail_mask == 0:
            if included_mask.bit_count() > _BEST_SIZE:
                _BEST_SIZE = included_mask.bit_count()
                s = []
                m = included_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s.append(idx + 1)
                    m ^= lsb
                _BEST_SET = s
            return
            
        # 3. Select Vertex
        v_idx = -1
        for v in _ORDER:
            if (avail_mask & (1 << (v-1))) != 0:
                v_idx = v
                break
                
        if v_idx == -1:
            return
            
        v_bit = 1 << (v_idx - 1)
        
        # Branch 1: Exclude v
        recurse(included_mask, avail_mask & ~v_bit)
        
        # Branch 2: Include v
        valid = True
        inc_m = included_mask
        while inc_m:
            lsb = inc_m & -inc_m
            a_idx = lsb.bit_length()
            a = a_idx
            key = (a, v_idx) if a < v_idx else (v_idx, a)
            thirds = PAIR_TO_THIRDS.get(key, [])
            for w in thirds:
                if (included_mask & (1 << (w-1))) != 0:
                    valid = False
                    break
            if not valid:
                break
            inc_m ^= lsb
            
        if not valid:
            return
            
        to_exclude = 0
        inc_m = included_mask
        while inc_m:
            lsb = inc_m & -inc_m
            w_idx = lsb.bit_length()
            w = w_idx
            key = (v_idx, w) if v_idx < w else (w, v_idx)
            thirds = PAIR_TO_THIRDS.get(key, [])
            for u in thirds:
                if u != v_idx and u != w:
                    to_exclude |= (1 << (u-1))
            inc_m ^= lsb
            
        new_avail = avail_mask & ~v_bit & ~to_exclude
        new_included = included_mask | v_bit
        
        recurse(new_included, new_avail)

    _BEST_SIZE = best_size
    _BEST_SET = list(best_set)
    
    recurse(0, initial_mask)
    
    return _BEST_SET

def construct():
    global _ORDER, _TRIPLE_MASKS
    if not BAD_TRIPLES:
        _precompute()
        _TRIPLE_MASKS = [ (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) for a,b,c in BAD_TRIPLES ]
        
    # Calculate global degree for ordering
    degrees = [0] * 129
    for a, b, c in BAD_TRIPLES:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1
        
    # Sort vertices 1..128 by degree descending
    _ORDER = sorted(range(1, 129), key=lambda x: -degrees[x])
    
    # Initial Greedy Solution
    selected = []
    for x in range(1, 129):
        if all(not (lcm(a,b)==lcm(a,x)==lcm(b,x)) for a,b in combinations(selected,2)):
            selected.append(x)
            
    # Full mask
    initial_mask = (1 << 128) - 1
    
    result = _solve_bnb(initial_mask, selected, len(selected))
    
    return result
