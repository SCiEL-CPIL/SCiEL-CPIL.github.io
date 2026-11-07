"""
Strategy:
1. LCM of divisors = 30030, so the problem is periodic.
2. For a given window width W, split divisors into rigid (d > W) and flexible.
3. Rigid divisors have at most one multiple in [L, L+W], so their assignment is forced.
4. Precompute which L mod d values give valid offsets for each rigid divisor.
5. Use CRT to quickly filter L values that work for ALL rigid divisors simultaneously.
6. For remaining L values, use fast backtracking for flexible divisors.
"""

import time
from math import gcd
from functools import reduce

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    divisors = P['divisors']
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    best_solution = None
    
    # Try W from 16 upwards (16 is known optimal)
    for W in range(16, 50):
        if time.time() > deadline:
            break
        
        # Identify rigid and flexible divisors
        rigid_indices = []
        flexible_indices = []
        for i, d in enumerate(divisors):
            if d > W:
                rigid_indices.append(i)
            else:
                flexible_indices.append(i)
        
        rigid_divs = [divisors[i] for i in rigid_indices]
        
        # For each rigid divisor, precompute valid (residue -> offset) mappings
        # offset = (d - (L % d)) % d, must be <= W
        rigid_valid = []
        for d in rigid_divs:
            valid_map = {}
            for r in range(d):
                offset = (d - r) % d
                if offset <= W:
                    valid_map[r] = offset
            rigid_valid.append(valid_map)
        
        # Check if any L in [0, LCM) works
        # Optimization: iterate L but with fast rejection
        found = False
        for L in range(LCM):
            if time.time() > deadline:
                break
            
            # Quick check for rigid divisors
            used_offsets = set()
            rigid_ok = True
            rigid_assignments = {}
            
            for ri, idx in enumerate(rigid_indices):
                d = rigid_divs[ri]
                r = L % d
                if r not in rigid_valid[ri]:
                    rigid_ok = False
                    break
                offset = rigid_valid[ri][r]
                if offset in used_offsets:
                    rigid_ok = False
                    break
                used_offsets.add(offset)
                rigid_assignments[idx] = L + offset
            
            if not rigid_ok:
                continue
            
            # Now try to assign flexible divisors
            assignments = [None] * n
            for idx, val in rigid_assignments.items():
                assignments[idx] = val
            
            sol = _assign_flexible(divisors, flexible_indices, L, W, used_offsets, assignments)
            if sol is not None:
                best_solution = sol
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        return [int(d) for d in P['divisors']]
    
    # Validate
    result = []
    for i, val in enumerate(best_solution):
        v = int(val)
        if not (1 <= v <= 1000000) or v % divisors[i] != 0:
            return [int(d) for d in P['divisors']]
        result.append(v)
    
    if len(set(result)) != len(result):
        return [int(d) for d in P['divisors']]
    
    return result

def _assign_flexible(divisors, flexible_indices, L, W, used_offsets, assignments):
    """Fast backtracking with MRV for flexible divisors."""
    if not flexible_indices:
        return assignments
    
    # Generate candidate offsets for each flexible divisor
    candidates = []
    for i in flexible_indices:
        d = divisors[i]
        cands = []
        k = (L + d - 1) // d
        val = k * d
        while val <= L + W:
            offset = val - L
            if offset not in used_offsets:
                cands.append(offset)
            val += d
        candidates.append(cands)
    
    for cands in candidates:
        if not cands:
            return None
    
    # MRV: order by fewest candidates
    order = sorted(range(len(flexible_indices)), key=lambda x: len(candidates[x]))
    
    def backtrack(pos):
        if pos == len(order):
            return True
        fi = order[pos]
        orig_idx = flexible_indices[fi]
        for offset in candidates[fi]:
            if offset not in used_offsets:
                used_offsets.add(offset)
                assignments[orig_idx] = L + offset
                if backtrack(pos + 1):
                    return True
                used_offsets.remove(offset)
                assignments[orig_idx] = None
        return False
    
    return assignments if backtrack(0) else None
