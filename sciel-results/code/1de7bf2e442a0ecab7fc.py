# EVOLVE-BLOCK-START

import time
from fractions import Fraction

def _base_p_set(n=256, p=11):
    """
    Generate a 3-AP-free set using base-p digits.
    Numbers with no zero digits in base p are 3-AP-free.
    """
    S = []
    for x in range(1, n + 1):
        num = x
        has_zero = False
        while num > 0:
            if num % p == 0:
                has_zero = True
                break
            num //= p
        if not has_zero:
            S.append(x)
    return S

def _iter_set_bits(bitset):
    """Iterate through indices of set bits in an integer bitmask."""
    b = bitset
    while b:
        lb = b & -b
        idx = lb.bit_length() - 1
        yield idx + 1
        b ^= lb

def _can_add_to_bitset(x, bitset):
    """
    Check if x can be added to the set represented by bitset.
    bitset is an integer where bit i is 1 if i+1 is in the set.
    """
    for a in _iter_set_bits(bitset):
        # Case 1: a, x, c => c = 2x - a
        c = 2 * x - a
        if 1 <= c <= 256 and c != x:
            if (bitset >> (c-1)) & 1:
                return False
        # Case 2: a, b, x => b = (a+x)/2
        s = a + x
        if s % 2 == 0:
            b = s // 2
            if b != x:
                if (bitset >> (b-1)) & 1:
                    return False
    return True

def _local_search(start_set, deadline):
    """
    Local search using bitmask.
    """
    n = 256
    S = set(start_set)
    bitset = 0
    for x in S:
        bitset |= (1 << (x-1))
    
    # Precompute reciprocals
    inv = {x: Fraction(1, x) for x in range(1, n+1)}
    current_val = sum(inv[x] for x in S)
    best_val = current_val
    best_set = set(S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        
        # Try swaps (u in S, v not in S)
        # We iterate v in ascending order (best reciprocal first)
        # and u in descending order (worst reciprocal first) to find improvements quickly
        # But we must ensure u > v for improvement (1/v > 1/u)
        
        for v in range(1, n + 1):
            if v in S:
                continue
            
            # We need u such that u > v
            # Iterate through S_list
            # To speed up, we could sort S_list descending, but list is small.
            for u in S_list:
                if u <= v:
                    continue
                
                # Check if v can be added to S \ {u}
                bitset_no_u = bitset & ~(1 << (u-1))
                
                if _can_add_to_bitset(v, bitset_no_u):
                    # Valid swap
                    S.remove(u)
                    S.add(v)
                    bitset = bitset_no_u | (1 << (v-1))
                    
                    current_val = current_val - inv[u] + inv[v]
                    
                    if current_val > best_val:
                        best_val = current_val
                        best_set = set(S)
                    
                    improved = True
                    break # Restart outer loop
            
            if improved:
                break
        
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    
    # 1. Base 11 construction
    initial_set = _base_p_set(256, 11)
    
    # 2. Local search
    best_set = _local_search(initial_set, deadline)
    
    return best_set

# EVOLVE-BLOCK-END
