"""Pack distinct multiples: minimize enclosing interval length."""
import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000
LCM = 30030

def construct():
    deadline = time.time() + 85
    
    def check_feasible(L):
        """
        Checks if there exists a window of length L starting at some residue r
        that can accommodate distinct multiples for all divisors.
        Returns (is_feasible, assignment)
        """
        # Iterate residues
        for r in range(LCM):
            if time.time() > deadline:
                return False, None
            
            # Build candidate bitmasks for each divisor
            cands = []
            feasible = True
            for d in DIVISORS:
                lo_k = (r + d - 1) // d
                hi_k = (r + L) // d
                if lo_k > hi_k:
                    feasible = False
                    break
                # Generate bits
                bits = []
                for k in range(lo_k, hi_k + 1):
                    v = k * d
                    # v is in [r, r+L]
                    bit = v - r
                    # Sanity check: 0 <= bit <= L
                    if 0 <= bit <= L:
                        bits.append(bit)
                cands.append(bits)
            
            if not feasible:
                continue
            
            # MRV ordering
            order = sorted(range(N), key=lambda i: len(cands[i]))
            if len(cands[order[0]]) == 0:
                continue
            
            used_mask = 0
            assigned_bits = [None] * N
            
            def backtrack(idx):
                nonlocal used_mask
                if idx == N:
                    return True
                i = order[idx]
                for bit in cands[i]:
                    mask = 1 << bit
                    if not (used_mask & mask):
                        used_mask |= mask
                        assigned_bits[i] = bit
                        if backtrack(idx + 1):
                            return True
                        used_mask ^= mask
                        assigned_bits[i] = None
                return False
            
            if backtrack(0):
                # Reconstruct
                assignment = [r + b for b in assigned_bits]
                # Check positive and max val
                if all(1 <= v <= MAX_VAL for v in assignment):
                    return True, assignment
        
        return False, None

    # Linear search for small L
    best_assignment = None
    for L in range(10, 201):
        if time.time() > deadline:
            break
        is_f, assign = check_feasible(L)
        if is_f:
            best_assignment = assign
            break
    
    if best_assignment is None:
        # Binary search for larger L
        lo = 201
        hi = LCM
        while lo < hi:
            if time.time() > deadline:
                break
            mid = (lo + hi) // 2
            is_f, assign = check_feasible(mid)
            if is_f:
                hi = mid
                best_assignment = assign
            else:
                lo = mid + 1
        
        # Check final candidate
        if best_assignment is None:
            is_f, assign = check_feasible(lo)
            if is_f:
                best_assignment = assign

    if best_assignment:
        return [int(v) for v in best_assignment]
    
    # Fallback
    return DIVISORS[:]
