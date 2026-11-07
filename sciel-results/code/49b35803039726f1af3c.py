import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
LCM_ALL = 30030

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    
    # Precompute masks for each residue
    # mask[r] = bitmask of divisors that divide r
    masks = [0] * LCM_ALL
    for r in range(LCM_ALL):
        m = 0
        for i, d in enumerate(DIVISORS):
            if r % d == 0:
                m |= (1 << i)
        masks[r] = m
    
    best_W = float('inf')
    best_assignment = None
    
    # For each starting position L
    for L in range(1, LCM_ALL + 1):
        if time.time() - start_time > 85.0:
            break
        
        # Try increasing window widths
        # Start from width 0 (single value) up to best_W - 1
        max_width = int(best_W) - 1 if best_W < float('inf') else 200
        
        for W in range(0, max_width + 1):
            R = L + W
            if R > MAX_VAL:
                break
            
            # Collect candidates: values in [L, R] and their masks
            # Map to residues
            candidates = []  # list of (value, mask)
            for v in range(L, R + 1):
                r = v % LCM_ALL
                m = masks[r]
                if m != 0:
                    candidates.append((v, m))
            
            if not candidates:
                continue
            
            # Check if union of masks covers all divisors
            union = 0
            for _, m in candidates:
                union |= m
            if union != (1 << N) - 1:
                continue
            
            # Try to find a matching
            # Use DFS with MRV
            M = len(candidates)
            
            # For each divisor i, list of candidate indices
            div_to_cands = [[] for _ in range(N)]
            for idx, (v, m) in enumerate(candidates):
                for i in range(N):
                    if (m >> i) & 1:
                        div_to_cands[i].append(idx)
            
            # Sort divisors by number of candidates (MRV)
            div_order = sorted(range(N), key=lambda i: len(div_to_cands[i]))
            
            # Check if any divisor has 0 candidates
            if any(len(div_to_cands[i]) == 0 for i in div_order):
                continue
            
            # DFS
            used = [False] * M
            assignment = [0] * N
            
            def dfs(pos):
                if pos == N:
                    return True
                i = div_order[pos]
                for c_idx in div_to_cands[i]:
                    if not used[c_idx]:
                        used[c_idx] = True
                        assignment[i] = candidates[c_idx][0]
                        if dfs(pos + 1):
                            return True
                        used[c_idx] = False
                return False
            
            if dfs(0):
                best_W = W
                best_assignment = assignment[:]
                break  # Found min W for this L
        
        if best_W == 0:
            break  # Can't do better than 0, but 0 is impossible for distinct values
    
    if best_assignment is not None:
        return [int(x) for x in best_assignment]
    
    # Fallback: should not reach here
    return [int(d * 100 * (i + 1)) for i, d in enumerate(DIVISORS)]
