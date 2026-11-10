# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

N = 12
LCM_12 = 27720  # LCM of 1..12

def can_assign(m, L):
    """Check if we can assign distinct b_i in (m, m+L] with i | b_i for i=1..12."""
    candidates = []
    for i in range(1, N + 1):
        cands = []
        target_rem = (-m) % i
        if target_rem == 0:
            first_k = i
        else:
            first_k = target_rem
        
        k = first_k
        while k <= L:
            cands.append(k)
            k += i
            
        if not cands:
            return False
        candidates.append(cands)
        
    order = sorted(range(N), key=lambda idx: len(candidates[idx]))
    
    def backtrack(pos, used_mask):
        if pos == N:
            return True
        i_idx = order[pos]
        for k in candidates[i_idx]:
            bit = 1 << k
            if not (used_mask & bit):
                if backtrack(pos + 1, used_mask | bit):
                    return True
        return False
    
    return backtrack(0, 0)

def compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct valid b_i."""
    for L in range(1, 145):
        if can_assign(m, L):
            return L
    return 144

def construct():
    best_m = 0
    best_L = -1
    for m in range(LCM_12):
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
    return best_m
# EVOLVE-BLOCK-END
