import time
import sys

# EVOLVE-BLOCK-START

def _has_matching(m, L):
    """
    Check if there exist distinct b_1...b_12 in (m, m+L] such that i | b_i.
    Returns True if such a matching exists.
    """
    n = 12
    # candidates[i] is a list of valid values for b_i
    # We only care about distinctness, so we can just store the values.
    # To speed up backtracking, we can sort indices by number of candidates (ascending).
    
    candidates = [[] for _ in range(n)]
    # Range is (m, m+L], so x in {m+1, ..., m+L}
    for x in range(m + 1, m + L + 1):
        for i in range(1, n + 1):
            if x % i == 0:
                candidates[i-1].append(x)
                
    # If any i has no candidates, fail
    for i in range(n):
        if not candidates[i]:
            return False
            
    # Order indices by number of candidates (most constrained first)
    indices = list(range(n))
    indices.sort(key=lambda i: len(candidates[i]))
    
    used = set()
    
    def dfs(idx):
        if idx == n:
            return True
        i = indices[idx]
        for x in candidates[i]:
            if x not in used:
                used.add(x)
                if dfs(idx + 1):
                    return True
                used.remove(x)
        return False
        
    return dfs(0)

def _compute_f(m):
    """
    Compute the least positive L such that (m, m+L] contains distinct b_1...b_12 with i | b_i.
    """
    # Max L is bounded by 144 according to the prompt, but let's use 200 to be safe.
    # The prompt says "raw value ... is at most 144".
    for L in range(1, 200):
        if _has_matching(m, L):
            return L
    return 200 # Fallback, shouldn't happen

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 # Safety margin
    
    # LCM(1..12) = 27720
    M = 27720
    
    best_m = 0
    best_L = -1
    
    # We only need to check one period because of periodicity.
    # f(12, m) depends only on m mod M.
    # We want to maximize f(12, m).
    # Since the function is periodic, the max over [0, 10^6] is the max over [0, M-1].
    # Any m in [0, 10^6] with m % M == best_residue is a valid answer.
    # We can just return best_residue as it is < 10^6.
    
    for m in range(M):
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        L = _compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
            # Optional: early exit if we hit the theoretical max?
            # The prompt says max is 144. If we find 144, we can't do better.
            if best_L == 144:
                break

    return int(best_m)

# EVOLVE-BLOCK-END
