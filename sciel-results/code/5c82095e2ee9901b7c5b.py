"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _can_add(S_set, sums_mask, x):
    """
    Check if x can be added to S_set given the current sums_mask.
    sums_mask is an integer where bit i is 1 if sum i is already used.
    Returns (True, new_mask) if valid, (False, 0) otherwise.
    """
    new_sums = []
    for a in S_set:
        new_sums.append(x + a)
    new_sums.append(2 * x)
    
    # Check for conflicts within new_sums and with existing mask
    # We can just check each new sum
    for s in new_sums:
        if (sums_mask >> s) & 1:
            return False, 0
    
    # Check distinctness among new_sums
    # If x+a1 == x+a2, then a1==a2, which is impossible as S_set has distinct elements.
    # If x+a == 2*x, then a==x, which is impossible as x not in S_set.
    # So new_sums are distinct from each other automatically.
    
    # Update mask
    new_mask = sums_mask
    for s in new_sums:
        new_mask |= (1 << s)
        
    return True, new_mask

def _maximize_randomly(seed=None, order="random"):
    """
    Construct a maximal Sidon set by greedily adding elements.
    order: 'random', 'asc', 'desc'
    """
    if seed is not None:
        random.seed(seed)
        
    S_set = set()
    sums_mask = 0
    
    candidates = list(range(1, N + 1))
    if order == "random":
        random.shuffle(candidates)
    elif order == "desc":
        candidates.sort(reverse=True)
    # 'asc' is default sorted
    
    for x in candidates:
        can, new_mask = _can_add(S_set, sums_mask, x)
        if can:
            S_set.add(x)
            sums_mask = new_mask
            
    return list(S_set)

def _is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x preserves Sidon
        ok = True
        new_sums = [x + a for a in S] + [2 * x]
        if len(set(new_sums)) != len(new_sums):
            ok = False
        else:
            for s in new_sums:
                if s in sums:
                    ok = False
                    break
        if ok:
            return False
    return True

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Phase 1: Randomized Greedy Maximization
    # Try many random orders
    for seed in range(50000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 2:
            break
            
        # Vary the strategy slightly
        if seed % 3 == 0:
            S = _maximize_randomly(seed=seed, order="random")
        elif seed % 3 == 1:
            S = _maximize_randomly(seed=seed, order="asc")
        else:
            S = _maximize_randomly(seed=seed, order="desc")
            
        if len(S) < best_size:
            # Verify it is actually maximal (should be, but safe check)
            if _is_sidon(S) and _is_maximal(S, N):
                best_size = len(S)
                best_set = S
            # If we find a very small set, we might break early?
            # Theoretical lower bound for Sidon set in [1,N] is ~sqrt(N).
            # Maximal sets are usually larger.
            if best_size <= 20: # Heuristic lower bound
                 break

    # If nothing found (should not happen), use deterministic
    if best_set is None:
        S = _maximize_randomly(seed=0, order="asc")
        best_set = S

    return best_set
# EVOLVE-BLOCK-END
