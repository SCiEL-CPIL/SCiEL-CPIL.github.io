# EVOLVE-BLOCK-START
import time
import random

N = 256
_MAX_SUM = 2 * N
_DEADLINE = 55.0

def _greedy_sidon_bitmask(randomize=False, seed=None, order="asc"):
    """Build a Sidon set greedily using bitmasks for speed."""
    if seed is not None:
        random.seed(seed)
    
    selected = []
    sums_mask = 0  # Bit i is 1 if sum i is present
    
    candidates = list(range(1, N + 1))
    if randomize:
        random.shuffle(candidates)
    elif order == "desc":
        candidates.sort(reverse=True)

    for x in candidates:
        # Calculate mask for new sums involving x
        # New sums are x+a for a in selected, and 2*x
        # We can't easily shift the whole sums_mask because sums_mask contains a+b, not just a.
        # We must iterate selected to build the new mask part.
        
        new_mask = 1 << (2 * x)
        for a in selected:
            new_mask |= 1 << (x + a)
            
        # Check for conflicts with existing sums
        if sums_mask & new_mask:
            continue
            
        # Check for internal conflicts in new_mask?
        # x+a1 == x+a2 => a1==a2 (distinct in selected)
        # x+a == 2x => a==x (x not in selected)
        # So new_mask is always sparse enough (no internal collisions) for distinct a.
        
        # Accept x
        selected.append(x)
        sums_mask |= new_mask

    selected.sort()
    return selected

def _is_maximal_bitmask(S, N, sums_mask):
    """Check if S is inclusion-maximal Sidon in [1,N] using precomputed sums_mask."""
    S_set = set(S)
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
            
        new_mask = 1 << (2 * x)
        for a in S:
            new_mask |= 1 << (x + a)
            
        if not (sums_mask & new_mask):
            return False
            
    return True

def _sidon_and_maximal_fast(S, N):
    if not S:
        return False
        
    # Verify Sidon property and build mask
    sums_mask = 0
    n = len(S)
    for i in range(n):
        a = S[i]
        # Check against existing sums in mask
        # We can build the mask for pairs involving S[i] and check against current sums_mask
        # But simpler: just rebuild or check incrementally.
        # Let's just rebuild to be safe and simple, it's O(n^2) but n is small (~10-20).
        # Actually, we can use the greedy logic to verify.
        pass
        
    # Re-verify Sidon property strictly
    sums_set = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums_set:
                return False
            sums_set.add(s)
            
    # Build mask for maximality check
    for s in sums_set:
        sums_mask |= 1 << s
        
    return _is_maximal_bitmask(S, N, sums_mask)

def _try_minimize_by_removal(S, N):
    """Try to remove elements while keeping Sidon + maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal_fast(candidate, N):
                S = candidate
                improved = True
                break
    return S

def _local_search_swap(S, N):
    """Try to improve S by swapping one element for another (or removing/adding)."""
    S_set = set(S)
    best_S = list(S)
    best_size = len(S)
    
    # Try removing an element
    for i in range(len(S)):
        candidate = S[:i] + S[i+1:]
        if _sidon_and_maximal_fast(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_S = candidate
    
    if len(best_S) < best_size:
        S = best_S
    S_set = set(S)
    
    # Try swapping
    for i in range(len(S)):
        for x in range(1, N + 1):
            if x in S_set:
                continue
            candidate = S[:i] + S[i+1:] + [x]
            candidate.sort()
            if _sidon_and_maximal_fast(candidate, N):
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_S = candidate
                    S = best_S
                    S_set = set(S)
    
    return best_S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Phase 1: Random greedy constructions with bitmask speedup
    for seed in range(2000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
            
        if seed % 3 == 0:
            S = _greedy_sidon_bitmask(randomize=True, seed=seed, order="asc")
        elif seed % 3 == 1:
            S = _greedy_sidon_bitmask(randomize=False, seed=seed, order="desc")
        else:
            S = _greedy_sidon_bitmask(randomize=True, seed=seed, order="desc")
            
        if len(S) < best_size:
            S_min = _try_minimize_by_removal(S, N)
            if _sidon_and_maximal_fast(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                
            if elapsed < _DEADLINE - 15:
                S_swap = _local_search_swap(S_min, N)
                if _sidon_and_maximal_fast(S_swap, N) and len(S_swap) < best_size:
                    best_size = len(S_swap)
                    best_set = S_swap
            
            if elapsed > _DEADLINE - 10:
                break
    
    if best_set is None:
        S = _greedy_sidon_bitmask(randomize=False, seed=0, order="asc")
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal_fast(S_min, N):
            best_set = S_min
        else:
            best_set = S
            
    return best_set
# EVOLVE-BLOCK-END
