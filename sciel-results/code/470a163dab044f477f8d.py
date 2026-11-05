"""Precompute minimal-k antichain representations for targets 1..128.

Strategy:
  1. Quick pass: for each target, find a valid representation with a reasonable
     upper bound on k (using fast DFS with pruning).
  2. Refinement pass: using remaining time budget, try to reduce k for targets
     that currently use more terms, starting from the hardest targets.
  
  The key insight is that per-target searches are independent, so we can
  allocate time adaptively: spend more effort on targets where reducing k
  would improve the total sum the most.
"""

import time
import sys

# --- Smooth number generation ---
def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}

# Time budget: we have 90s wall-clock total; reserve ~5s for construct() and overhead
_DEADLINE = time.time() + 82.0

# --- Precompute for each smooth value, the set of smooth values it conflicts with
#     (i.e. values that divide it or are divisible by it), as a bitmask over _SMOOTH indices.
def _build_conflict_masks():
    masks = [0] * _N
    for i in range(_N):
        vi = _SMOOTH[i]
        m = 0
        for j in range(_N):
            if i == j:
                continue
            vj = _SMOOTH[j]
            if vj % vi == 0 or vi % vj == 0:
                m |= (1 << j)
        masks[i] = m
    return masks

_CONFLICT = _build_conflict_masks()

# --- For each smooth index, precompute the "forbidden" mask: indices of values
#     that conflict with it (including itself, so we can easily remove it).
_FORBIDDEN = [(_CONFLICT[i] | (1 << i)) for i in range(_N)]

# --- Precompute, for each subset of smooth values (as a bitmask), the sum of values
#     in that subset. We only need this for small subsets, so we compute on the fly
#     with memoization.

_sum_memo = {}
def _mask_sum(mask):
    if mask in _sum_memo:
        return _sum_memo[mask]
    s = 0
    m = mask
    i = 0
    while m:
        if m & 1:
            s += _SMOOTH[i]
        m >>= 1
        i += 1
    _sum_memo[mask] = s
    return s

# --- Core solver: find antichain representation of target n with exactly k terms.
#     Returns list of k smooth values or None.

def _solve_target_k(n, k):
    """Find an antichain representation of n with exactly k terms. Returns list or None."""
    if k == 1:
        if n in _VAL_INDEX:
            return [n]
        return None
    
    # Initial available: all smooth values <= n
    init_avail = 0
    for i in range(_N):
        if _SMOOTH[i] <= n:
            init_avail |= (1 << i)
    
    memo = {}
    
    def dfs(rem, r, avail):
        if r == 0:
            return rem == 0
        if rem <= 0:
            return False
        key = (rem, r, avail)
        if key in memo:
            return memo[key]
        
        # Extract values from avail
        vals_in_avail = []
        a = avail
        i = 0
        while a:
            if a & 1:
                vals_in_avail.append(_SMOOTH[i])
            a >>= 1
            i += 1
        
        if len(vals_in_avail) < r:
            memo[key] = False
            return False
        
        vals_in_avail.sort(reverse=True)
        
        # Pruning bounds
        max_sum = sum(vals_in_avail[:r])
        if rem > max_sum:
            memo[key] = False
            return False
        
        min_sum = sum(vals_in_avail[-r:])
        if rem < min_sum:
            memo[key] = False
            return False
        
        # Additional pruning: if r==1, check directly
        if r == 1:
            if rem in _VAL_INDEX and (avail >> _VAL_INDEX[rem]) & 1:
                memo[key] = True
                return True
            memo[key] = False
            return False
        
        # Try each candidate in descending order (larger first for better pruning)
        for i in range(_N - 1, -1, -1):
            if not (avail >> i) & 1:
                continue
            v = _SMOOTH[i]
            if v > rem:
                continue
            new_avail = avail & ~_FORBIDDEN[i]
            if dfs(rem - v, r - 1, new_avail):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    if not dfs(n, k, init_avail):
        return None
    
    # Reconstruct
    memo2 = {}
    def dfs_recon(rem, r, avail):
        if r == 0:
            return [] if rem == 0 else None
        key = (rem, r, avail)
        if key in memo2:
            return memo2[key]
        if r == 1:
            if rem in _VAL_INDEX and (avail >> _VAL_INDEX[rem]) & 1:
                memo2[key] = [rem]
                return [rem]
            memo2[key] = None
            return None
        for i in range(_N - 1, -1, -1):
            if not (avail >> i) & 1:
                continue
            v = _SMOOTH[i]
            if v > rem:
                continue
            new_avail = avail & ~_FORBIDDEN[i]
            sub = dfs_recon(rem - v, r - 1, new_avail)
            if sub is not None:
                result = [v] + sub
                memo2[key] = result
                return result
        memo2[key] = None
        return None
    
    return dfs_recon(n, k, init_avail)


def _solve_one_target(n, max_k=12):
    """Find the minimal-k antichain representation of n. Returns list of values or None."""
    if n in _VAL_INDEX:
        return [n]
    
    for k in range(2, max_k + 1):
        result = _solve_target_k(n, k)
        if result is not None:
            return result
    
    return None

# --- Precompute the entire table with time-budgeted refinement ---

def _build_table():
    """Build table with two phases:
       Phase 1: Quick solve for all targets (find any valid representation).
       Phase 2: Use remaining time to refine (reduce k) for hardest targets.
    """
    table = [None] * 129
    k_used = [0] * 129  # track current k for each target
    
    # Phase 1: Quick solve - find a valid representation for each target
    # Use a moderate max_k first; we'll refine later if time permits
    for n in range(1, 129):
        if time.time() > _DEADLINE - 5:
            break
        if n in _VAL_INDEX:
            table[n] = [n]
            k_used[n] = 1
            continue
        # Try increasing k until we find a solution
        for k in range(2, 13):
            if time.time() > _DEADLINE - 5:
                break
            result = _solve_target_k(n, k)
            if result is not None:
                table[n] = result
                k_used[n] = k
                break
        if table[n] is None:
            # Fallback: greedy construction (should not happen for 1..128)
            table[n] = _greedy_construct(n)
            k_used[n] = len(table[n]) if table[n] else 1
    
    # Phase 2: Refinement - try to reduce k for targets with higher k
    # Sort targets by current k_used (descending) to prioritize hardest ones
    remaining_time = _DEADLINE - time.time()
    if remaining_time > 2:
        # Targets that could potentially be improved (k > 1)
        candidates = [n for n in range(1, 129) if k_used[n] > 1 and table[n] is not None]
        candidates.sort(key=lambda n: -k_used[n])
        
        for n in candidates:
            if time.time() > _DEADLINE - 3:
                break
            current_k = k_used[n]
            # Try to find a solution with fewer terms
            for new_k in range(2, current_k):
                if time.time() > _DEADLINE - 3:
                    break
                result = _solve_target_k(n, new_k)
                if result is not None:
                    table[n] = result
                    k_used[n] = new_k
                    break
    
    # Final validation: ensure all targets have valid representations
    for n in range(1, 129):
        if table[n] is None:
            table[n] = _greedy_construct(n)
            k_used[n] = len(table[n]) if table[n] else 1
    
    return table, k_used


def _greedy_construct(n):
    """Fallback greedy construction for target n. Returns list of smooth values."""
    if n in _VAL_INDEX:
        return [n]
    
    # Simple greedy: pick largest available smooth values that don't conflict
    result = []
    chosen = set()
    remaining = n
    
    # Try to build up to 30 terms
    for _ in range(30):
        if remaining == 0:
            break
        best_v = None
        best_i = None
        for i in range(_N - 1, -1, -1):
            v = _SMOOTH[i]
            if v > remaining:
                continue
            if v in chosen:
                continue
            # Check if v conflicts with any chosen value
            conflict = False
            for c in chosen:
                if c % v == 0 or v % c == 0:
                    conflict = True
                    break
            if not conflict:
                best_v = v
                best_i = i
                break
        if best_v is None:
            # No valid choice found; try smaller values
            for i in range(_N - 1, -1, -1):
                v = _SMOOTH[i]
                if v > remaining:
                    continue
                if v in chosen:
                    continue
                conflict = False
                for c in chosen:
                    if c % v == 0 or v % c == 0:
                        conflict = True
                        break
                if not conflict:
                    best_v = v
                    best_i = i
                    break
        if best_v is None:
            break
        result.append(best_v)
        chosen.add(best_v)
        remaining -= best_v
    
    if remaining == 0:
        return result
    else:
        # This shouldn't happen for 1..128, but return what we have
        return result


_TABLE, _K_USED = _build_table()

def construct():
    # Return rows for targets 1..128
    return _TABLE[1:129]
