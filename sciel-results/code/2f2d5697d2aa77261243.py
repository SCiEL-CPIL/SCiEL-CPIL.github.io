"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
N = 128

def _build_progressions():
    """Precompute all nonconstant 4-term progressions within [0, N-1]."""
    progs = []
    for a in range(N):
        for d in range(1, N):
            e = a + 3 * d
            if e >= N:
                break
            progs.append((a, a + d, a + 2 * d, e))
    return progs

def _check_valid_fast(colors, progs):
    """Check validity using precomputed progressions."""
    for (a, b, c, e) in progs:
        ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
        # Count distinct colors among the four
        if ca == cb == cc == ce:
            return False
        # Check if at most 2 distinct colors
        distinct = 0
        if ca != cb:
            distinct += 1
        if ca != cc:
            distinct += 1
        if ca != ce:
            distinct += 1
        if cb != cc and cb != ca:
            distinct += 1
        if cb != ce and cb != ca:
            distinct += 1
        if cc != ce and cc != ca and cc != cb:
            distinct += 1
        # If distinct < 2, we have at most 2 colors (need at least 3)
        # Actually let's just count properly
        if len({ca, cb, cc, ce}) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_with_k_colors(k, progs):
    """Greedy: assign smallest valid color in [0,k-1] to each position."""
    colors = [-1] * N
    # Build reverse index: for each position x, which progressions contain x as the last element?
    # Actually, we check progressions ending at x when assigning x.
    # A progression (a, a+d, a+2d, a+3d) ends at position x=a+3d.
    # When we assign position x, we need to check all progressions where x is the 4th element.
    # Also progressions where x is the 1st, 2nd, or 3rd element but the 4th is already assigned?
    # No - we assign left to right, so when assigning x, only progressions with all 4 positions <= x matter.
    # Those are progressions where x is the 4th element (a+3d = x, so a = x-3d >= 0).
    
    for x in range(N):
        found = False
        for color in range(k):
            colors[x] = color
            valid = True
            # Check progressions where x is the last element: a+3d = x => a = x - 3d
            for d in range(1, x // 3 + 1):
                a = x - 3 * d
                if a < 0:
                    break
                b = a + d
                c = a + 2 * d
                if len({colors[a], colors[b], colors[c], colors[x]}) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def _try_merge(colors, c1, c2, progs):
    """Try merging color c2 into c1. Returns True if valid."""
    # Only positions with color c2 change
    changed = [i for i in range(N) if colors[i] == c2]
    if not changed:
        return False
    # Check all progressions that include at least one changed position
    for (a, b, c, e) in progs:
        if colors[a] == c2 or colors[b] == c2 or colors[c] == c2 or colors[e] == c2:
            ca = c1 if colors[a] == c2 else colors[a]
            cb = c1 if colors[b] == c2 else colors[b]
            cc = c1 if colors[c] == c2 else colors[c]
            ce = c1 if colors[e] == c2 else colors[e]
            if len({ca, cb, cc, ce}) < 3:
                return False
    return True

def _minimize_colors(colors, progs, deadline):
    """Iteratively try to merge colors to reduce count."""
    colors = list(colors)
    unique = sorted(set(colors))
    # Remap to 0..k-1
    mapping = {c: i for i, c in enumerate(unique)}
    colors = [mapping[c] for c in colors]
    k = len(unique)
    
    # Repeatedly try to merge pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c1 in range(k):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, k):
                if _try_merge(colors, c1, c2, progs):
                    # Perform merge
                    for i in range(N):
                        if colors[i] == c2:
                            colors[i] = c1
                    # Remap to compact
                    present = sorted(set(colors))
                    new_map = {c: i for i, c in enumerate(present)}
                    colors = [new_map[c] for c in colors]
                    k = len(present)
                    improved = True
                    break
            if improved:
                break
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    progs = _build_progressions()
    
    # Try greedy with increasing number of colors
    best_result = None
    for k in range(3, 20):
        if time.time() > deadline - 10:
            break
        colors = _greedy_with_k_colors(k, progs)
        if colors is not None:
            if _check_valid_fast(colors, progs):
                # Try to minimize
                optimized = _minimize_colors(colors, progs, deadline - 2)
                if _check_valid_fast(optimized, progs):
                    cnt = _count_colors(optimized)
                    if best_result is None or cnt < _count_colors(best_result):
                        best_result = optimized
                    return optimized
                else:
                    if best_result is None or _count_colors(colors) < _count_colors(best_result):
                        best_result = colors
                    return colors
    
    # Fallback: use many colors
    if best_result is not None:
        return best_result
    
    # Emergency: assign distinct colors
    return list(range(N))
# EVOLVE-BLOCK-END
