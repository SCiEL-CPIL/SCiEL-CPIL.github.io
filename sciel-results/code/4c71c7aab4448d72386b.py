"""Bidirectional snake-in-the-box growth with Warnsdorff's rule for Q_9."""
import time

N = 9
TOTAL = 1 << N
MASK = TOTAL - 1

# Precompute neighbor lists for fast lookup
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(TOTAL)]

# Precompute popcount for all masks (bit_count is available, but let's use it directly)
# We'll use int.bit_count() which is fast in Python 3.8+


def _elapsed(deadline):
    return time.time() >= deadline


def _valid_extension(v, used_set, path, exclude_tail_idx=None):
    """Check if v can be added to the path at the current end.
    
    v must not be in used, and must differ from all non-adjacent path vertices
    by at least 2 bits. The vertex adjacent to the end (path[-1] or path[0]) 
    is exempt since it differs by exactly 1 bit.
    """
    if v in used_set:
        return False
    # v must differ from all path vertices except the one it connects to
    # by at least 2 bits. The connecting vertex differs by exactly 1 bit (guaranteed).
    # We need to check all other vertices.
    # Optimization: only check vertices that are NOT the immediate predecessor.
    # For a new vertex at the tail end (connecting to path[-1]), 
    # we check path[0:-1] (all except the last).
    # For a new vertex at the head end (connecting to path[0]),
    # we check path[1:] (all except the first).
    for i, p in enumerate(path):
        # Skip the vertex we're connecting to
        if exclude_tail_idx is not None and i == exclude_tail_idx:
            continue
        if (v ^ p).bit_count() == 1:
            return False
    return True


def _count_options(v, used_set, path, connecting_idx):
    """Count how many valid extensions exist from vertex v if it were added
    connecting to path[connecting_idx]. This is used for Warnsdorff's rule."""
    if v in used_set:
        return 0
    # Check v against all path vertices except the connecting one
    for i, p in enumerate(path):
        if i == connecting_idx:
            continue
        if (v ^ p).bit_count() == 1:
            return 0
    # Count neighbors of v that could be added next (not in used, and valid)
    count = 0
    for nb in _NEIGHBORS[v]:
        if nb in used_set:
            continue
        # nb must differ from all path vertices except v (which it will by 1 bit)
        # Actually nb connects to v, so we check nb against path[0:-1] if adding to tail,
        # or path[1:] if adding to head. But since we're just counting options from v,
        # we check nb against all path vertices except v itself (v isn't in path yet).
        # nb differs from v by 1 bit, so we check nb against all current path vertices.
        ok = True
        for p in path:
            if (nb ^ p).bit_count() == 1:
                ok = False
                break
        if ok:
            count += 1
    return count


def _grow_bidirectional(path, used_set, deadline):
    """Grow the path bidirectionally using Warnsdorff's rule."""
    while not _elapsed(deadline):
        # Try to extend at both ends, pick the more constrained one
        best_end = None
        best_choice = None
        best_score = -1
        
        for end in (0, 1):  # 0 = head, 1 = tail
            cur = path[0] if end == 0 else path[-1]
            connect_idx = 0 if end == 0 else len(path) - 1
            
            best_for_end = None
            best_for_end_score = -1
            
            for nb in _NEIGHBORS[cur]:
                if nb in used_set:
                    continue
                # Check validity: nb must differ from all path vertices except cur
                valid = True
                for i, p in enumerate(path):
                    if i == connect_idx:
                        continue
                    if (nb ^ p).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Warnsdorff: count how many options nb would have
                score = _count_options(nb, used_set, path, connect_idx)
                # We want the MOST constrained (fewest options) to avoid dead ends
                # But we also want to avoid 0 (dead end). Prefer low positive counts.
                # Actually, Warnsdorff's rule for longest path: choose the vertex 
                # with the FEWEST available next moves (to avoid trapping).
                # But if score is 0, it's a dead end - we should avoid it unless forced.
                # Let's use: prefer lowest score, but avoid 0 if possible.
                if best_for_end is None or score < best_for_end_score:
                    best_for_end_score = score
                    best_for_end = nb
            
            if best_for_end is not None:
                # Combine: prefer the end that leads to a more constrained choice
                # Use negative score so lower is better, but penalize 0
                combined = best_for_end_score if best_for_end_score > 0 else 1000
                if combined < best_score or best_end is None:
                    best_score = combined
                    best_end = end
                    best_choice = best_for_end
        
        if best_end is None:
            break
        
        # Add the chosen vertex
        path.append(best_choice) if best_end == 1 else path.insert(0, best_choice)
        used_set.add(best_choice)
        
        if _elapsed(deadline):
            break
    
    return path


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with a greedy seed path
    path = [0]
    used = {0}
    
    # Forward greedy to get an initial path
    while not _elapsed(deadline):
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check against all path vertices except the last
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    
    if _elapsed(deadline):
        return path
    
    # Now grow bidirectionally with Warnsdorff
    path = _grow_bidirectional(path, used, deadline)
    
    return path
