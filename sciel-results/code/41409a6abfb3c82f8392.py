"""Direct approach for minimum maximal no-isosceles set."""
N = 16
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
TOTAL = N * N

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

# Precompute iso_third[i][j] = bool array length TOTAL: c such that {i,j,c} isosceles
def _precompute():
    pts = [_idx_point(i) for i in range(TOTAL)]
    iso = np.zeros((TOTAL, TOTAL, TOTAL), dtype=bool)
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                continue
            a = pts[i]; b = pts[j]
            da = (a[0]-b[0])**2 + (a[1]-b[1])**2
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                c = pts[k]
                db = (a[0]-c[0])**2 + (a[1]-c[1])**2
                dc = (b[0]-c[0])**2 + (b[1]-c[1])**2
                if da == db or da == dc or db == dc:
                    iso[i, j, k] = True
    return iso

_iso_third = _precompute()

# Precompute: for each point p, the set of points c that are "blocked" by p alone
# blocked_by_single[p] = set of c where iso[p][p][c] or iso[p][c][p]... 
# Actually, a single point doesn't block anything by itself.
# What matters: for a set S, point c is blocked if exists p,q in S such that iso[p][q][c].
# Let's precompute: conflict[p][q] = set of c such that {p,q,c} is isosceles
# This is just the union of iso[p][q] and iso[q][p] (but they should be the same by symmetry of the isosceles condition)
# Actually iso[p][q][c] means {p,q,c} is isosceles, which is symmetric in p,q.
# So conflict[p][q] = iso[p][q] (as a boolean vector over c)

def _compute_blocked(chosen):
    """Given a list of chosen point indices, compute the blocked mask."""
    if not chosen:
        return np.zeros(TOTAL, dtype=bool)
    blocked = np.zeros(TOTAL, dtype=bool)
    n = len(chosen)
    for i in range(n):
        for j in range(i+1, n):
            p = chosen[i]
            q = chosen[j]
            blocked |= _iso_third[p, q]
    blocked[chosen] = True
    return blocked

def _is_maximal(chosen, blocked):
    """Check if no unchosen point can be added (i.e., all unchosen are blocked)."""
    if not chosen:
        return True
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[chosen] = True
    unchosen = ~cmask
    return not (~blocked[unchosen]).any()

def _greedy(order):
    """order: list of point indices. Returns (chosen_list, blocked_mask)."""
    chosen = []
    blocked = np.zeros(TOTAL, dtype=bool)
    for p in order:
        if blocked[p]:
            continue
        # Add p: update blocked
        for q in chosen:
            blocked |= _iso_third[p, q]
        blocked[p] = True
        chosen.append(p)
    return chosen, blocked

def _shrink_inclusion_minimal(chosen, blocked):
    """Repeatedly remove any point whose removal keeps the set maximal and no-isosceles.
    Returns (new_chosen, new_blocked)."""
    changed = True
    while changed:
        changed = False
        n = len(chosen)
        for idx in range(n):
            p = chosen[idx]
            # Try removing p
            new_chosen = chosen[:idx] + chosen[idx+1:]
            if not new_chosen:
                new_blocked = np.zeros(TOTAL, dtype=bool)
            else:
                new_blocked = _compute_blocked(new_chosen)
            if _is_maximal(new_chosen, new_blocked):
                chosen = new_chosen
                blocked = new_blocked
                changed = True
                break
    return chosen, blocked

def _perturb_repair(chosen, blocked, n_remove, rng):
    """Remove n_remove random points from chosen, then greedily add back to restore maximality.
    Returns (new_chosen, new_blocked) or None if failed."""
    if len(chosen) <= n_remove:
        return None
    # Pick n_remove random points to remove
    to_remove = set(rng.sample(chosen, n_remove))
    new_chosen = [p for p in chosen if p not in to_remove]
    new_blocked = _compute_blocked(new_chosen)
    
    # Now greedily add blocked points to restore maximality
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[new_chosen] = True
    unchosen = ~cmask
    unblocked = unchosen & ~new_blocked
    
    if unblocked.any():
        # Add them in random order
        candidates = np.where(unblocked)[0].tolist()
        rng.shuffle(candidates)
        for p in candidates:
            if new_blocked[p]:
                continue
            # Add p
            cmask[p] = True
            for q in new_chosen:
                new_blocked |= _iso_third[p, q]
            new_blocked[p] = True
            new_chosen.append(p)
    
    if not _is_maximal(new_chosen, new_blocked):
        return None
    return new_chosen, new_blocked

def _greedy_with_conflict_order(chosen_set, rng):
    """Given a starting set, try to build a maximal set by greedily adding points
    that block the most additional points (conflict-driven ordering)."""
    chosen = list(chosen_set)
    blocked = _compute_blocked(chosen)
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[chosen] = True
    
    # Get all unblocked, unchosen points
    unchosen = ~cmask
    unblocked = unchosen & ~blocked
    
    if not unblocked.any():
        return chosen, blocked
    
    candidates = np.where(unblocked)[0].tolist()
    rng.shuffle(candidates)
    
    for p in candidates:
        if blocked[p]:
            continue
        # Compute how many new points this would block
        new_blocks = 0
        for q in chosen:
            new_blocks += np.sum(_iso_third[p, q] & ~blocked)
        
        # Accept if it helps (or randomly for diversity)
        if new_blocks > 0 or rng.random() < 0.3:
            for q in chosen:
                blocked |= _iso_third[p, q]
            blocked[p] = True
            chosen.append(p)
            cmask[p] = True
    
    return chosen, blocked

def construct():
    all_idx = list(range(TOTAL))
    best = None
    best_size = TOTAL + 1
    start = time.time()
    deadline = start + 320  # safety margin before 330s kill

    rng = random.Random(42)
    
    # First try a deterministic ordering (row-major) as baseline
    order = all_idx[:]
    chosen, blocked = _greedy(order)
    if _is_maximal(chosen, blocked):
        # Shrink
        chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    # Try several different deterministic orderings
    # Diagonal ordering
    diag_order = []
    for d in range(2 * N - 1):
        for x in range(N):
            y = d - x
            if 0 <= y < N:
                diag_order.append(x * N + y)
    chosen, blocked = _greedy(diag_order)
    if _is_maximal(chosen, blocked):
        chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    # Reverse row-major
    rev_order = all_idx[::-1]
    chosen, blocked = _greedy(rev_order)
    if _is_maximal(chosen, blocked):
        chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    # Iterated local search
    current_chosen = best[:] if best else None
    current_blocked = blocked if best else None
    
    for attempt in range(100000):
        if time.time() > deadline:
            break
        
        if current_chosen is None:
            # Start fresh with random greedy
            order = all_idx[:]
            rng.shuffle(order)
            chosen, blocked = _greedy(order)
            if not _is_maximal(chosen, blocked):
                continue
            chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen[:]
            current_chosen = chosen[:]
            current_blocked = blocked.copy()
            continue
        
        # Perturb and repair from current
        # Try different perturbation sizes
        n_remove = rng.randint(1, max(1, len(current_chosen) // 3))
        result = _perturb_repair(current_chosen, current_blocked, n_remove, rng)
        
        if result is not None:
            new_chosen, new_blocked = result
            # Shrink the new solution
            new_chosen, new_blocked = _shrink_inclusion_minimal(new_chosen, new_blocked)
            
            if len(new_chosen) < best_size:
                best_size = len(new_chosen)
                best = new_chosen[:]
                current_chosen = new_chosen[:]
                current_blocked = new_blocked.copy()
            elif len(new_chosen) < len(current_chosen):
                # Accept if it improves over current (even if not global best)
                current_chosen = new_chosen[:]
                current_blocked = new_blocked.copy()
            
            if best_size <= 10:
                break
        else:
            # Failed to repair; occasionally restart
            if rng.random() < 0.15:
                current_chosen = None
                current_blocked = None

    if best is None:
        best = all_idx[:]
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
