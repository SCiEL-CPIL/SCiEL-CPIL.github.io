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

def _compute_blocked(chosen):
    """Given a list of chosen point indices, compute the blocked mask."""
    if not chosen:
        return np.zeros(TOTAL, dtype=bool)
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[chosen] = True
    blocked = np.zeros(TOTAL, dtype=bool)
    for p in chosen:
        sub = _iso_third[p]  # shape (TOTAL, TOTAL): [q][c]
        row = sub[cmask]  # shape (n_chosen, TOTAL)
        if row.size:
            blocked |= row.any(axis=0)
    blocked[chosen] = True
    return blocked

def _is_maximal(chosen, blocked):
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[chosen] = True
    unchosen = ~cmask
    return not (blocked[unchosen] == False).any()

def _greedy(order):
    """order: list of point indices. Returns (chosen_list, blocked_mask)."""
    chosen = []
    blocked = np.zeros(TOTAL, dtype=bool)
    for p in order:
        if blocked[p]:
            continue
        cmask = np.zeros(TOTAL, dtype=bool)
        cmask[chosen] = True
        sub = _iso_third[p]
        row = sub[cmask]
        if row.size:
            blocked |= row.any(axis=0)
        blocked[p] = True
        chosen.append(p)
    return chosen, blocked

def _shrink_inclusion_minimal(chosen, blocked):
    """Repeatedly remove any point whose removal keeps the set maximal and no-isosceles.
    Returns (new_chosen, new_blocked)."""
    changed = True
    while changed:
        changed = False
        for idx in range(len(chosen)):
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
    # Get unblocked points
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
            # Check no-isosceles: p should not form iso with any pair in new_chosen
            # Actually if p is unblocked, it means no q in new_chosen has iso[p][q][p]=True
            # Wait, blocked[p] means exists q in chosen such that iso[p][q][p] or iso[q][p][p]...
            # Let me re-examine: blocked[c] = exists p,q in chosen with iso[p][q][c]
            # So if p is unblocked, adding it won't create iso triangles.
            cmask[p] = True
            sub = _iso_third[p]
            row = sub[cmask]
            if row.size:
                new_blocked |= row.any(axis=0)
            new_blocked[p] = True
            new_chosen.append(p)
    
    if not _is_maximal(new_chosen, new_blocked):
        return None
    return new_chosen, new_blocked

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

    # Iterated local search
    current_chosen = best[:] if best else None
    current_blocked = blocked if best else None
    
    for attempt in range(50000):
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
        n_remove = rng.randint(1, max(1, len(current_chosen) // 4))
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
            if rng.random() < 0.1:
                current_chosen = None
                current_blocked = None

    if best is None:
        best = all_idx[:]
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
