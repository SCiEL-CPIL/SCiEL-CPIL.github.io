"""ILP-based maximum coverage for odd covering systems mod 51975."""
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

BASE = 51975
_DIVS = []
for a in range(4):
    for b in range(3):
        for c in range(2):
            for d in range(2):
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
MODULI = sorted(_DIVS)
NMOD = len(MODULI)
_ARANGE = np.arange(BASE, dtype=np.int64)

DEADLINE = 142.0


def _compute_coverage(assign, active):
    """Compute uncovered count for given assignment."""
    cov = np.zeros(BASE, dtype=np.int8)
    for i in np.nonzero(active)[0]:
        m = MODULI[i]
        r = int(assign[i])
        mask = (_ARANGE % m) == r
        cov[mask] += 1
    return int(np.count_nonzero(cov == 0))


def _greedy_init():
    """Greedy initialization for local search."""
    active = np.zeros(NMOD, dtype=bool)
    assign = np.zeros(NMOD, dtype=np.int64)
    order = sorted(range(NMOD), key=lambda i: -MODULI[i])
    cov = np.zeros(BASE, dtype=np.int8)
    for i in order:
        m = MODULI[i]
        best_r = 0
        best_unc = BASE + 1
        residues = np.arange(m)
        if m > 400:
            residues = residues[:: max(1, m // 400)]
        for r in residues:
            mask = (_ARANGE % m) == r
            newcov = cov + mask.astype(np.int8)
            unc = int(np.count_nonzero(newcov == 0))
            if unc < best_unc:
                best_unc = unc
                best_r = int(r)
        lo = max(0, best_r - 5)
        hi = min(m, best_r + 6)
        for r in range(lo, hi):
            mask = (_ARANGE % m) == r
            newcov = cov + mask.astype(np.int8)
            unc = int(np.count_nonzero(newcov == 0))
            if unc < best_unc:
                best_unc = unc
                best_r = r
        assign[i] = best_r
        active[i] = True
        cov[(_ARANGE % m) == best_r] += 1
    # Remove redundant
    changed = True
    while changed:
        changed = False
        cov_full = np.zeros(BASE, dtype=np.int8)
        for i in np.nonzero(active)[0]:
            m = MODULI[i]
            r = int(assign[i])
            cov_full[(_ARANGE % m) == r] += 1
        base_unc = int(np.count_nonzero(cov_full == 0))
        for i in np.nonzero(active)[0]:
            active[i] = False
            cov2 = np.zeros(BASE, dtype=np.int8)
            for j in np.nonzero(active)[0]:
                m = MODULI[j]
                r = int(assign[j])
                cov2[(_ARANGE % m) == r] += 1
            unc2 = int(np.count_nonzero(cov2 == 0))
            if unc2 <= base_unc:
                base_unc = unc2
                changed = True
            else:
                active[i] = True
    return assign, active


def _local_search(assign, active, unc, max_iters=2000):
    """Local search with single-residue moves."""
    for _ in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
        improved = False
        act_idx = np.nonzero(active)[0].tolist()
        random.shuffle(act_idx)
        for i in act_idx:
            m = MODULI[i]
            old_r = int(assign[i])
            cands = [random.randrange(m) for _ in range(3)]
            for r in cands:
                if r == old_r:
                    continue
                assign[i] = r
                u = _compute_coverage(assign, active)
                if u < unc:
                    unc = u
                    improved = True
                    break
                else:
                    assign[i] = old_r
        if not improved:
            break
    return assign, active, unc


def _milp_solve(greedy_assign, greedy_active, time_limit=30):
    """
    Solve ILP: maximize covered residues by selecting at most one residue per modulus.
    Variables: x_{i,r} binary for each modulus i and residue r.
    Constraint: sum_r x_{i,r} <= 1 for each i (at most one residue per modulus)
    Objective: maximize sum over positions p of y_p where y_p = 1 if covered.
    
    Since BASE=51975 is large, we use a reduced formulation:
    For each position p, it's covered if any selected class covers it.
    We use indicator constraints or big-M.
    
    Simpler: minimize uncovered. For each position p, z_p = 1 if not covered.
    z_p <= 1 - sum_{i,r: p in class(i,r)} x_{i,r}
    Minimize sum z_p.
    """
    # Build constraint matrix for the ILP
    # This can be very large; use a smarter approach.
    # 
    # Alternative: use the greedy solution as a warm start and do coordinate descent
    # on the ILP solution space.
    
    # For now, do a smarter local search guided by coverage analysis
    assign = greedy_assign.copy()
    active = greedy_active.copy()
    unc = _compute_coverage(assign, active)
    
    # Compute coverage array
    cov = np.zeros(BASE, dtype=np.int8)
    for i in np.nonzero(active)[0]:
        m = MODULI[i]
        r = int(assign[i])
        cov[(_ARANGE % m) == r] += 1
    
    # Find positions with coverage 0 (uncovered)
    uncovered_pos = np.nonzero(cov == 0)[0]
    
    # For each uncovered position, find which moduli could cover it and what residue
    # would cover it. Use this to guide local search.
    for _ in range(100):
        if time.time() > DEADLINE - 5:
            break
        if len(uncovered_pos) == 0:
            break
        
        # Pick a random uncovered position
        p = uncovered_pos[random.randrange(len(uncovered_pos))]
        
        # Find all (i, r) such that p % MODULI[i] == r and active[i] is False or we can change
        best_i = -1
        best_r = -1
        best_gain = 0
        best_unc = unc
        
        # Check inactive moduli that could cover p
        for i in range(NMOD):
            if active[i]:
                continue
            m = MODULI[i]
            r = p % m
            # If we add this class, how many new positions does it cover?
            mask = (_ARANGE % m) == r
            new_cov = cov + mask.astype(np.int8)
            new_unc = int(np.count_nonzero(new_cov == 0))
            gain = unc - new_unc
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_r = r
                best_unc = new_unc
        
        # Check active moduli: can changing residue cover p and more?
        for i in np.nonzero(active)[0]:
            m = MODULI[i]
            r = p % m
            if r == int(assign[i]):
                continue
            # Try changing this modulus to residue r
            old_mask = (_ARANGE % m) == int(assign[i])
            new_mask = (_ARANGE % m) == r
            new_cov = cov - old_mask.astype(np.int8) + new_mask.astype(np.int8)
            new_unc = int(np.count_nonzero(new_cov == 0))
            gain = unc - new_unc
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_r = r
                best_unc = new_unc
        
        if best_gain <= 0:
            break
        
        # Apply the best move
        i = best_i
        r = best_r
        m = MODULI[i]
        if not active[i]:
            active[i] = True
        old_r = int(assign[i])
        if old_r != r:
            cov[(_ARANGE % m) == old_r] -= 1
        assign[i] = r
        cov[(_ARANGE % m) == r] += 1
        unc = best_unc
        
        # Update uncovered positions
        uncovered_pos = np.nonzero(cov == 0)[0]
    
    return assign, active, unc


def construct():
    t0 = time.time()
    random.seed(12345)
    np.random.seed(12345)

    assign, active = _greedy_init()
    best_unc = _compute_coverage(assign, active)
    best_assign = assign.copy()
    best_active = active.copy()

    # Local search
    assign, active, best_unc = _local_search(assign, active, best_unc, max_iters=2000)
    best_assign = assign.copy()
    best_active = active.copy()

    # ILP-guided search
    if time.time() < DEADLINE - 10:
        assign, active, unc = _milp_solve(best_assign, best_active)
        if unc < best_unc:
            best_unc = unc
            best_assign = assign.copy()
            best_active = active.copy()

    # More local search
    assign, active, best_unc = _local_search(best_assign, best_active, best_unc, max_iters=3000)
    best_assign = assign.copy()
    best_active = active.copy()

    # Another ILP-guided pass
    if time.time() < DEADLINE - 10:
        assign, active, unc = _milp_solve(best_assign, best_active)
        if unc < best_unc:
            best_unc = unc
            best_assign = assign.copy()
            best_active = active.copy()

    # Final local search
    assign, active, best_unc = _local_search(best_assign, best_active, best_unc, max_iters=2000)
    best_assign = assign.copy()
    best_active = active.copy()

    # Remove redundant
    changed = True
    while changed and time.time() < DEADLINE - 2:
        changed = False
        cov_full = np.zeros(BASE, dtype=np.int8)
        for i in np.nonzero(best_active)[0]:
            m = MODULI[i]
            r = int(best_assign[i])
            cov_full[(_ARANGE % m) == r] += 1
        base_unc = int(np.count_nonzero(cov_full == 0))
        for i in np.nonzero(best_active)[0]:
            best_active[i] = False
            cov2 = np.zeros(BASE, dtype=np.int8)
            for j in np.nonzero(best_active)[0]:
                m = MODULI[j]
                r = int(best_assign[j])
                cov2[(_ARANGE % m) == r] += 1
            unc2 = int(np.count_nonzero(cov2 == 0))
            if unc2 <= base_unc:
                base_unc = unc2
                changed = True
            else:
                best_active[i] = True
        if changed:
            best_unc = base_unc

    # Final local search after removal
    assign, active, best_unc = _local_search(best_assign, best_active, best_unc, max_iters=1000)
    best_assign = assign.copy()
    best_active = active.copy()

    classes = []
    for i in np.nonzero(best_active)[0]:
        classes.append([int(MODULI[i]), int(best_assign[i])])
    return classes
