"""Rewritten: fast NumPy scoring + greedy/SA/local-search + coverage-removal."""
import time
import random
import numpy as np

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
COUNTS = np.array([BASE // m for m in MODULI], dtype=np.int64)

_ARANGE = np.arange(BASE, dtype=np.int64)
_MOD_ARR = [(_ARANGE % m) for m in MODULI]  # each length BASE, int64

DEADLINE = 142.0


def _coverage_from(assign, active):
    """Return coverage count array (int8) and uncovered count for given assign (len NMOD) and active (bool len NMOD)."""
    cov = np.zeros(BASE, dtype=np.int8)
    for i in np.nonzero(active)[0]:
        m = MODULI[i]
        r = int(assign[i])
        cov[_MOD_ARR[i] == r] += 1
    unc = int(np.count_nonzero(cov == 0))
    return cov, unc


def _remove_redundant(assign, active, cov, unc):
    """Remove active classes whose removal does not increase uncovered count.
    Updates cov and unc in place (via return)."""
    changed = True
    while changed:
        changed = False
        for i in np.nonzero(active)[0]:
            r = int(assign[i])
            mask = _MOD_ARR[i] == r
            # If we remove this, positions with cov==1 in mask become 0 (uncovered).
            # If cov > 1, they remain covered.
            # So uncovered count increases by count of (cov[mask] == 1).
            newly_uncovered = int(np.sum(cov[mask] == 1))
            if newly_uncovered == 0:
                # Safe to remove
                cov[mask] -= 1
                active[i] = False
                changed = True
    return active, cov, unc


def _greedy_init():
    """Greedy: add moduli one by one, each choosing residue minimizing uncovered."""
    active = np.zeros(NMOD, dtype=bool)
    assign = np.zeros(NMOD, dtype=np.int64)
    # Start with the largest moduli (cover most at once) but choose residue well.
    # Use a heuristic order: larger moduli first to reduce uncovered faster.
    order = sorted(range(NMOD), key=lambda i: -MODULI[i])
    cov = np.zeros(BASE, dtype=np.int8)
    unc = BASE
    for i in order:
        m = MODULI[i]
        best_r = 0
        best_unc = BASE + 1
        # Evaluate residues; for large m there are many residues. Sample if too many.
        residues = np.arange(m)
        if m > 400:
            residues = residues[:: max(1, m // 400)]
        for r in residues:
            mask = _MOD_ARR[i] == r
            # new_unc = unc - count(cov==0 in mask)
            newly_covered = int(np.sum(cov[mask] == 0))
            u = unc - newly_covered
            if u < best_unc:
                best_unc = u
                best_r = int(r)
        # Refine around best_r among full residues near it
        lo = max(0, best_r - 5)
        hi = min(m, best_r + 6)
        for r in range(lo, hi):
            mask = _MOD_ARR[i] == r
            newly_covered = int(np.sum(cov[mask] == 0))
            u = unc - newly_covered
            if u < best_unc:
                best_unc = u
                best_r = r
        assign[i] = best_r
        active[i] = True
        mask = _MOD_ARR[i] == best_r
        cov[mask] += 1
        unc = best_unc
    
    active, cov, unc = _remove_redundant(assign, active, cov, unc)
    return assign, active, cov, unc


def _apply_move(cov, unc, i, old_r, new_r):
    """Apply a move changing residue of modulus i from old_r to new_r.
    Returns new unc. Updates cov in place."""
    if old_r == new_r:
        return unc
    
    mask_old = _MOD_ARR[i] == old_r
    mask_new = _MOD_ARR[i] == new_r
    
    # Positions that lose coverage
    losing = mask_old & ~mask_new
    # Positions that gain coverage
    gaining = mask_new & ~mask_old
    
    # Calculate change in uncovered count
    # Losing positions: if cov was 1, it becomes 0 (uncovered +1)
    newly_uncovered = int(np.sum(cov[losing] == 1))
    # Gaining positions: if cov was 0, it becomes 1 (uncovered -1)
    newly_covered = int(np.sum(cov[gaining] == 0))
    
    # Update cov
    cov[losing] -= 1
    cov[gaining] += 1
    
    unc += newly_uncovered - newly_covered
    return unc


def construct():
    t0 = time.time()
    random.seed(12345)
    np.random.seed(12345)

    assign, active, cov, unc = _greedy_init()
    best_unc = unc
    best_assign = assign.copy()
    best_active = active.copy()
    best_cov = cov.copy()

    # Local search: single-residue moves on active moduli
    def local_search(assign, active, cov, unc, max_iters=2000):
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            act_idx = np.nonzero(active)[0]
            if len(act_idx) == 0: break
            random.shuffle(act_idx.tolist())
            for i in act_idx:
                m = MODULI[i]
                old_r = int(assign[i])
                # Try a few candidate residues
                cands = [random.randrange(m) for _ in range(3)]
                for r in cands:
                    if r == old_r:
                        continue
                    # Apply
                    new_unc = _apply_move(cov, unc, i, old_r, r)
                    assign[i] = r
                    if new_unc < unc:
                        unc = new_unc
                        improved = True
                        break
                    else:
                        # Revert
                        _apply_move(cov, unc, i, r, old_r)
                        assign[i] = old_r
            if not improved:
                break
        return assign, active, cov, unc

    assign, active, cov, unc = local_search(assign, active, cov, unc)
    if unc < best_unc:
        best_unc = unc
        best_assign = assign.copy()
        best_active = active.copy()
        best_cov = cov.copy()

    # Simulated annealing with restarts
    T = 200.0
    cool = 0.999
    while time.time() < DEADLINE - 5:
        # Restart from best
        assign = best_assign.copy()
        active = best_active.copy()
        cov = best_cov.copy()
        unc = best_unc
        
        # Anneal
        for step in range(3000):
            if time.time() > DEADLINE - 2:
                break
            act_idx = np.nonzero(active)[0]
            if len(act_idx) == 0:
                break
            i = act_idx[random.randrange(len(act_idx))]
            m = MODULI[i]
            old_r = int(assign[i])
            new_r = random.randrange(m)
            if new_r == old_r:
                continue
            
            new_unc = _apply_move(cov, unc, i, old_r, new_r)
            assign[i] = new_r
            
            d = new_unc - unc
            if d <= 0 or random.random() < np.exp(-d / max(T, 1e-6)):
                unc = new_unc
                if unc < best_unc:
                    best_unc = unc
                    best_assign = assign.copy()
                    best_active = active.copy()
                    best_cov = cov.copy()
            else:
                # Revert
                _apply_move(cov, unc, i, new_r, old_r)
                assign[i] = old_r
            
            T *= cool
        
        # Local search after each anneal
        assign, active, cov, unc = local_search(assign, active, cov, unc, max_iters=500)
        if unc < best_unc:
            best_unc = unc
            best_assign = assign.copy()
            best_active = active.copy()
            best_cov = cov.copy()
            
        # Occasionally try adding a previously-inactive modulus
        if time.time() < DEADLINE - 10:
            inactive = np.nonzero(~best_active)[0]
            if len(inactive) > 0:
                i = inactive[random.randrange(len(inactive))]
                m = MODULI[i]
                # pick best residue for this modulus given current coverage
                cov_trial = best_cov.copy()
                best_r = 0
                best_u = best_unc
                residues = np.arange(m)
                if m > 300:
                    residues = residues[:: max(1, m // 300)]
                for r in residues:
                    mask = _MOD_ARR[i] == r
                    newly_covered = int(np.sum(cov_trial[mask] == 0))
                    u = best_unc - newly_covered
                    if u < best_u:
                        best_u = u
                        best_r = int(r)
                
                if best_u < best_unc:
                    trial_active = best_active.copy()
                    trial_active[i] = True
                    trial_assign = best_assign.copy()
                    trial_assign[i] = best_r
                    # Apply the move to get correct cov
                    mask = _MOD_ARR[i] == best_r
                    cov_trial[mask] += 1
                    trial_active, cov_trial, u2 = _remove_redundant(trial_assign, trial_active, cov_trial, best_u)
                    
                    if u2 < best_unc:
                        best_unc = u2
                        best_assign = trial_assign
                        best_active = trial_active
                        best_cov = cov_trial

    # Final redundant removal and local search
    best_active, best_cov, best_unc = _remove_redundant(best_assign, best_active, best_cov, best_unc)
    best_assign, best_active, best_cov, best_unc = local_search(best_assign, best_active, best_cov, best_unc, max_iters=1000)
    best_active, best_cov, best_unc = _remove_redundant(best_assign, best_active, best_cov, best_unc)

    classes = []
    for i in np.nonzero(best_active)[0]:
        classes.append([int(MODULI[i]), int(best_assign[i])])
    return classes
