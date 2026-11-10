"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _conflict(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 - r2) % g == 0

def _compatible(m, r, selected):
    for (sm, sr) in selected:
        if _conflict(m, r, sm, sr):
            return False
    return True

def _smallest_residue(m, selected):
    for r in range(m):
        if _compatible(m, r, selected):
            return r
    return None

def _score(pairs):
    return sum(1.0 / m for (m, r) in pairs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # ---- Phase 1: Greedy, small m first, smallest compatible residue ----
    selected = []
    used_moduli = set()
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = _smallest_residue(m, selected)
        if r is not None:
            selected.append((m, r))
            used_moduli.add(m)

    best = selected[:]
    best_val = _score(best)

    # ---- Phase 2: Local search via swaps and residue reassignment ----
    # Build list of all candidate (m, r) pairs we might use.
    all_moduli = list(range(20, 201))

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # (a) Try swapping: remove a selected pair (large m) and add an
        #     unselected pair (small m) if it raises the score.
        sel = best[:]
        sel_moduli = set(m for (m, r) in sel)
        # sort selected by m descending (candidates to drop)
        drop_candidates = sorted(sel, key=lambda p: -p[0])
        # sort unselected by m ascending (candidates to add)
        add_candidates = [m for m in all_moduli if m not in sel_moduli]

        for (dm, dr) in drop_candidates:
            if time.time() > deadline:
                break
            new_sel = [p for p in sel if p[0] != dm]
            for am in add_candidates:
                if time.time() > deadline:
                    break
                # need 1/am > 1/dm to gain, i.e. am < dm
                if am >= dm:
                    break
                ar = _smallest_residue(am, new_sel)
                if ar is None:
                    continue
                cand = new_sel + [(am, ar)]
                cv = _score(cand)
                if cv > best_val + 1e-15:
                    best = cand
                    best_val = cv
                    improved = True
                    sel = best[:]
                    sel_moduli = set(m for (m, r) in sel)
                    drop_candidates = sorted(sel, key=lambda p: -p[0])
                    add_candidates = [m for m in all_moduli if m not in sel_moduli]
                    break

        # (b) Try reassigning a selected residue to a different compatible one
        #     that might unlock a smaller unselected m.
        sel = best[:]
        sel_moduli = set(m for (m, r) in sel)
        add_candidates = [m for m in all_moduli if m not in sel_moduli]
        for idx, (sm, sr) in enumerate(sel):
            if time.time() > deadline:
                break
            # try alternative residues for sm
            for nr in range(sm):
                if nr == sr:
                    continue
                trial = sel[:idx] + [(sm, nr)] + sel[idx+1:]
                # check internal compatibility of trial
                ok = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        if _conflict(trial[i][0], trial[i][1], trial[j][0], trial[j][1]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # try to add the smallest unselected m compatible with trial
                new_moduli = set(m for (m, r) in trial)
                for am in add_candidates:
                    if am in new_moduli:
                        continue
                    ar = _smallest_residue(am, trial)
                    if ar is None:
                        continue
                    cand = trial + [(am, ar)]
                    cv = _score(cand)
                    if cv > best_val + 1e-15:
                        best = cand
                        best_val = cv
                        improved = True
                        break
                if improved:
                    break

    # ---- Return as list of [m, r] ----
    return [[m, r] for (m, r) in best]
# EVOLVE-BLOCK-END
