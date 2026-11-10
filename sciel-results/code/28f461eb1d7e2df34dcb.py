"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

def _greedy_construct(seed_order=None):
    """Greedy construction with optional custom modulus ordering."""
    selected = []
    used_moduli = set()
    
    if seed_order is None:
        # Default: small m first
        moduli_to_try = list(range(20, 201))
    else:
        moduli_to_try = seed_order
    
    for m in moduli_to_try:
        if m in used_moduli:
            continue
        r = _smallest_residue(m, selected)
        if r is not None:
            selected.append((m, r))
            used_moduli.add(m)
    
    return selected

def _local_search(best, best_val, deadline):
    """Aggressive local search with 1-swap, 2-swap, and residue reassignment moves."""
    all_moduli = list(range(20, 201))
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline:
        improved = False
        sel = best[:]
        sel_moduli = set(m for (m, r) in sel)
        add_candidates = [m for m in all_moduli if m not in sel_moduli]
        drop_candidates = sorted(sel, key=lambda p: -p[0])
        
        # (a) 1-swap: drop one large m, add one small m
        for (dm, dr) in drop_candidates:
            if time.time() > deadline:
                break
            new_sel = [p for p in sel if p[0] != dm]
            for am in add_candidates:
                if time.time() > deadline:
                    break
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
        
        # (b) 2-swap: drop two large m, add two small m
        if not improved and time.time() < deadline:
            sel = best[:]
            sel_moduli = set(m for (m, r) in sel)
            add_candidates = [m for m in all_moduli if m not in sel_moduli]
            drop_candidates = sorted(sel, key=lambda p: -p[0])
            
            for i in range(len(drop_candidates)):
                if time.time() > deadline:
                    break
                dm1, dr1 = drop_candidates[i]
                for j in range(i + 1, len(drop_candidates)):
                    if time.time() > deadline:
                        break
                    dm2, dr2 = drop_candidates[j]
                    new_sel = [p for p in sel if p[0] not in (dm1, dm2)]
                    gain_needed = 1.0/dm1 + 1.0/dm2 - 1e-15
                    for am1 in add_candidates:
                        if time.time() > deadline:
                            break
                        if 1.0/am1 >= gain_needed:
                            continue
                        ar1 = _smallest_residue(am1, new_sel)
                        if ar1 is None:
                            continue
                        trial1 = new_sel + [(am1, ar1)]
                        for am2 in add_candidates:
                            if time.time() > deadline:
                                break
                            if am2 <= am1:
                                continue
                            if 1.0/am1 + 1.0/am2 >= gain_needed:
                                continue
                            ar2 = _smallest_residue(am2, trial1)
                            if ar2 is None:
                                continue
                            cand = trial1 + [(am2, ar2)]
                            cv = _score(cand)
                            if cv > best_val + 1e-15:
                                best = cand
                                best_val = cv
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # (c) Residue reassignment to unlock smaller moduli
        if not improved and time.time() < deadline:
            sel = best[:]
            sel_moduli = set(m for (m, r) in sel)
            add_candidates = [m for m in all_moduli if m not in sel_moduli]
            for idx, (sm, sr) in enumerate(sel):
                if time.time() > deadline:
                    break
                # Try a few random alternative residues, not all
                tried = 0
                for _ in range(min(sm, 20)):
                    if time.time() > deadline:
                        break
                    nr = random.randint(0, sm - 1)
                    if nr == sr:
                        continue
                    trial = sel[:idx] + [(sm, nr)] + sel[idx+1:]
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
                
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    return best, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # ---- Phase 1: Multiple greedy constructions with different orderings ----
    best = None
    best_val = 0.0
    
    # Greedy 1: Small m first
    cand = _greedy_construct()
    val = _score(cand)
    if val > best_val:
        best = cand
        best_val = val
    
    # Greedy 2: Small m first but skip some to allow better packing
    if time.time() < deadline:
        # Try greedy with slight randomization in order
        for trial in range(5):
            if time.time() > deadline:
                break
            order = list(range(20, 201))
            random.shuffle(order[10:])  # Keep small moduli in order, shuffle larger
            cand = _greedy_construct(order)
            val = _score(cand)
            if val > best_val:
                best = cand
                best_val = val
    
    # ---- Phase 2: Local search ----
    best, best_val = _local_search(best, best_val, deadline)
    
    # ---- Phase 3: More local search if time remains ----
    if time.time() < deadline - 10:
        best, best_val = _local_search(best, best_val, deadline)
    
    # ---- Return as list of [m, r] ----
    return [[m, r] for (m, r) in best]
# EVOLVE-BLOCK-END
