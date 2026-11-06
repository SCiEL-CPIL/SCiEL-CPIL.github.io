"""Broadband antireflection coating design: recursive admittance (synthetic-layer)
construction + fast vectorized coordinate-descent polish + light structural SA."""
import time
import numpy as np
import random
import math

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0,
                            substrate_index=1.52, materials_dict=None):
    """Reflectance at all wavelengths via characteristic-matrix method (vectorized)."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r * r, dtype=np.float64)
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        new_M11 = M11 * cos_b + M12 * (n * 1.0j * sin_b)
        new_M12 = M11 * (1.0j * sin_b / n) + M12 * cos_b
        new_M21 = M21 * cos_b + M22 * (n * 1.0j * sin_b)
        new_M22 = M21 * (1.0j * sin_b / n) + M22 * cos_b
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return np.real(R)

def _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict):
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index,
                                    substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True

def _clamp_thick(t, max_layer_nm):
    return max(0.0, min(t, max_layer_nm))

def _admittance_stack(n_layers, nL, nH, n_sub, n_amb):
    """Recursive admittance (synthetic-layer) construction.

    Works from the substrate outward. At each step we know the admittance Y of the
    already-designed portion (as seen from that side). We pick the next layer index
    (L or H) and a quarter-wave thickness so that the resulting admittance
    Y' = n / Y moves toward the ambient index n_amb. We choose the layer whose
    quarter-wave transform brings the admittance closest to n_amb.
    Returns (materials, thicknesses) ordered air->substrate.
    """
    # Start: admittance of bare substrate as seen from inside
    Y = n_sub
    mats = []
    thicks = []
    # We build from substrate side toward air. Track running optical thickness budget.
    # Use a central wavelength ~600 nm for quarter-wave thicknesses.
    wl = 600.0
    for _ in range(n_layers):
        # candidate: put L layer (quarter wave) on top of current Y -> new admittance nL / Y
        # candidate: put H layer -> nH / Y
        cand_L = nL / Y
        cand_H = nH / Y
        # pick the one closer to n_amb (target ambient admittance)
        if abs(cand_L - n_amb) <= abs(cand_H - n_amb):
            choose = 'L'
            Y = cand_L
        else:
            choose = 'H'
            Y = cand_H
        n_val = nL if choose == 'L' else nH
        d = wl / (4.0 * n_val)
        mats.append(choose)
        thicks.append(d)
    # Reverse so that air side is first
    mats.reverse()
    thicks.reverse()
    return mats, thicks

def _quarter_wave_stack(n_layers, nL, nH, n_sub, n_amb, start='L', wl=600.0):
    mats = []
    thicks = []
    for i in range(n_layers):
        m = start if (i % 2 == 0) else ('H' if start == 'L' else 'L')
        n_val = nL if m == 'L' else nH
        mats.append(m)
        thicks.append(wl / (4.0 * n_val))
    return mats, thicks

def _make_valid(materials, thicknesses, problem):
    """Clip to constraints: max layers, max total, max layer."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    mats = list(materials)[:max_layers]
    thicks = list(thicknesses)[:max_layers]
    thicks = [_clamp_thick(t, max_layer_nm) for t in thicks]
    total = sum(thicks)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicks = [t * scale for t in thicks]
    # drop any zero thickness layers
    keep_m, keep_t = [], []
    for m, t in zip(mats, thicks):
        if t > 1e-6:
            keep_m.append(m)
            keep_t.append(t)
    return keep_m, keep_t

def _coordinate_descent(materials, thicknesses, problem, wls, materials_dict,
                        ambient_index, substrate_index, deadline, max_rounds=60):
    """Fast coordinate descent with per-layer adaptive step, vectorized eval."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    cur = [float(t) for t in thicknesses]
    cur_score = _evaluate(materials, cur, wls, ambient_index, substrate_index, materials_dict)
    steps = [8.0] * n_layers
    min_step = 0.005
    max_step = 40.0
    factors = [1.0, 0.5, 0.25, 0.125, 2.0, 4.0]

    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_layers):
            best_s = cur_score
            best_t = cur[i]
            for f in factors:
                step = steps[i] * f
                for sign in (1, -1):
                    nt = cur[i] + sign * step
                    if nt < 0.0:
                        nt = 0.0
                    if nt > max_layer_nm:
                        nt = max_layer_nm
                    trial = list(cur)
                    trial[i] = nt
                    if sum(trial) > max_total_nm + 1e-9:
                        continue
                    s = _evaluate(materials, trial, wls, ambient_index, substrate_index, materials_dict)
                    if s < best_s - 1e-12:
                        best_s = s
                        best_t = nt
            if best_t != cur[i]:
                cur[i] = best_t
                cur_score = best_s
                improved = True
        if not improved:
            steps = [s * 0.6 for s in steps]
            if min(steps) < min_step:
                break
        else:
            steps = [min(s * 1.15, max_step) for s in steps]
    return materials, cur

def _nelder_mead(materials, thicknesses, problem, wls, materials_dict,
                 ambient_index, substrate_index, deadline, max_iter=1200):
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    rng = random.Random(7)
    simplex = [list(thicknesses)]
    for i in range(n_layers):
        p = list(thicknesses)
        p[i] = _clamp_thick(p[i] + 12.0 * (1 if rng.random() < 0.5 else -1), max_layer_nm)
        simplex.append(p)
    scores = []
    for sp in simplex:
        scores.append(_evaluate(materials, sp, wls, ambient_index, substrate_index, materials_dict)
                      if _is_valid(materials, sp, problem) else 1e6)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        if abs(scores[-1] - scores[0]) < 1e-14:
            break
        centroid = np.mean([np.array(simplex[i]) for i in range(len(simplex) - 1)], axis=0)
        worst = np.array(simplex[-1])
        ref = np.array([_clamp_thick(v, max_layer_nm) for v in (centroid + alpha * (centroid - worst))])
        ref_score = _evaluate(materials, ref, wls, ambient_index, substrate_index, materials_dict) \
            if _is_valid(materials, ref, problem) else 1e6
        if ref_score < scores[0]:
            exp = np.array([_clamp_thick(v, max_layer_nm) for v in (centroid + gamma * (ref - centroid))])
            exp_score = _evaluate(materials, exp, wls, ambient_index, substrate_index, materials_dict) \
                if _is_valid(materials, exp, problem) else 1e6
            if exp_score < ref_score:
                simplex[-1] = exp.tolist(); scores[-1] = exp_score
            else:
                simplex[-1] = ref.tolist(); scores[-1] = ref_score
        elif ref_score < scores[-2]:
            simplex[-1] = ref.tolist(); scores[-1] = ref_score
        else:
            con = np.array([_clamp_thick(v, max_layer_nm) for v in (centroid + rho * (worst - centroid))])
            con_score = _evaluate(materials, con, wls, ambient_index, substrate_index, materials_dict) \
                if _is_valid(materials, con, problem) else 1e6
            if con_score < scores[-1]:
                simplex[-1] = con.tolist(); scores[-1] = con_score
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n_layers)]
                    scores[i] = _evaluate(materials, simplex[i], wls, ambient_index, substrate_index, materials_dict) \
                        if _is_valid(materials, simplex[i], problem) else 1e6
    bi = int(np.argmin(scores))
    return materials, simplex[bi]

def _structural_sa(materials, thicknesses, problem, wls, materials_dict,
                   ambient_index, substrate_index, deadline, rng, T0=2.0, T1=0.0005, n_steps=15000):
    """Lightweight SA that can change layer count/type for structural improvement."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    nL = materials_dict['L']; nH = materials_dict['H']
    cur_mats = list(materials); cur_thicks = list(thicknesses)
    if not _is_valid(cur_mats, cur_thicks, problem):
        return materials, thicknesses
    cur_score = _evaluate(cur_mats, cur_thicks, wls, ambient_index, substrate_index, materials_dict)
    best_mats, best_thicks, best_score = list(cur_mats), list(cur_thicks), cur_score
    T = T0
    cooling = (T1 / T0) ** (1.0 / max(n_steps, 1))
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        T = max(T * cooling, T1)
        beta = 1.0 / max(T, 1e-10)
        mt = rng.random()
        nm = list(cur_mats); nt = list(cur_thicks)
        if mt < 0.5:
            if len(cur_mats) == 0:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            nt[i] = _clamp_thick(cur_thicks[i] + rng.uniform(-25, 25), max_layer_nm)
        elif mt < 0.7:
            if len(cur_mats) == 0:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            nm[i] = 'H' if cur_mats[i] == 'L' else 'L'
        elif mt < 0.85:
            if len(cur_mats) >= max_layers:
                continue
            pos = rng.randint(0, len(cur_mats))
            m = rng.choice(['L', 'H'])
            n_val = nL if m == 'L' else nH
            d = 600.0 / (4.0 * n_val)
            if sum(cur_thicks) + d > max_total_nm:
                d = max(0.0, max_total_nm - sum(cur_thicks))
            if d < 5:
                continue
            nm.insert(pos, m); nt.insert(pos, d)
        else:
            if len(cur_mats) <= 1:
                continue
            i = rng.randint(0, len(cur_mats) - 1)
            nm.pop(i); nt.pop(i)
        if not _is_valid(nm, nt, problem):
            continue
        ns = _evaluate(nm, nt, wls, ambient_index, substrate_index, materials_dict)
        d = ns - cur_score
        if d < 0 or rng.random() < math.exp(-d * beta):
            cur_mats, cur_thicks, cur_score = nm, nt, ns
            if cur_score < best_score:
                best_score = cur_score
                best_mats, best_thicks = list(cur_mats), list(cur_thicks)
    return best_mats, best_thicks

def design(problem):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    materials_dict = problem["materials"]
    nL = materials_dict["L"]; nH = materials_dict["H"]
    n_sub = problem["substrate_index"]; n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem.get("max_layers", 20)
    wls = np.linspace(band_nm[0], band_nm[1], 401)

    # Build a pool of strong analytical starting points.
    candidates = []
    for n in range(3, 15):
        # recursive admittance (auto L/H choice)
        m, t = _admittance_stack(n, nL, nH, n_sub, n_amb)
        m, t = _make_valid(m, t, problem)
        if m:
            candidates.append((m, t))
        # quarter-wave L-start and H-start
        for start in ('L', 'H'):
            m, t = _quarter_wave_stack(n, nL, nH, n_sub, n_amb, start=start)
            m, t = _make_valid(m, t, problem)
            if m:
                candidates.append((m, t))

    # Evaluate and keep the best few distinct starting points.
    scored = []
    seen = set()
    for m, t in candidates:
        key = tuple(m)
        if key in seen:
            continue
        seen.add(key)
        s = _evaluate(m, t, wls, n_amb, n_sub, materials_dict)
        scored.append((s, m, t))
    scored.sort(key=lambda x: x[0])

    if not scored:
        return ['L'], [600.0 / (4.0 * nL)]

    best_score, best_materials, best_thicknesses = scored[0]

    # Phase A: coordinate-descent polish on the top starting points.
    rng_sa = random.Random(11)
    top = min(5, len(scored))
    for idx in range(top):
        if time.time() > deadline - 40:
            break
        s, m, t = scored[idx]
        pm, pt = _coordinate_descent(list(m), list(t), problem, wls, materials_dict,
                                     n_amb, n_sub, deadline - 35, max_rounds=50)
        if _is_valid(pm, pt, problem):
            ps = _evaluate(pm, pt, wls, n_amb, n_sub, materials_dict)
            if ps < best_score:
                best_score, best_materials, best_thicknesses = ps, list(pm), list(pt)

    # Phase B: structural SA from the current best (allow layer count/type changes).
    if time.time() < deadline - 25:
        sm, st = _structural_sa(list(best_materials), list(best_thicknesses), problem, wls,
                                materials_dict, n_amb, n_sub, deadline - 20, rng_sa,
                                T0=2.0, T1=0.0005, n_steps=12000)
        if _is_valid(sm, st, problem):
            ss = _evaluate(sm, st, wls, n_amb, n_sub, materials_dict)
            if ss < best_score:
                best_score, best_materials, best_thicknesses = ss, list(sm), list(st)

    # Phase C: Nelder-Mead refinement on the best.
    if time.time() < deadline - 12 and len(best_materials) > 0:
        nm, nt = _nelder_mead(list(best_materials), list(best_thicknesses), problem, wls,
                              materials_dict, n_amb, n_sub, deadline - 6, max_iter=1200)
        if _is_valid(nm, nt, problem):
            ns = _evaluate(nm, nt, wls, n_amb, n_sub, materials_dict)
            if ns < best_score:
                best_score, best_materials, best_thicknesses = ns, list(nm), list(nt)

    # Phase D: final coordinate-descent polish.
    if time.time() < deadline - 3 and len(best_materials) > 0:
        try:
            cm, ct = _coordinate_descent(list(best_materials), list(best_thicknesses), problem, wls,
                                         materials_dict, n_amb, n_sub, deadline - 1, max_rounds=40)
            if _is_valid(cm, ct, problem):
                cs = _evaluate(cm, ct, wls, n_amb, n_sub, materials_dict)
                if cs < best_score:
                    best_score, best_materials, best_thicknesses = cs, list(cm), list(ct)
        except Exception:
            pass

    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ['L']
        best_thicknesses = [600.0 / (4.0 * nL)]

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
