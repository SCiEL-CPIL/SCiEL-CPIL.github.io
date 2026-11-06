import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_reflectance(wavelengths, n0, ns, n_layers, t_layers):
    """
    Compute mean reflectance (percent) for given wavelengths and layer stack
    using the characteristic-matrix method.
    """
    if len(n_layers) == 0:
        r = (n0 - ns) / (n0 + ns)
        return r**2 * 100.0

    N = len(n_layers)
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)

    for i in range(N):
        ni = n_layers[i]
        ti = t_layers[i]
        delta = 2.0 * np.pi * ni * ti / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        m11 = cos_d
        m12 = 1j * sin_d / ni
        m21 = 1j * ni * sin_d
        m22 = cos_d

        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22

        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22

    numerator = ns * M11 + 1j * M12
    denominator = ns * M21 + 1j * M22
    Y = numerator / denominator
    r = (Y - n0) / (Y + n0)
    R = np.abs(r)**2
    return np.mean(R) * 100.0


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    wavelengths = np.linspace(band[0], band[1], 401)

    def _objective(t_vec, n_seq):
        if np.any(t_vec < 0) or np.any(t_vec > max_layer_nm):
            return 1000.0
        if np.sum(t_vec) > max_total_nm:
            return 1000.0
        return _compute_reflectance(wavelengths, n0, ns, n_seq, t_vec)

    def _optimize_for_sequence(seq, time_budget=5.0):
        N = len(seq)
        if N == 0:
            return None, float('inf')

        n_seq = [nL if m == "L" else nH for m in seq]
        bounds = [(0.0, max_layer_nm)] * N

        best_val = float('inf')
        best_t = None
        t_start = time.time()

        starts = []
        for lam0 in [480.0, 520.0, 560.0, 600.0, 640.0, 680.0, 720.0]:
            t_qw = np.array([lam0 / (4.0 * n) for n in n_seq])
            if np.all(t_qw <= max_layer_nm) and np.sum(t_qw) <= max_total_nm:
                starts.append(t_qw)

        np.random.seed(42)
        for _ in range(6):
            t_rand = np.random.uniform(0.1, max_layer_nm, N)
            if np.sum(t_rand) > max_total_nm:
                t_rand = t_rand * (max_total_nm / np.sum(t_rand))
            starts.append(t_rand)

        # Global search with DE
        if time.time() < t_start + time_budget * 0.35:
            try:
                result_de = differential_evolution(
                    lambda t: _objective(t, n_seq),
                    bounds,
                    maxiter=20,
                    popsize=8,
                    seed=42,
                    tol=0.05,
                    init='latinhypercube'
                )
                if result_de.fun < best_val:
                    best_val = result_de.fun
                    best_t = result_de.x
            except Exception:
                pass

        # Local refinement
        for t0 in starts:
            if time.time() > t_start + time_budget * 0.85:
                break
            try:
                res = minimize(
                    lambda t: _objective(t, n_seq),
                    t0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-9}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_t = res.x
            except Exception:
                pass

        # Polish
        if best_t is not None and time.time() < t_start + time_budget:
            try:
                res = minimize(
                    lambda t: _objective(t, n_seq),
                    best_t,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-9}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_t = res.x
            except Exception:
                pass

        if best_t is not None:
            best_t = np.clip(best_t, 0.0, max_layer_nm)
            if np.sum(best_t) > max_total_nm:
                best_t = best_t * (max_total_nm / np.sum(best_t))

        return best_t, best_val

    sequences = []

    # Alternating patterns
    for N in range(2, min(16, max_layers + 1)):
        seq_l = ["L" if i % 2 == 0 else "H" for i in range(N)]
        sequences.append(seq_l)
        seq_h = ["H" if i % 2 == 0 else "L" for i in range(N)]
        sequences.append(seq_h)

    # Symmetric patterns
    for N_half in range(2, min(8, max_layers // 2 + 1)):
        half = ["L" if i % 2 == 0 else "H" for i in range(N_half)]
        seq_sym = half + half[::-1]
        if len(seq_sym) <= max_layers:
            sequences.append(seq_sym)
        # Also try starting with H
        half_h = ["H" if i % 2 == 0 else "L" for i in range(N_half)]
        seq_sym_h = half_h + half_h[::-1]
        if len(seq_sym_h) <= max_layers:
            sequences.append(seq_sym_h)

    # Defect patterns
    import random
    random.seed(123)
    for N in range(4, min(12, max_layers + 1)):
        for _ in range(4):
            seq = ["L" if i % 2 == 0 else "H" for i in range(N)]
            for _d in range(random.randint(1, 2)):
                pos = random.randint(1, N - 2)
                seq[pos] = "H" if seq[pos] == "L" else "L"
            sequences.append(seq)

    # Deduplicate
    unique_seqs = []
    seen = set()
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_seqs.append(seq)

    unique_seqs.sort(key=lambda s: len(s))

    best_score = float('inf')
    best_materials = []
    best_thicknesses = []

    # Phase 1: Quick evaluation
    for seq in unique_seqs:
        if time.time() > deadline - 20.0:
            break
        t_opt, val_opt = _optimize_for_sequence(seq, time_budget=2.5)
        if t_opt is not None and val_opt < best_score:
            best_score = val_opt
            best_materials = list(seq)
            best_thicknesses = [float(t) for t in t_opt]

    # Phase 2: Refine best candidates
    if time.time() < deadline - 15.0:
        # Re-evaluate top sequences with more time
        candidates = []
        for seq in unique_seqs:
            if time.time() > deadline - 15.0:
                break
            t_opt, val_opt = _optimize_for_sequence(seq, time_budget=4.0)
            if t_opt is not None:
                candidates.append((val_opt, seq, t_opt))

        candidates.sort(key=lambda x: x[0])
        for val, seq, t in candidates[:6]:
            if time.time() > deadline - 5.0:
                break
            t_opt, val_opt = _optimize_for_sequence(seq, time_budget=8.0)
            if t_opt is not None and val_opt < best_score:
                best_score = val_opt
                best_materials = list(seq)
                best_thicknesses = [float(t) for t in t_opt]

    if not best_materials:
        lam0 = 600.0
        t0 = lam0 / (4.0 * nL)
        if t0 <= max_layer_nm and t0 <= max_total_nm:
            best_materials = ["L"]
            best_thicknesses = [t0]
        else:
            best_materials = ["L"]
            best_thicknesses = [1.0]

    return best_materials, best_thicknesses
