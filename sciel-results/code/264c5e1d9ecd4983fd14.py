"""Broadband antireflection coating design via characteristic-matrix model + timed search."""
import math
import random
import time

import numpy as np

# Fixed task parameters (from the contract)
N_AMB = 1.0
N_SUB = 1.52
N_L = 1.38
N_H = 2.35
WL0 = 400.0
WL1 = 800.0
N_WL = 401
MAX_LAYERS = 20
MAX_TOTAL_NM = 1000.0
MAX_LAYER_NM = 400.0

# Evaluation wavelengths (401 equally spaced in [400,800])
_WL = np.linspace(WL0, WL1, N_WL)
_K = 2.0 * math.pi / _WL          # 1/nm
# Precompute per-material complex refractive index arrays
_NL = np.full(N_WL, N_L)
_NH = np.full(N_WL, N_H)
_NS = np.full(N_WL, N_SUB)
_NA = np.full(N_WL, N_AMB)

# Deadline guard (wall-clock)
_DEADLINE = None


def _reflectance(labels, thicknesses):
    """Mean normal-incidence reflectance (%) over the band, characteristic-matrix method."""
    n = len(labels)
    # Build stack matrices M_i = [[cos δ, i sin δ / n], [i n sin δ, cos δ]]
    M = np.eye(2, dtype=complex)
    for i in range(n):
        ni = _NL if labels[i] == 'L' else _NH
        delta = _K * ni * thicknesses[i]
        c = np.cos(delta)
        s = np.sin(delta)
        Mi = np.array([[c, 1j * s / ni], [1j * ni * s, c]], dtype=complex)
        M = M @ Mi
    # Characteristic matrix of the stack
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]
    # Admittance seen from ambient: Y = (m11 + m12 * n_sub) / (m21 + m22 * n_sub)
    Y = (m11 + m12 * _NS) / (m21 + m22 * _NS)
    # Reflectance amplitude at air/stack interface
    r = (Y - _NA) / (Y + _NA)
    R = (r * np.conj(r)).real
    return 100.0 * np.mean(R)


def _valid(labels, thicknesses):
    if len(labels) != len(thicknesses):
        return False
    if len(labels) > MAX_LAYERS:
        return False
    total = 0.0
    for t in thicknesses:
        if t < -1e-6 or t > MAX_LAYER_NM + 1e-6:
            return False
        total += t
    if total > MAX_TOTAL_NM + 1e-6:
        return False
    return True


def _clamp_thicknesses(thicknesses):
    return [min(max(t, 0.0), MAX_LAYER_NM) for t in thicknesses]


def _random_stack(rng, n_layers):
    labels = ['L' if rng.random() < 0.5 else 'H' for _ in range(n_layers)]
    # Random thicknesses, scaled so total stays within budget
    raw = [rng.uniform(10.0, MAX_LAYER_NM) for _ in range(n_layers)]
    total = sum(raw)
    if total > MAX_TOTAL_NM:
        scale = MAX_TOTAL_NM / total
        raw = [t * scale for t in raw]
    return labels, raw


def design(problem):
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 140.0  # safety margin under 150s wall-clock

    rng = random.Random(12345)

    # --- Build a good initial population ---
    # Best-known AR designs: quarter-wave stacks. Start with several candidates.
    candidates = []  # list of (score, labels, thicknesses)

    # 1) Single quarter-wave L at band center
    c = 0.5 * (WL0 + WL1)
    labels = ['L']
    th = [c / (4.0 * N_L)]
    candidates.append((_reflectance(labels, th), labels, th))

    # 2) Quarter-wave H at band center
    labels = ['H']
    th = [c / (4.0 * N_H)]
    candidates.append((_reflectance(labels, th), labels, th))

    # 3) L-H quarter-wave pair (classic broadband attempt)
    labels = ['L', 'H']
    th = [c / (4.0 * N_L), c / (4.0 * N_H)]
    candidates.append((_reflectance(labels, th), labels, th))

    # 4) H-L quarter-wave pair
    labels = ['H', 'L']
    th = [c / (4.0 * N_H), c / (4.0 * N_L)]
    candidates.append((_reflectance(labels, th), labels, th))

    # 5) Multi-layer quarter-wave stacks with varying counts
    for n in range(3, 9):
        labels = []
        th = []
        for i in range(n):
            if i % 2 == 0:
                labels.append('L')
                th.append(c / (4.0 * N_L))
            else:
                labels.append('H')
                th.append(c / (4.0 * N_H))
        if sum(th) <= MAX_TOTAL_NM:
            candidates.append((_reflectance(labels, th), labels, th))

    # 6) Random initial candidates
    for _ in range(30):
        n = rng.randint(1, 12)
        labels, th = _random_stack(rng, n)
        if _valid(labels, th):
            candidates.append((_reflectance(labels, th), labels, th))

    # Sort and keep best
    candidates.sort(key=lambda x: x[0])
    best_score, best_labels, best_th = candidates[0]

    # --- Local improvement: gradient-free coordinate descent on thicknesses ---
    def _improve(labels, thicknesses, max_iter=200):
        nonlocal best_score, best_labels, best_th
        cur_score = _reflectance(labels, thicknesses)
        cur_th = list(thicknesses)
        n = len(labels)
        step = 20.0
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < _DEADLINE:
            improved = False
            it += 1
            for i in range(n):
                for d in [step, -step]:
                    new_th = cur_th[:]
                    new_th[i] = max(0.0, min(MAX_LAYER_NM, new_th[i] + d))
                    if sum(new_th) > MAX_TOTAL_NM:
                        continue
                    s = _reflectance(labels, new_th)
                    if s < cur_score:
                        cur_score = s
                        cur_th = new_th
                        improved = True
                        if s < best_score:
                            best_score = s
                            best_labels = labels[:]
                            best_th = cur_th[:]
                # try small perturbations
                for d in [step * 0.3, -step * 0.3, step * 0.1, -step * 0.1]:
                    new_th = cur_th[:]
                    new_th[i] = max(0.0, min(MAX_LAYER_NM, new_th[i] + d))
                    if sum(new_th) > MAX_TOTAL_NM:
                        continue
                    s = _reflectance(labels, new_th)
                    if s < cur_score:
                        cur_score = s
                        cur_th = new_th
                        improved = True
                        if s < best_score:
                            best_score = s
                            best_labels = labels[:]
                            best_th = cur_th[:]
            step *= 0.7
        return cur_score, cur_th

    # Improve top few candidates
    for score, labels, th in candidates[:5]:
        if time.time() >= _DEADLINE:
            break
        s, th2 = _improve(labels, th, max_iter=150)

    # --- Simulated-annealing-like global search ---
    if time.time() < _DEADLINE:
        # Start from best
        cur_labels = best_labels[:]
        cur_th = best_th[:]
        cur_score = best_score
        T = 0.5  # initial "temperature" in percent-reflectance units
        T_min = 0.001
        cooling = 0.985

        while time.time() < _DEADLINE and T > T_min:
            # Perturbation: change a random layer's thickness, or add/remove a layer
            op = rng.random()
            if op < 0.5:
                # Perturb thickness of a random layer
                i = rng.randint(0, len(cur_th) - 1)
                delta = rng.gauss(0, 30)
                new_th = cur_th[:]
                new_th[i] = max(0.0, min(MAX_LAYER_NM, new_th[i] + delta))
                if sum(new_th) <= MAX_TOTAL_NM:
                    s = _reflectance(cur_labels, new_th)
                    if s < cur_score or rng.random() < math.exp(-(s - cur_score) / T):
                        cur_th = new_th
                        cur_score = s
                        if s < best_score:
                            best_score = s
                            best_labels = cur_labels[:]
                            best_th = cur_th[:]
            elif op < 0.7 and len(cur_th) < MAX_LAYERS:
                # Add a layer
                pos = rng.randint(0, len(cur_th))
                new_labels = cur_labels[:]
                new_labels.insert(pos, 'L' if rng.random() < 0.5 else 'H')
                new_th = cur_th[:]
                new_th.insert(pos, rng.uniform(20.0, 100.0))
                if sum(new_th) <= MAX_TOTAL_NM:
                    s = _reflectance(new_labels, new_th)
                    if s < cur_score or rng.random() < math.exp(-(s - cur_score) / T):
                        cur_labels = new_labels
                        cur_th = new_th
                        cur_score = s
                        if s < best_score:
                            best_score = s
                            best_labels = cur_labels[:]
                            best_th = cur_th[:]
            elif op < 0.85 and len(cur_th) > 1:
                # Remove a layer
                pos = rng.randint(0, len(cur_th) - 1)
                new_labels = cur_labels[:]
                del new_labels[pos]
                new_th = cur_th[:]
                del new_th[pos]
                s = _reflectance(new_labels, new_th)
                if s < cur_score or rng.random() < math.exp(-(s - cur_score) / T):
                    cur_labels = new_labels
                    cur_th = new_th
                    cur_score = s
                    if s < best_score:
                        best_score = s
                        best_labels = cur_labels[:]
                        best_th = cur_th[:]
            else:
                # Swap two layer labels
                if len(cur_th) >= 2:
                    i, j = rng.sample(range(len(cur_th)), 2)
                    new_labels = cur_labels[:]
                    new_labels[i], new_labels[j] = new_labels[j], new_labels[i]
                    s = _reflectance(new_labels, cur_th)
                    if s < cur_score or rng.random() < math.exp(-(s - cur_score) / T):
                        cur_labels = new_labels
                        cur_score = s
                        if s < best_score:
                            best_score = s
                            best_labels = cur_labels[:]
                            best_th = cur_th[:]
            T *= cooling

    # Final polish on best
    if time.time() < _DEADLINE:
        s, th2 = _improve(best_labels, best_th, max_iter=300)

    # Ensure validity
    best_th = _clamp_thicknesses(best_th)
    if not _valid(best_labels, best_th):
        # Fallback to a known-good simple design
        best_labels = ['L']
        best_th = [600.0 / (4.0 * N_L)]

    return best_labels, best_th
