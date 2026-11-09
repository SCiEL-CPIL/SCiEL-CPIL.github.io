"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Differential evolution (global) over a parameterized gradient stack of alternating
     L/H layers, where thicknesses follow a linear ramp from air-side to substrate-side.
  3. Nelder-Mead local polish on the best DE candidates.
  4. Additional exploration with random restarts on diverse layer counts.
  5. Explicit deadline guard: return the best feasible result found so far.
"""

import time
import math
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _gradient_thicknesses(num_layers, t0, t1, max_total, max_layer):
    """Generate a linear-gradient stack of alternating L/H layers.
    
    num_layers: total number of layers
    t0: first layer thickness
    t1: last layer thickness
    Returns (materials, thicknesses) with alternating L starting from air side.
    """
    materials = []
    thicknesses = []
    if num_layers == 1:
        materials = ['L']
        thicknesses = [t0]
    else:
        for i in range(num_layers):
            frac = i / (num_layers - 1)
            t = t0 + (t1 - t0) * frac
            t = max(0.0, min(max_layer, t))
            materials.append('L' if i % 2 == 0 else 'H')
            thicknesses.append(t)
    
    # Enforce total thickness constraint
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
        # Re-clamp
        for i in range(len(thicknesses)):
            thicknesses[i] = max(0.0, min(max_layer, thicknesses[i]))
        total2 = sum(thicknesses)
        if total2 > max_total:
            scale = max_total / total2
            thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses


def _nelder_mead(materials, init_ths, problem, deadline, max_layer, max_total, max_iters=300):
    """Nelder-Mead local polish."""
    ths = list(init_ths)
    n = len(ths)
    
    def constrain(ths):
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        for i in range(n):
            ths[i] = max(0.0, min(max_layer, ths[i]))
        total2 = sum(ths)
        if total2 > max_total:
            scale = max_total / total2
            ths = [t * scale for t in ths]
        return ths
    
    ths = constrain(ths)
    best_score = _mean_reflectance(materials, ths, problem)
    best_ths = list(ths)
    
    step = 15.0
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    simplex = [list(ths)]
    for i in range(n):
        pt = list(ths)
        pt[i] = max(0.0, min(max_layer, pt[i] + step))
        simplex.append(pt)
    
    scores = [_mean_reflectance(materials, simplex[i], problem) for i in range(n + 1)]
    
    it = 0
    while it < max_iters and time.time() < deadline:
        it += 1
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_ths = list(simplex[0])
        
        centroid = np.mean(simplex[:-1], axis=0)
        xr = constrain(list(centroid + alpha * (centroid - simplex[-1])))
        fr = _mean_reflectance(materials, xr, problem)
        
        if fr < scores[0]:
            xe = constrain(list(centroid + gamma * (centroid - simplex[-1])))
            fe = _mean_reflectance(materials, xe, problem)
            if fe < fr:
                simplex[-1], scores[-1] = xe, fe
            else:
                simplex[-1], scores[-1] = xr, fr
        elif fr < scores[-2]:
            simplex[-1], scores[-1] = xr, fr
        else:
            if fr < scores[-1]:
                xc = constrain(list(centroid + rho * (simplex[-1] - centroid)))
                fc = _mean_reflectance(materials, xc, problem)
                if fc <= fr:
                    simplex[-1], scores[-1] = xc, fc
                else:
                    break
            else:
                for i in range(1, n + 1):
                    simplex[i] = constrain([simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)])
                    scores[i] = _mean_reflectance(materials, simplex[i], problem)
                if np.max(scores) - np.min(scores) < 1e-7:
                    break
    
    return best_ths, best_score


def design(problem):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    best_result = None
    best_score = float('inf')

    # --- Phase 1: Differential Evolution on gradient-parameterized stacks ---
    # Parameters: (num_layers, t0, t1) where t0 is air-side thickness, t1 is substrate-side
    # We'll run DE over a fixed set of layer counts, optimizing (t0, t1) for each.
    
    random.seed(123)
    np.random.seed(123)
    
    # Layer counts to explore
    layer_counts = list(range(2, min(max_layers + 1, 16)))
    
    # For each layer count, run a small DE over (t0, t1)
    de_pop_size = 12
    de_max_gen = 40
    de_f = 0.8
    de_cr = 0.9
    
    for n_layers in layer_counts:
        if time.time() >= deadline - 40:
            break
        
        # DE over 2D parameter space: (t0, t1)
        # t0, t1 in [5, 350] (reasonable range for quarter-wave ~ 90-145 nm each)
        lo = np.array([5.0, 5.0])
        hi = np.array([350.0, 350.0])
        
        # Initialize population
        pop = np.random.uniform(lo, hi, size=(de_pop_size, 2))
        # Ensure total thickness constraint
        for i in range(de_pop_size):
            mats, ths = _gradient_thicknesses(n_layers, pop[i, 0], pop[i, 1], max_total, max_layer)
            pop[i, 0] = ths[0]
            pop[i, 1] = ths[-1]
        
        # Evaluate initial population
        pop_scores = []
        pop_mats_ths = []
        for i in range(de_pop_size):
            mats, ths = _gradient_thicknesses(n_layers, pop[i, 0], pop[i, 1], max_total, max_layer)
            score = _mean_reflectance(mats, ths, problem)
            pop_scores.append(score)
            pop_mats_ths.append((mats, ths))
        
        best_idx = int(np.argmin(pop_scores))
        best_local_score = pop_scores[best_idx]
        best_local_mats = pop_mats_ths[best_idx][0]
        best_local_ths = pop_mats_ths[best_idx][1]
        
        # DE iterations
        for gen in range(de_max_gen):
            if time.time() >= deadline - 40:
                break
            
            new_pop = [list(pop[best_idx])]  # keep best
            new_scores = [pop_scores[best_idx]]
            
            for i in range(de_pop_size - 1):
                # Select three distinct random indices
                a, b, c = random.sample(range(de_pop_size), 3)
                mutant = pop[a] + de_f * (pop[b] - pop[c])
                mutant = np.clip(mutant, lo, hi)
                
                # Crossover
                trial = list(pop[i])
                jrand = random.randint(0, 1)
                for j in range(2):
                    if random.random() < de_cr or j == jrand:
                        trial[j] = mutant[j]
                
                # Evaluate trial
                mats, ths = _gradient_thicknesses(n_layers, trial[0], trial[1], max_total, max_layer)
                score = _mean_reflectance(mats, ths, problem)
                
                if score <= pop_scores[i]:
                    new_pop.append(trial)
                    new_scores.append(score)
                else:
                    new_pop.append(list(pop[i]))
                    new_scores.append(pop_scores[i])
            
            pop = np.array(new_pop)
            pop_scores = new_scores
            best_idx = int(np.argmin(pop_scores))
            if pop_scores[best_idx] < best_local_score:
                best_local_score = pop_scores[best_idx]
                mats, ths = _gradient_thicknesses(n_layers, pop[best_idx, 0], pop[best_idx, 1], max_total, max_layer)
                best_local_mats = mats
                best_local_ths = ths
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_result = (best_local_mats, best_local_ths)

    # --- Phase 2: Nelder-Mead polish on best result ---
    if best_result is not None and time.time() < deadline - 15:
        mats, ths = best_result
        polished_ths, polished_score = _nelder_mead(
            mats, ths, problem, deadline - 10, max_layer, max_total, max_iters=200
        )
        if polished_score < best_score:
            best_score = polished_score
            best_result = (mats, polished_ths)

    # --- Phase 3: Additional random restarts on diverse structures ---
    if time.time() < deadline - 20:
        # Try a few more random patterns with random initial thicknesses
        for attempt in range(30):
            if time.time() >= deadline - 15:
                break
            n_layers = random.randint(3, min(max_layers, 14))
            materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
            # Random initial thicknesses around quarter-wave
            qw_L = wl_center / (4.0 * 1.38)
            qw_H = wl_center / (4.0 * 2.35)
            init_ths = []
            for m in materials:
                qw = qw_L if m == 'L' else qw_H
                t = qw * random.uniform(0.5, 1.5)
                init_ths.append(max(0.0, min(max_layer, t)))
            
            # Quick local polish
            if time.time() < deadline - 15:
                polished_ths, polished_score = _nelder_mead(
                    materials, init_ths, problem, deadline - 15, max_layer, max_total, max_iters=150
                )
                if polished_score < best_score:
                    best_score = polished_score
                    best_result = (materials, polished_ths)

    # --- Phase 4: Final polish if time remains ---
    if best_result is not None and time.time() < deadline - 5:
        mats, ths = best_result
        polished_ths, polished_score = _nelder_mead(
            mats, ths, problem, deadline - 2, max_layer, max_total, max_iters=100
        )
        if polished_score < best_score:
            best_score = polished_score
            best_result = (mats, polished_ths)

    if best_result is None:
        # Fallback: single quarter-wave L layer
        qw_L = wl_center / (4.0 * 1.38)
        return ['L'], [qw_L]

    mats, ths = best_result
    return list(mats), list(ths)
