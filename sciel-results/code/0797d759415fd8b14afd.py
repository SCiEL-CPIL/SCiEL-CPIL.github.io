"""Rewritten program: random phase signal model for second autocorrelation inequality."""
import numpy as np
import time
import random

N = 50
DELTA = 1.0 / (2 * N)

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) with proper scaling."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    # The continuous supremum of the autoconvolution of piecewise-constant functions
    # on intervals of width DELTA is max(conv) * DELTA
    conv_inf = np.max(conv) * DELTA
    if conv_inf < 1e-15:
        return 1e15
    return l2_sq / (l1 * conv_inf)

def _make_phase_signal(theta, phi, n=N):
    """Generate h_i = max(0, cos(theta*i + phi)) for i=0..n-1."""
    i = np.arange(n)
    return np.maximum(0.0, np.cos(theta * i + phi))

def _grid_search_phase(theta_grid, phi_grid):
    """Grid search over theta and phi to find best ratio."""
    best_ratio = 0.0
    best_h = None
    for theta in theta_grid:
        for phi in phi_grid:
            h = _make_phase_signal(theta, phi)
            r = _compute_ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
    return best_h, best_ratio

def _local_search_h(h, time_budget=5.0):
    """Local search by perturbing individual heights."""
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    start = time.time()
    step = 0.1
    while time.time() - start < time_budget and step > 1e-5:
        improved = False
        for i in range(N):
            if time.time() - start > time_budget:
                break
            for delta in [step, -step]:
                new_h = best_h.copy()
                new_h[i] = max(0.0, best_h[i] + delta)
                r = _compute_ratio(new_h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = new_h
                    improved = True
        if not improved:
            step *= 0.7
    return best_h, best_ratio

def _random_restart_search(time_budget=10.0):
    """Try random initializations and local search."""
    best_h = None
    best_ratio = 0.0
    start = time.time()
    for trial in range(50):
        if time.time() - start > time_budget:
            break
        # Random sparse initialization
        h = np.zeros(N)
        n_spikes = random.randint(3, 12)
        indices = random.sample(range(N), n_spikes)
        for idx in indices:
            h[idx] = random.uniform(0.5, 2.0)
        # Local search
        h, r = _local_search_h(h, time_budget=2.0)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    return best_h, best_ratio

def _sidon_search(time_budget=15.0):
    """Search for good Sidon sets with optimized weights."""
    best_h = None
    best_ratio = 0.0
    start = time.time()
    
    for n_spikes in [8, 7, 6, 5]:
        if time.time() - start > time_budget:
            break
        for trial in range(20):
            if time.time() - start > time_budget:
                break
            random.seed(1000 + trial * 7 + n_spikes)
            # Greedy Sidon set construction
            indices = [random.randint(0, N // 4)]
            for _ in range(n_spikes - 1):
                best_idx = None
                for candidate in range(N - 1, max(indices), -1):
                    new_indices = indices + [candidate]
                    # Check Sidon property
                    diffs = set()
                    valid = True
                    for i in range(len(new_indices)):
                        for j in range(i):
                            d = new_indices[i] - new_indices[j]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        best_idx = candidate
                        break
                if best_idx is None:
                    for candidate in range(max(indices) - 1, -1, -1):
                        if candidate in indices:
                            continue
                        new_indices = sorted(indices + [candidate])
                        diffs = set()
                        valid = True
                        for i in range(len(new_indices)):
                            for j in range(i):
                                d = new_indices[i] - new_indices[j]
                                if d in diffs:
                                    valid = False
                                    break
                                diffs.add(d)
                            if not valid:
                                break
                        if valid:
                            best_idx = candidate
                            break
                if best_idx is None:
                    break
                indices.append(best_idx)
            
            if len(indices) < n_spikes:
                continue
            
            # Optimize weights on this support
            weights = np.ones(len(indices))
            h = np.zeros(N)
            for idx, w in zip(indices, weights):
                h[idx] = w
            
            h, r = _local_search_h(h, time_budget=3.0)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
    
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_h = None
    best_ratio = 0.0
    
    # Strategy 1: Phase signal grid search
    if time.time() < deadline:
        # Coarse grid
        theta_grid = np.linspace(0.01, 3.0, 30)
        phi_grid = np.linspace(0, 2 * np.pi, 20)
        h, r = _grid_search_phase(theta_grid, phi_grid)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
        
        # Refine around best found
        if best_h is not None and time.time() < deadline - 30:
            # Find best theta/phi from grid
            best_theta, best_phi = 0, 0
            for theta in theta_grid:
                for phi in phi_grid:
                    h_test = _make_phase_signal(theta, phi)
                    r_test = _compute_ratio(h_test)
                    if r_test > best_ratio - 1e-6:
                        best_theta, best_phi = theta, phi
                        break
                if best_theta != 0 or best_phi != 0:
                    break
            
            # Fine grid around best
            theta_fine = np.linspace(max(0, best_theta - 0.2), best_theta + 0.2, 20)
            phi_fine = np.linspace(best_phi - 0.3, best_phi + 0.3, 20)
            h, r = _grid_search_phase(theta_fine, phi_fine)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
            
            # Local search refinement
            if best_h is not None and time.time() < deadline - 20:
                h, r = _local_search_h(best_h, time_budget=15.0)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h.copy()
    
    # Strategy 2: Random restart search
    if time.time() < deadline - 10:
        h, r = _random_restart_search(time_budget=10.0)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Strategy 3: Sidon set search
    if time.time() < deadline - 5:
        h, r = _sidon_search(time_budget=10.0)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Fallback: single spike
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_ratio = _compute_ratio(best_h)
    
    return [float(x) for x in best_h]
