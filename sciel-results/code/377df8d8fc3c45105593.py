import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    # Deadline: 140 seconds to be safe (evaluator kills at 150s)
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Precompute cost vector
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    
    indices = np.arange(num_states)
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1

    # 2. Efficient state evolution using precomputed mixer matrices
    def apply_rx_all(state, c, s):
        """Apply R_x(c,s) to all n qubits efficiently."""
        shape = [2] * n
        tensor = state.reshape(shape)
        for q in range(n):
            tensor = np.moveaxis(tensor, q, 0)
            orig_shape = tensor.shape
            tensor = tensor.reshape(2, -1)
            v0 = tensor[0, :]
            v1 = tensor[1, :]
            new_v0 = c * v0 - 1j * s * v1
            new_v1 = -1j * s * v0 + c * v1
            tensor = np.vstack((new_v0, new_v1))
            tensor = tensor.reshape(orig_shape)
            tensor = np.moveaxis(tensor, 0, q)
        return tensor.reshape(num_states)

    def compute_expected_cut(g1, b1, g2, b2):
        """Compute expected cut for given parameters."""
        c1 = np.cos(b1 / 2.0)
        s1 = np.sin(b1 / 2.0)
        c2 = np.cos(b2 / 2.0)
        s2 = np.sin(b2 / 2.0)
        
        # State initialization: |+>^n
        state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Layer 1: cost phase
        state = state * np.exp(-1j * g1 * cost_vec)
        # Layer 1: mixer
        state = apply_rx_all(state, c1, s1)
        # Layer 2: cost phase
        state = state * np.exp(-1j * g2 * cost_vec)
        # Layer 2: mixer
        state = apply_rx_all(state, c2, s2)
        
        probs = np.abs(state) ** 2
        return np.dot(probs, cost_vec)

    def objective(params):
        g1, b1, g2, b2 = params
        return -compute_expected_cut(g1, b1, g2, b2)

    # 3. Graph-scaled parameter ranges
    m = len(edges)
    # Scale ranges based on graph density
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # Graph-scaled ranges: wider for denser graphs
    g_range = (0.05, 0.8 + 0.3 * density)
    b_range = (0.05, 0.6 + 0.2 * density)

    # 4. Global grid search over coarsely spaced parameter space
    # Use adaptive grid resolution based on time budget
    grid_size = 8  # 8x8x8x8 = 4096 points initially
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Create grid points
    g_vals = np.linspace(g_range[0], g_range[1], grid_size)
    b_vals = np.linspace(b_range[0], b_range[1], grid_size)
    
    # Evaluate grid points
    for g1 in g_vals:
        if time.time() > deadline - 30.0:
            break
        for b1 in b_vals:
            if time.time() > deadline - 30.0:
                break
            for g2 in g_vals:
                if time.time() > deadline - 30.0:
                    break
                for b2 in b_vals:
                    if time.time() > deadline - 30.0:
                        break
                    try:
                        val = -compute_expected_cut(g1, b1, g2, b2)
                        if val < best_val:
                            best_val = val
                            best_params = np.array([g1, b1, g2, b2])
                    except Exception:
                        pass
    
    # 5. Multi-start Nelder-Mead refinement from best grid point and neighbors
    # Collect top candidates from grid for refinement
    candidates = [best_params.copy()]
    
    # Add neighbors of best point
    step_g = (g_range[1] - g_range[0]) / grid_size
    step_b = (b_range[0] - b_range[0]) / grid_size
    for dg in [-step_g, 0, step_g]:
        for db in [-step_b, 0, step_b]:
            for dg2 in [-step_g, 0, step_g]:
                for db2 in [-step_b, 0, step_b]:
                    cand = best_params + np.array([dg, db, dg2, db2])
                    if np.all(cand >= 0) and np.all(cand <= np.array([g_range[1], b_range[1], g_range[1], b_range[1]])):
                        candidates.append(cand)
    
    # Add spectral/degree-based initialization
    # Compute degree sequence
    degrees = np.zeros(n)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    max_deg = np.max(degrees) if n > 0 else 1
    
    # Degree-based initial guesses
    spec_guesses = [
        np.array([0.3 * (max_deg / max(1, m/n)), 0.2 * (max_deg / max(1, m/n)),
                  0.3 * (max_deg / max(1, m/n)), 0.2 * (max_deg / max(1, m/n))]),
        np.array([0.4, 0.3, 0.4, 0.3]),
        np.array([0.25, 0.35, 0.25, 0.35]),
    ]
    candidates.extend(spec_guesses)
    
    # Add random points for diversity
    rng = np.random.default_rng(42)
    for _ in range(10):
        candidates.append(rng.uniform([g_range[0], b_range[0], g_range[0], b_range[0]],
                                      [g_range[1], b_range[1], g_range[1], b_range[1]]))
    
    # 6. Refine each candidate with Nelder-Mead
    for x0 in candidates:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True, 'maxfev': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # 7. Final high-precision polish from the best found
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True, 'maxfev': 2000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Return in required shape (p, 2)
    return best_params.reshape(p, 2)
