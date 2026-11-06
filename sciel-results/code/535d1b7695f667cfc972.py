"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0


def _time_left():
    return _DEADLINE - time.time()


class QAOASimulator:
    def __init__(self, n, edges):
        self.n = n
        self.edges = list(edges)
        self.dim = 1 << n
        self._build_cost_diag()

    def _build_cost_diag(self):
        diag = np.zeros(self.dim, dtype=np.float64)
        for state_idx in range(self.dim):
            cut_val = 0.0
            for u, v in self.edges:
                if ((state_idx >> u) & 1) != ((state_idx >> v) & 1):
                    cut_val += 1.0
            diag[state_idx] = cut_val
        self.H_C_diag = diag

    def expected_cut(self, params):
        n = self.n
        dim = self.dim
        # Initial state: uniform superposition, all amplitudes = 1/sqrt(dim)
        # We can work with amplitudes directly.
        # state[i] = 1/sqrt(dim) initially.
        
        # To save memory and time, we can note that the initial state is uniform.
        # After cost unitary: state[i] = (1/sqrt(dim)) * exp(-i * g1 * H_C_diag[i])
        # After mixer: this is a linear combination.
        
        # Let's just simulate. For n<=10, dim <= 1024.
        # state is a complex array of size dim.
        
        state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
        
        g1, b1, g2, b2 = params
        
        # Layer 1: Cost
        state *= np.exp(-1j * g1 * self.H_C_diag)
        
        # Layer 1: Mixer
        state = self._apply_mixer(state, b1)
        
        # Layer 2: Cost
        state *= np.exp(-1j * g2 * self.H_C_diag)
        
        # Layer 2: Mixer
        state = self._apply_mixer(state, b2)
        
        # Expected cut: sum_i |state[i]|^2 * H_C_diag[i]
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self.H_C_diag))

    def _apply_mixer(self, state, beta):
        # Apply exp(-i beta sum X_i)
        # This is a tensor product of Rx(beta) on each qubit.
        # Rx(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        c = np.cos(beta)
        s = np.sin(beta)
        Rx = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
        
        n = self.n
        # Reshape to tensor
        state_tensor = state.reshape([2] * n)
        
        for axis in range(n):
            # Move axis to front
            state_tensor = np.moveaxis(state_tensor, axis, 0)
            # Contract with Rx
            # state_tensor shape: (2, 2, ..., 2)
            # result[i, rest...] = sum_k Rx[i, k] state_tensor[k, rest...]
            state_tensor = np.tensordot(Rx, state_tensor, axes=([1], [0]))
            # Move axis back
            state_tensor = np.moveaxis(state_tensor, 0, axis)
            
        return state_tensor.reshape(self.dim)


def angles(n, edges, p):
    p = int(p)
    if _time_left() < 5.0:
        return np.tile([0.35, 0.25], (p, 1))

    sim = QAOASimulator(n, edges)
    
    # Grid search for initialization
    # We use a coarse grid first, then refine.
    # Parameters: g1, b1, g2, b2 in [0, pi]
    
    # Coarse grid: 8 points per dimension -> 8^4 = 4096 evaluations
    # Each evaluation is O(n 2^n). For n=10, ~10k ops. 4096 * 10k = 40M ops. ~0.5s.
    # Let's try 10 points per dimension -> 10000 evals. ~1s.
    
    grid_size = 12
    grid = np.linspace(0.0, np.pi, grid_size)
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Iterative refinement: start coarse, then narrow down around best.
    # Or just do a decent grid and then local optimization.
    
    # Let's do a 3-stage grid search:
    # Stage 1: 8^4
    # Stage 2: 10^4 around best +/- pi/8
    # Stage 3: 10^4 around best +/- pi/16
    
    def grid_search(center, radius, size, num_points):
        nonlocal best_val, best_params
        g = np.linspace(center[0] - radius, center[0] + radius, num_points)
        # Ensure within [0, pi]
        g = np.clip(g, 0.0, np.pi)
        
        for g1 in g:
            for b1 in g:
                for g2 in g:
                    for b2 in g:
                        if _time_left() < 5.0:
                            return
                        val = sim.expected_cut([g1, b1, g2, b2])
                        if val > best_val:
                            best_val = val
                            best_params = np.array([g1, b1, g2, b2])

    # Stage 1
    grid_search(np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]), np.pi/2, 8, 8)
    
    # Stage 2
    if _time_left() > 10.0:
        grid_search(best_params, np.pi/8, 10, 10)
        
    # Stage 3
    if _time_left() > 10.0:
        grid_search(best_params, np.pi/16, 10, 10)

    # Local refinement using Nelder-Mead
    if _time_left() > 5.0:
        def neg_cut(params):
            return -sim.expected_cut(params)
            
        try:
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7, 'initial_simplex': 1e-2})
            if res.fun < neg_cut(best_params):
                best_params = res.x
        except Exception:
            pass
        
        # Second local refinement with tighter tolerance
        if _time_left() > 5.0:
            try:
                res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-9, 'initial_simplex': 1e-4})
                if res.fun < neg_cut(best_params):
                    best_params = res.x
            except Exception:
                pass

    best_params = np.asarray(best_params, dtype=float)
    best_params = np.clip(best_params, 0.0, np.pi)

    if p == 2:
        return best_params.reshape(2, 2)
    if p <= 2:
        return best_params[:2 * p].reshape(p, 2)
    return np.tile(best_params.reshape(2, 2), ((p + 1) // 2, 1))[:p]
