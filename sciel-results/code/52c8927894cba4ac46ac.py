"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class _CircuitSim:
    """Precomputed circuit simulator for fast state evolution."""
    def __init__(self, n):
        self.n = n
        self.dim = 2 ** n
        self.indices = np.arange(self.dim)
        # Precompute bit masks for each qubit
        self.bit_masks = [(1 << (n - 1 - i)) for i in range(n)]
        # Precompute which indices have bit i set to 0 or 1
        self.zero_indices = {}
        self.one_indices = {}
        for i in range(n):
            mask = self.bit_masks[i]
            self.zero_indices[i] = self.indices[(self.indices & mask) == 0]
            self.one_indices[i] = self.indices[(self.indices & mask) != 0]
        # Precompute CZ sign flips for adjacent pairs
        self.cz_flip_indices = {}
        for i in range(n - 1):
            mask_a = self.bit_masks[i]
            mask_b = self.bit_masks[i + 1]
            both_set = self.indices[(self.indices & mask_a) != 0]
            both_set = both_set[(both_set & mask_b) != 0]
            self.cz_flip_indices[i] = both_set

    def _apply_ry(self, state, qubit, theta):
        """Apply Ry(theta) to qubit using vectorized operations."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        mask = self.bit_masks[qubit]
        zero_idx = self.zero_indices[qubit]
        one_idx = self.one_indices[qubit]
        
        new_state = state.copy()
        # For indices where qubit is 0: new[idx] = c*state[idx], new[idx|mask] += s*state[idx]
        # For indices where qubit is 1: new[idx] = c*state[idx], new[idx&~mask] += -s*state[idx]
        
        # Process zero-bit indices
        vals_zero = state[zero_idx]
        new_state[zero_idx] = c * vals_zero
        idx_one = zero_idx | mask
        new_state[idx_one] += s * vals_zero
        
        # Process one-bit indices
        vals_one = state[one_idx]
        new_state[one_idx] = c * vals_one
        idx_zero = one_idx & ~mask
        new_state[idx_zero] += -s * vals_one
        
        return new_state

    def _apply_cz(self, state, pair_idx):
        """Apply CZ to adjacent pair using precomputed flip indices."""
        new_state = state.copy()
        flip_idx = self.cz_flip_indices[pair_idx]
        new_state[flip_idx] = -state[flip_idx]
        return new_state

    def evolve(self, theta, layers):
        """Evolve the state through the circuit."""
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        
        for l in range(layers):
            for i in range(self.n):
                state = self._apply_ry(state, i, theta[l, i])
            if l < layers - 1:
                for i in range(self.n - 1):
                    state = self._apply_cz(state, i)
        
        return state


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i or j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op
    
    # X terms: -field * X_i
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _circuit_energy(sim, H, theta, layers):
    """Compute the energy of the circuit state."""
    state = sim.evolve(theta, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _make_objective(sim, H, n, ground_energy, layers):
    """Create a closure for the objective function."""
    def objective(params):
        theta = params.reshape((layers, n))
        energy = _circuit_energy(sim, H, theta, layers)
        return (energy - ground_energy) / n
    return objective


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Create circuit simulator with precomputed indices
    sim = _CircuitSim(n)
    objective = _make_objective(sim, H, n, ground_energy, layers)
    
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Try multiple initializations with different strategies
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Last layer pi/2
    init3 = np.zeros(layers * n)
    init3[-n:] = np.pi / 2
    initializations.append(init3)
    
    # 4. Physics-based interpolation
    init4 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init4[l * n + i] = np.pi / 2 * min(1.0, field)
    initializations.append(init4)
    
    # 5-8. Random initializations with different seeds
    for seed in [42, 123, 7, 99]:
        rng = np.random.RandomState(seed)
        initializations.append(rng.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 9. Symmetric initialization based on field strength
    init9 = np.zeros(layers * n)
    base_angle = np.arctan(field / coupling) if coupling > 0 else np.pi/2
    for l in range(layers):
        scale = 1.0 - 0.3 * l
        for i in range(n):
            init9[l * n + i] = base_angle * scale
    initializations.append(init9)
    
    # 10. Another physics-inspired init
    init10 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            pos_factor = 0.8 + 0.2 * np.sin(np.pi * i / max(1, n-1))
            init10[l * n + i] = (np.pi / 2) * min(1.0, field) * pos_factor * (1.0 - 0.2*l)
    initializations.append(init10)
    
    # Run Nelder-Mead from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 1500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Refine with L-BFGS-B from the best result
    if time.time() < deadline - 15:
        try:
            result2 = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 500,
                    'maxfun': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-10
                }
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Try local refinements from slightly perturbed best params
    if time.time() < deadline - 5:
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        for pert in [0.01, 0.05, 0.1]:
            if time.time() > deadline - 3:
                break
            try:
                perturbed = best_params + rng.uniform(-pert, pert, layers * n)
                result3 = minimize(
                    objective,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * (layers * n),
                    options={
                        'maxiter': 300,
                        'maxfun': 1000,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                
                if result3.fun < best_score:
                    best_score = result3.fun
                    best_params = result3.x.copy()
            except Exception:
                continue
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result4 = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 200,
                    'maxfun': 500,
                    'ftol': 1e-16,
                    'gtol': 1e-12
                }
            )
            
            if result4.fun < best_score:
                best_score = result4.fun
                best_params = result4.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
