"""Optimize 3-layer Ry/CZ circuit parameters for transverse-field Ising chains."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

# Pre-compute for speed
# n=4,6; layers=3; coupling=1; fields=0.45,1,1.65
# Dev fields: 0.3, 0.8, 1.4

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for open chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    # Z_i = diag(1, -1) for qubit i
    # For open chain, pairs (0,1), (1,2), ..., (n-2, n-1)
    # Bit i corresponds to qubit i (little-endian? "qubit i is bit i")
    # Let's use standard: bit 0 is least significant, so qubit 0 is LSB.
    # Z_i Z_j acting on basis state |x> gives eigenvalue (+1 if x_i==x_j else -1)
    # Actually Z|0>=|0>, Z|1>=-|1>. So Z_i Z_j |x> = (-1)^{x_i + x_j} |x>
    
    for x in range(dim):
        zz_val = 0.0
        for i in range(n - 1):
            xi = (x >> i) & 1
            xj = (x >> (i + 1)) & 1
            if xi == xj:
                zz_val += 1.0
            else:
                zz_val -= 1.0
        H[x, x] -= coupling * zz_val
    
    # X terms: -field * sum X_i
    # X flips bit i
    for i in range(n):
        for x in range(dim):
            xp = x ^ (1 << i)
            H[xp, x] -= field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground energy using eigendecomposition."""
    H = _build_hamiltonian(n, coupling, field)
    # For small n (4,6), eigendecomposition is feasible
    # Use eigh for Hermitian matrices
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _apply_circuit_state(theta, n, layers):
    """
    Apply the circuit to |0>^n and return the state vector.
    theta: (layers, n) array of angles
    Circuit:
    - Start |0>^n
    - For each layer l:
        - Apply Ry(theta[l, i]) to every qubit i
        - If not final layer, apply CZ on every adjacent pair in ascending order
    """
    dim = 1 << n
    # Start state |0>^n = |0...0>
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Pre-compute rotation matrices
    # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    
    for l in range(layers):
        # Apply Ry to every qubit
        for i in range(n):
            theta_i = theta[l, i]
            c = np.cos(theta_i / 2)
            s = np.sin(theta_i / 2)
            
            # Apply Ry on qubit i
            # For each basis state, if bit i is 0, multiply by c and add s * |bit i = 1>
            # We can do this by iterating over states
            new_state = np.zeros_like(state)
            for x in range(dim):
                if state[x] == 0:
                    continue
                bit_i = (x >> i) & 1
                if bit_i == 0:
                    # |0> -> c|0> + s|1>
                    new_state[x] += c * state[x]
                    xp = x | (1 << i)
                    new_state[xp] += s * state[x]
                else:
                    # |1> -> -s|0> + c|1>
                    new_state[x] += c * state[x]
                    xm = x & ~(1 << i)
                    new_state[xm] -= s * state[x]
            state = new_state
        
        # Apply CZ on adjacent pairs if not final layer
        if l < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1
                # CZ|00> = |00>, CZ|01> = |01>, CZ|10> = |10>, CZ|11> = -|11>
                # So we flip the sign of states where both bit i and bit i+1 are 1
                for x in range(dim):
                    if state[x] != 0:
                        bi = (x >> i) & 1
                        bj = (x >> (i + 1)) & 1
                        if bi == 1 and bj == 1:
                            state[x] = -state[x]
    
    return state


def _circuit_energy(theta, n, coupling, field, H):
    """Compute <psi|H|psi> where psi is the circuit state."""
    psi = _apply_circuit_state(theta, n, layers=3)
    # Energy = psi^dagger H psi
    energy = np.vdot(psi, H @ psi).real
    return energy


def _objective_flat(theta_flat, n, coupling, field, H, ground_energy):
    """Objective function: (circuit energy - ground energy) / n"""
    theta = theta_flat.reshape(3, n)
    energy = _circuit_energy(theta, n, coupling, field, H)
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    
    Args:
        n: number of qubits (4 or 6)
        coupling: Ising coupling (1.0)
        field: transverse field (0.45, 1.0, 1.65)
        layers: number of layers (3)
    
    Returns:
        (layers, n) array of angles in radians
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build Hamiltonian and compute exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    dim = 1 << n
    total_params = layers * n
    
    # Strategy: Use differential evolution for global search, then L-BFGS-B for polish
    # For n=4: 12 parameters, for n=6: 18 parameters
    
    best_theta = None
    best_obj = np.inf
    
    # Initial guess: try a few heuristics
    # 1. All zeros
    # 2. Small angles
    # 3. Symmetric initialization
    
    initial_guesses = []
    
    # Guess 1: zeros
    initial_guesses.append(np.zeros(total_params))
    
    # Guess 2: small random
    rng = np.random.RandomState(42)
    initial_guesses.append(rng.uniform(-np.pi/4, np.pi/4, total_params))
    
    # Guess 3: symmetric (same angle for all qubits in each layer)
    for l in range(layers):
        initial_guesses.append(np.array([0.1 * (l + 1)] * n).flatten())
    
    # Try L-BFGS-B from each initial guess
    for init in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective_flat,
                init,
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_obj:
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # If we have time, do differential evolution for global search
    if time.time() < deadline - 30:
        # Differential evolution with bounded search
        # Angles are in radians, typically in [-pi, pi]
        bounds = [(-np.pi, np.pi)] * total_params
        
        try:
            de_result = differential_evolution(
                _objective_flat,
                bounds,
                args=(n, coupling, field, H, ground_energy),
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_theta = de_result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if we found a good solution
    if best_theta is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_flat,
                best_theta.flatten(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # If no good solution found, return a reasonable default
    if best_theta is None:
        # Product state approximation: try to find best product state
        # For each layer, try to optimize
        best_theta = np.zeros((layers, n))
        # Simple gradient-free search for product state
        for _ in range(50):
            if time.time() > deadline:
                break
            # Random perturbation
            perturbation = np.random.uniform(-0.1, 0.1, (layers, n))
            candidate = best_theta + perturbation
            obj = _objective_flat(candidate.flatten(), n, coupling, field, H, ground_energy)
            if obj < best_obj:
                best_obj = obj
                best_theta = candidate
    
    return best_theta
