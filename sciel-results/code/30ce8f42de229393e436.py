"""Optimized quantum circuit parameters for Ising ground states.
Strategy: Global search via Differential Evolution with physics-based seeding.
Exploits open-chain symmetry (mirror symmetry) to reduce effective search space.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 145.0
_START_TIME = time.time()


def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i using Kronecker products."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H = np.zeros((total_dim, total_dim), dtype=complex)

    # ZZ terms
    for i in range(n - 1):
        op = np.array([[1.0]])
        for q in range(n):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op

    # X terms
    for i in range(n):
        op = np.array([[1.0]])
        for q in range(n):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op

    return H


def _get_ground_energy(H):
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0].real)


def _circuit_state(params, n, layers):
    """Build circuit state using vectorized operations."""
    total_dim = 2 ** n
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0

    # Precompute bit masks
    for layer in range(layers):
        # Apply Ry to every qubit (vectorized)
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            new_state = np.zeros_like(state)
            for idx in range(total_dim):
                if (idx & mask) == 0:
                    new_state[idx] += c * state[idx]
                    new_state[idx | mask] += s * state[idx]
                else:
                    new_state[idx] += c * state[idx]
                    new_state[idx ^ mask] += -s * state[idx]
            state = new_state

        # Apply CZ between adjacent pairs
        if layer < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                for idx in range(total_dim):
                    if (idx & mask1) and (idx & mask2):
                        state[idx] = -state[idx]

    return state


def _circuit_energy(params, n, coupling, field, layers, H):
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _physics_seed(n, coupling, field, layers):
    """
    Generate physics-based initial parameters.
    For open chain, the ground state is symmetric.
    We estimate the 'natural' rotation angle for each site based on local field/coupling balance.
    Near boundaries, the effective field is stronger relative to coupling, so angles should be larger (closer to pi/2).
    In the middle, coupling dominates, so angles should be smaller.
    """
    params = np.zeros((layers, n))
    
    # Estimate local effective field ratio
    # For a bulk spin, the effective transverse field is reduced by neighboring alignment.
    # Simple heuristic:
    # angle ~ arctan(field / (coupling * neighbors))
    # Boundary spins have 1 neighbor, bulk have 2.
    
    neighbors = np.ones(n)
    for i in range(n):
        if i == 0 or i == n - 1:
            neighbors[i] = 1.0
        else:
            neighbors[i] = 2.0
            
    # Effective ratio
    ratios = field / (coupling * neighbors)
    base_angles = np.arctan(ratios)
    
    # Distribute angles across layers.
    # Layer 0: large rotation to get close to product state.
    # Layer 1: refinement.
    # Layer 2: final correction.
    
    # We want the final state to have these angles.
    # Since Ry is applied sequentially, we can't just set one layer.
    # However, for a shallow circuit, we can approximate that the first layer does most of the work.
    # Let's assign weights to layers.
    weights = np.array([0.7, 0.2, 0.1])
    
    for l in range(layers):
        params[l, :] = base_angles * weights[l]
        
    return params


def angles(n, coupling, field, layers):
    """Optimize circuit parameters for the given Ising chain."""
    global _START_TIME
    _START_TIME = time.time()

    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(H)

    # 1. Generate diverse initial points
    # A: Physics-based
    p_physics = _physics_seed(n, coupling, field, layers)
    
    # B: Uniform small angles
    p_uniform_small = np.full((layers, n), 0.2)
    
    # C: Uniform large angles
    p_uniform_large = np.full((layers, n), 0.5)
    
    # D: Zero
    p_zero = np.zeros((layers, n))
    
    # E: Random
    np.random.seed(42)
    p_random = np.random.uniform(-np.pi/4, np.pi/4, size=(layers, n))
    
    candidates = [p_physics, p_uniform_small, p_uniform_large, p_zero, p_random]
    
    # Evaluate candidates to find best starting point for local refinement
    best_init = None
    best_init_score = np.inf
    
    for p in candidates:
        if not _check_deadline():
            break
        try:
            e = _circuit_energy(p, n, coupling, field, layers, H)
            score = (e - ground_energy) / n
            if score < best_init_score:
                best_init_score = score
                best_init = p.copy()
        except Exception:
            continue
            
    if best_init is None:
        best_init = p_physics
        best_init_score = np.inf

    # 2. Global Search with Differential Evolution
    # DE is robust against local minima.
    # We bound the angles to [-pi/2, pi/2] as rotations beyond this are often redundant or suboptimal for this shallow circuit depth.
    # Actually, Ry(theta) = Ry(theta + 2pi). And Ry(-theta) is related by global phase/sign.
    # Let's bound to [-pi, pi] to be safe, but [-pi/2, pi/2] is tighter.
    bounds = [(-np.pi/2, np.pi/2)] * (layers * n)
    
    def objective_flat(x):
        p = x.reshape(layers, n)
        e = _circuit_energy(p, n, coupling, field, layers, H)
        return (e - ground_energy) / n

    best_params = best_init
    best_score = best_init_score
    
    try:
        # Use DE with a small population to save time, but enough to explore.
        # maxiter=50, popsize=15 -> 750 evaluations.
        # Each evaluation is fast (n=6, dim=64).
        result_de = differential_evolution(
            objective_flat,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            atol=1e-8,
            polish=True  # Use L-BFGS-B on the best result
        )
        
        if result_de.fun < best_score:
            best_score = result_de.fun
            best_params = result_de.x.reshape(layers, n)
            
    except Exception:
        pass

    # 3. Local Refinement from Best
    if _check_deadline(margin=5.0):
        try:
            result_local = minimize(
                objective_flat,
                best_params.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result_local.fun < best_score:
                best_score = result_local.fun
                best_params = result_local.x.reshape(layers, n)
        except Exception:
            pass

    # 4. Symmetry Check: Mirror symmetry
    # For open chain, H is invariant under mirror reflection i -> n-1-i.
    # The ground state can be chosen to be symmetric.
    # If the circuit respects this symmetry, params[l, i] should equal params[l, n-1-i].
    # Let's try enforcing this symmetry to reduce search space dimensionality by ~half.
    # This is a structural change: reducing the parameter space.
    
    if _check_deadline(margin=10.0):
        # Define symmetric parameterization
        # For n=4: indices 0,1,2,3. Pairs: (0,3), (1,2). Unique: 0,1.
        # For n=6: indices 0..5. Pairs: (0,5), (1,4), (2,3). Unique: 0,1,2.
        
        def objective_symmetric(x_sym):
            p = np.zeros((layers, n))
            for l in range(layers):
                for i in range(n // 2):
                    val = x_sym[l * (n // 2) + i]
                    p[l, i] = val
                    p[l, n - 1 - i] = val
            return objective_flat(p.flatten())
            
        sym_bounds = [(-np.pi/2, np.pi/2)] * (layers * (n // 2))
        
        # Seed symmetric search with symmetric version of best_params
        sym_seed = np.zeros(layers * (n // 2))
        for l in range(layers):
            for i in range(n // 2):
                # Average the symmetric partners
                sym_seed[l * (n // 2) + i] = (best_params[l, i] + best_params[l, n - 1 - i]) / 2.0
        
        try:
            result_sym = differential_evolution(
                objective_symmetric,
                sym_bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-8,
                atol=1e-8,
                polish=True
            )
            
            if result_sym.fun < best_score:
                # Reconstruct full params
                p_sym_best = np.zeros((layers, n))
                x_best = result_sym.x
                for l in range(layers):
                    for i in range(n // 2):
                        val = x_best[l * (n // 2) + i]
                        p_sym_best[l, i] = val
                        p_sym_best[l, n - 1 - i] = val
                best_score = result_sym.fun
                best_params = p_sym_best
                
        except Exception:
            pass

    # Ensure finite output
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)

    return best_params
# EVOLVE-BLOCK-END
