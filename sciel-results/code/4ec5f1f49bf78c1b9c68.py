"""Optimized shallow-circuit Ising ground state approximation - structured ansatz + DE."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_T0 = None

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _time_left(margin=5.0):
    return 148.0 - _elapsed() - margin

def _build_hamiltonian(n, coupling, field):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for s in range(dim):
            bi = (s >> i) & 1
            bip1 = (s >> (i + 1)) & 1
            zz = (1 - 2 * bi) * (1 - 2 * bip1)
            H[s, s] += -coupling * zz
    for i in range(n):
        for s in range(dim):
            H[s, s ^ (1 << i)] += -field
    return H

def _ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    return np.linalg.eigvalsh(H)[0]

def _circuit_state(theta, n, layers):
    """Evolve |0>^n through Ry/CZ circuit. theta is (layers, n)."""
    dim = 2 ** n
    # Represent state as tensor of shape (2,)*n
    state = np.zeros((2,) * n, dtype=np.complex128)
    idx0 = [0] * n
    state[tuple(idx0)] = 1.0

    for l in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            th = theta[l, q]
            c = np.cos(th / 2.0)
            s = np.sin(th / 2.0)
            # Ry = [[c, -s], [s, c]]
            # For qubit q, new[...0...] = c*old[...0...] - s*old[...1...]
            # new[...1...] = s*old[...0...] + c*old[...1...]
            sl0 = [slice(None)] * n
            sl1 = [slice(None)] * n
            sl0[q] = 0
            sl1[q] = 1
            old0 = state[tuple(sl0)].copy()
            old1 = state[tuple(sl1)].copy()
            state[tuple(sl0)] = c * old0 - s * old1
            state[tuple(sl1)] = s * old0 + c * old1

        # Apply CZ on adjacent pairs in ascending order (if not last layer)
        if l < layers - 1:
            for i in range(n - 1):
                # CZ: phase -1 when both qubits are |1>
                sl = [slice(None)] * n
                sl[i] = 1
                sl[i + 1] = 1
                state[tuple(sl)] *= -1.0

    return state.reshape(dim)

def _energy(theta, n, coupling, field, H, ground_e):
    state = _circuit_state(theta, n, layers=3)
    e = np.real(np.vdot(state, H @ state))
    return (e - ground_e) / n

def _structured_theta(base, profile, n, layers):
    """Build theta from structured ansatz.
    profile: (layers,) array of scaling factors in [0, 1].
    theta[l, q] = (-1)^q * base * profile[l]
    """
    theta = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            sign = 1.0 if q % 2 == 0 else -1.0
            theta[l, q] = sign * base * profile[l]
    return theta

def _unstructured_theta(base, profile, n, layers):
    """Alternative: all qubits same sign, varying profile."""
    theta = np.zeros((layers, n))
    for l in range(layers):
        theta[l, :] = base * profile[l]
    return theta

def _objective_structured(x, n, coupling, field, H, ground_e, staggered=True):
    """x = [base, p0, p1, p2] where p are profile values in [0, 2]."""
    base = x[0]
    profile = x[1:]
    if staggered:
        theta = _structured_theta(base, profile, n, 3)
    else:
        theta = _unstructured_theta(base, profile, n, 3)
    return _energy(theta, n, coupling, field, H, ground_e)

def _objective_free(x, n, coupling, field, H, ground_e):
    theta = x.reshape(3, n)
    return _energy(theta, n, coupling, field, H, ground_e)

def _differential_evolution(func, bounds, popsize=10, maxiter=100, tol=1e-10, seed=42):
    """Simple differential evolution."""
    rng = np.random.default_rng(seed)
    ndim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.array([lb + rng.random(ndim) * (ub - lb) for _ in range(popsize)])
    pop[0] = (lb + ub) / 2.0  # center
    
    fvals = np.array([func(p) for p in pop])
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(maxiter):
        if _time_left(10.0) < 5.0:
            break
        new_pop = []
        new_fvals = []
        for i in range(popsize):
            # Select three distinct random individuals
            r1, r2, r3 = rng.choice(popsize, 3, replace=False)
            F = 0.8
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            # Clip to bounds
            mutant = np.clip(mutant, lb, ub)
            # Crossover
            CR = 0.5
            trial = pop[i].copy()
            for j in range(ndim):
                if rng.random() < CR:
                    trial[j] = mutant[j]
            if rng.random() > CR:
                # Ensure at least one gene is from mutant
                j = rng.randint(ndim)
                trial[j] = mutant[j]
            f_trial = func(trial)
            if f_trial <= fvals[i]:
                new_pop.append(trial)
                new_fvals.append(f_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
            else:
                new_pop.append(pop[i])
                new_fvals.append(fvals[i])
        pop = np.array(new_pop)
        fvals = np.array(new_fvals)
        
        # Convergence check
        if np.max(fvals) - np.min(fvals) < tol:
            break
    
    return best_x, best_f

def angles(n, coupling, field, layers):
    H = _build_hamiltonian(n, coupling, field)
    ground_e = _ground_energy(n, coupling, field)
    
    best_theta = None
    best_val = np.inf
    
    def update(theta, val):
        nonlocal best_theta, best_val
        if val < best_val:
            best_val = val
            best_theta = theta.copy()
    
    # --- Phase 1: Structured ansatz search (low-dimensional, physics-guided) ---
    # base angle: arctan2(field, coupling) is the mean-field tilt
    base_mf = np.arctan2(field, coupling)
    
    # Bounds for structured: base in [-pi/2, pi/2], profiles in [0, 2]
    bounds_struct = [(-np.pi/2, np.pi/2)] + [(0.0, 2.0)] * 3
    
    # Try staggered and non-staggered
    for staggered in [True, False]:
        if _time_left(10.0) < 10.0:
            break
        func = lambda x: _objective_structured(x, n, coupling, field, H, ground_e, staggered)
        
        # DE search
        x_best, f_best = _differential_evolution(
            func, bounds_struct, popsize=12, maxiter=80, tol=1e-10, seed=42
        )
        profile = x_best[1:]
        base = x_best[0]
        if staggered:
            theta = _structured_theta(base, profile, n, 3)
        else:
            theta = _unstructured_theta(base, profile, n, 3)
        val = _energy(theta, n, coupling, field, H, ground_e)
        update(theta, val)
        
        # Polish with L-BFGS-B on free parameters from this starting point
        if _time_left(10.0) > 5.0:
            try:
                res = minimize(
                    lambda x: _objective_free(x, n, coupling, field, H, ground_e),
                    theta.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * (3 * n),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                theta_pol = res.x.reshape(3, n)
                val_pol = _energy(theta_pol, n, coupling, field, H, ground_e)
                update(theta_pol, val_pol)
            except Exception:
                pass
    
    # --- Phase 2: Multi-start free-parameter search ---
    if _time_left(10.0) > 15.0:
        # Start from mean-field inspired
        theta_mf = np.zeros((3, n))
        for l in range(3):
            frac = (l + 1) / 3.0
            for q in range(n):
                sign = 1.0 if q % 2 == 0 else -1.0
                theta_mf[l, q] = sign * base_mf * frac
        val = _energy(theta_mf, n, coupling, field, H, ground_e)
        update(theta_mf, val)
        
        if _time_left(10.0) > 10.0:
            try:
                res = minimize(
                    lambda x: _objective_free(x, n, coupling, field, H, ground_e),
                    theta_mf.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * (3 * n),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                theta_pol = res.x.reshape(3, n)
                val_pol = _energy(theta_pol, n, coupling, field, H, ground_e)
                update(theta_pol, val_pol)
            except Exception:
                pass
    
    # --- Phase 3: Random restarts with DE on free params ---
    if _time_left(10.0) > 20.0:
        rng = np.random.default_rng(123)
        for restart in range(3):
            if _time_left(10.0) < 10.0:
                break
            x_init = rng.uniform(-np.pi, np.pi, size=(3, n))
            # DE on free parameters (small pop, few iters)
            bounds_free = [(-np.pi, np.pi)] * (3 * n)
            func_free = lambda x: _objective_free(x, n, coupling, field, H, ground_e)
            x_best, f_best = _differential_evolution(
                func_free, bounds_free, popsize=8, maxiter=50, tol=1e-10, seed=100+restart
            )
            theta_rand = x_best.reshape(3, n)
            val = _energy(theta_rand, n, coupling, field, H, ground_e)
            update(theta_rand, val)
            
            # Polish
            if _time_left(10.0) > 5.0:
                try:
                    res = minimize(
                        lambda x: _objective_free(x, n, coupling, field, H, ground_e),
                        theta_rand.flatten(),
                        method='L-BFGS-B',
                        bounds=[(-np.pi, np.pi)] * (3 * n),
                        options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    theta_pol = res.x.reshape(3, n)
                    val_pol = _energy(theta_pol, n, coupling, field, H, ground_e)
                    update(theta_pol, val_pol)
                except Exception:
                    pass
    
    # --- Phase 4: Final polish on best ---
    if best_theta is not None and _time_left(5.0) > 3.0:
        try:
            res = minimize(
                lambda x: _objective_free(x, n, coupling, field, H, ground_e),
                best_theta.flatten(),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (3 * n),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            theta_final = res.x.reshape(3, n)
            val_final = _energy(theta_final, n, coupling, field, H, ground_e)
            update(theta_final, val_final)
        except Exception:
            pass
    
    if best_theta is None:
        best_theta = np.zeros((3, n))
    
    return best_theta
# EVOLVE-BLOCK-END
