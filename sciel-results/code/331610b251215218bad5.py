"""
Morse cluster global minima finder using MD annealing + basin hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline
_DEADLINE = time.time() + 290  # 290s safety margin before 330s kill


def _morse_energy(pos, rho):
    """Compute exact Morse energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    iu = np.triu_indices(n, k=1)
    xij = x[iu]
    e = np.exp(xij) * (np.exp(xij) - 2.0)
    return float(np.sum(e))


def _morse_energy_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    exp_x = np.exp(x)
    # dE/dr = -rho * exp_x * (2*exp_x - 1)
    dEdr = -rho * exp_x * (2.0 * exp_x - 1.0)
    # Gradient: sum over j of dEdr_ij * (r_i - r_j)/r_ij
    # grad_i = sum_j dEdr_ij * (pos_i - pos_j) / r_ij
    # Use vectorized: grad_i = sum_j dEdr_ij * (pos_i - pos_j) / r_ij
    # = sum_j dEdr_ij/r_ij * (pos_i - pos_j)
    # = pos_i * sum_j (dEdr_ij/r_ij) - sum_j (dEdr_ij/r_ij) * pos_j
    coeff = dEdr / r  # shape (n, n)
    np.fill_diagonal(coeff, 0.0)
    grad = pos * coeff.sum(axis=1)[:, None] - coeff @ pos
    # Energy
    iu = np.triu_indices(n, k=1)
    e = exp_x[iu] * (exp_x[iu] - 2.0)
    energy = float(np.sum(e))
    return energy, grad


def _md_anneal(pos0, rho, n_steps=500, dt=0.005, T0=2.0, T1=0.01, seed=42):
    """
    Simple velocity-Verlet MD with linear cooling.
    Returns final positions and energy.
    """
    rng = np.random.default_rng(seed)
    pos = pos0.copy().astype(np.float64)
    vel = rng.normal(0, np.sqrt(T0), size=pos.shape)
    
    # Initialize
    E, grad = _morse_energy_grad(pos, rho)
    
    best_pos = pos.copy()
    best_E = E
    
    n = len(pos)
    
    for step in range(n_steps):
        # Linear cooling
        frac = step / max(n_steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        
        # Velocity Verlet
        # v_half = v + 0.5*dt*F/m  (m=1)
        vel_half = vel + 0.5 * dt * (-grad)
        # r_new = r + dt*v_half
        pos_new = pos + dt * vel_half
        # Compute forces at new position
        E_new, grad_new = _morse_energy_grad(pos_new, rho)
        # v_new = v_half + 0.5*dt*F_new/m
        vel_new = vel_half + 0.5 * dt * (-grad_new)
        
        # Thermostat: scale velocities to target temperature
        KE = 0.5 * np.sum(vel_new ** 2)
        target_KE = 1.5 * n * T  # 3 DOF per atom
        if KE > 1e-12:
            vel_new *= np.sqrt(target_KE / KE)
        
        pos = pos_new
        vel = vel_new
        E = E_new
        grad = grad_new
        
        if E < best_E:
            best_E = E
            best_pos = pos.copy()
    
    return best_pos, best_E


def _basin_hopping(pos, rho, n_restarts=5, step_size=0.5, seed=42):
    """
    Basin hopping: random perturb + local minimization.
    """
    rng = np.random.default_rng(seed)
    best_pos = pos.copy()
    best_E = _morse_energy(pos, rho)
    
    for i in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        # Random perturbation
        noise = rng.normal(0, step_size, size=pos.shape)
        pos_pert = pos + noise
        
        # Local minimization using BFGS
        try:
            res = minimize(
                lambda p: _morse_energy_grad(p, rho),
                pos_pert,
                jac=True,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_E:
                E_new = res.fun
                if E_new < best_E:
                    best_E = E_new
                    best_pos = res.x.copy()
        except Exception:
            pass
    
    return best_pos, best_E


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    atoms = []
    # FCC lattice points
    # Generate points in a cube and filter
    max_r = spacing * (n ** (1/3) + 2)
    n_pts = int(max_r * 4)
    pts = np.stack(np.meshgrid(
        np.linspace(-max_r, max_r, n_pts),
        np.linspace(-max_r, max_r, n_pts),
        np.linspace(-max_r, max_r, n_pts),
        indexing='ij'
    ), axis=-1).reshape(-1, 3)
    
    # FCC: keep points where all coords are even or exactly one is odd (scaled)
    # Simpler: just use a dense packing
    # Use a spherical shell approach
    atoms = []
    # Center atom
    atoms.append(np.array([0.0, 0.0, 0.0]))
    
    # Add layers
    shells = [
        [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],  # 6
        [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)],  # 12
        [(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2),(2,1,0),(2,-1,0),(-2,1,0),(-2,-1,0),(1,2,0),(-1,2,0),(1,-2,0),(-1,-2,0),(0,2,1),(0,2,-1),(0,-2,1),(0,-2,-1),(2,0,1),(2,0,-1),(-2,0,1),(-2,0,-1),(0,1,2),(0,1,-2),(0,-1,2),(0,-1,-2),(1,0,2),(-1,0,2),(1,0,-2),(-1,0,-2)],  # 32
    ]
    
    for shell in shells:
        for v in shell:
            if len(atoms) >= n:
                break
            atoms.append(np.array(v, dtype=float) * spacing * 0.72)
        if len(atoms) >= n:
            break
    
    # If still not enough, add random atoms near the cluster
    while len(atoms) < n:
        p = rng_uniform_atom(len(atoms))
        atoms.append(p)
    
    return np.array(atoms[:n])


def rng_uniform_atom(n_existing):
    """Generate a random atom position."""
    radius = 1.0 * n_existing ** (1/3) * 1.5
    while True:
        p = np.random.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            return p


def construct(instance):
    """
    Find low-energy Morse cluster configurations using MD annealing + basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    start_time = time.time()
    
    # Strategy: Multiple independent runs of MD annealing + basin hopping
    # Use different random seeds and initial configurations
    
    best_pos = None
    best_E = np.inf
    
    n_configs = 8  # Number of independent attempts
    
    for attempt in range(n_configs):
        if time.time() > _DEADLINE:
            break
        
        # Generate initial configuration
        if attempt == 0:
            # FCC-like initial guess
            pos0 = _generate_fcc(size, spacing=1.0)
        elif attempt == 1:
            # Spherical shell
            pos0 = _generate_shell(size)
        else:
            # Random compact cluster
            pos0 = _generate_random_compact(size, seed=1000 + attempt)
        
        # MD annealing
        pos_md, E_md = _md_anneal(pos0, rho, n_steps=300, dt=0.005, T0=1.5, T1=0.05, seed=2000 + attempt)
        
        # Basin hopping from MD result
        pos_bh, E_bh = _basin_hopping(pos_md, rho, n_restarts=3, step_size=0.4, seed=3000 + attempt)
        
        # Final local relaxation using the provided relax function
        try:
            E_relax, pos_final = relax(pos_bh, potential, rho)
        except Exception:
            E_relax = E_bh
            pos_final = pos_bh
        
        if E_relax < best_E:
            best_E = E_relax
            best_pos = pos_final.copy()
    
    if best_pos is None:
        # Fallback: just use relax on a random config
        pos0 = _generate_random_compact(size, seed=9999)
        E_relax, best_pos = relax(pos0, potential, rho)
    
    return best_pos


def _generate_shell(n):
    """Generate atoms on spherical shells."""
    atoms = [np.array([0.0, 0.0, 0.0])]
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2]
    shell_counts = [6, 12, 24, 42, 60]
    
    for radius, count in zip(shell_radii, shell_counts):
        for i in range(count):
            if len(atoms) >= n:
                break
            # Fibonacci sphere for even distribution
            golden = (1 + np.sqrt(5)) / 2
            theta = np.arccos(1 - 2 * (i + 0.5) / count)
            phi = 2 * np.pi * i / golden
            p = radius * np.array([np.sin(theta) * np.cos(phi),
                                    np.sin(theta) * np.sin(phi),
                                    np.cos(theta)])
            atoms.append(p)
        if len(atoms) >= n:
            break
    
    while len(atoms) < n:
        p = np.random.uniform(-5, 5, 3)
        atoms.append(p)
    
    return np.array(atoms[:n])


def _generate_random_compact(n, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1/3) * 0.8
    points = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check minimum distance
                if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                    points.append(p)
                    break
    return np.array(points)
# EVOLVE-BLOCK-END
