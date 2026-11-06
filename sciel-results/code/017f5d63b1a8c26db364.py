"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _energy_only(pos):
    """Compute energy only from positions."""
    try:
        result = dzugutov_energy_gradient(pos)
        if isinstance(result, tuple):
            return result[0]
        return result
    except Exception:
        return float('inf')

def _gradient_only(pos):
    """Compute gradient only from positions."""
    try:
        result = dzugutov_energy_gradient(pos)
        if isinstance(result, tuple):
            return result[1]
        # If it returns just gradient, return as-is
        return result
    except Exception:
        return np.zeros_like(pos)

def _local_minimize(pos, max_iter=2000):
    """
    Use L-BFGS-B with gradient to find the local minimum.
    Returns (energy, position).
    """
    try:
        def energy_func(x):
            return _energy_only(x)
        
        def gradient_func(x):
            return _gradient_only(x)
        
        res = minimize(
            energy_func,
            pos,
            jac=gradient_func,
            method='L-BFGS-B',
            options={'disp': False, 'maxiter': max_iter}
        )
        
        if res.success or res.fun < float('inf'):
            return res.fun, res.x
        else:
            # Fallback to relax
            energy, pos_relaxed = relax(pos, 'dzugutov', None)
            return energy, pos_relaxed
    except Exception:
        try:
            energy, pos_relaxed = relax(pos, 'dzugutov', None)
            return energy, pos_relaxed
        except Exception:
            return float('inf'), pos

def _generate_icosahedral_shells(n, rng):
    """
    Generate n points arranged in concentric icosahedral shells (Mackay icosahedra).
    Shell sizes: 1, 12, 42, 92, 150, ...
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    remaining = n
    
    # Shell 0: Center (1 atom)
    if remaining > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # Shell 1: 12 vertices
    if remaining > 0:
        shell_count = min(12, remaining)
        r1 = 1.1
        for i in range(shell_count):
            p = verts[i] * r1
            points.append(p)
        remaining -= shell_count
    
    # Shell 2: 42 vertices (approximate using icosahedron face centers + edges)
    # For simplicity, use a denser set of points on the icosahedron surface
    if remaining > 0:
        # Generate points on icosahedron surface using subdivision
        # For now, use random points on sphere of radius ~2.0
        r2 = 2.0
        for _ in range(remaining):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = v * r2
            points.append(p)
        remaining = 0
    
    return np.array(points)

def _generate_random_shell(n, rng, r_min=1.0, r_max=2.5):
    """Generate n points randomly distributed in a spherical shell."""
    pts = np.zeros((n, 3))
    for i in range(n):
        r = (r_min**3 + rng.random() * (r_max**3 - r_min**3)) ** (1/3)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts[i] = v * r
    return pts

def construct(instance):
    """
    Finds the lowest-energy structure for a Dzugutov cluster of size n.
    Uses Basin Hopping with icosahedral initializations and L-BFGS-B local minimization.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get('rho', instance.get('kind', None))
    
    start_time = time.time()
    time_limit = 45.0  # Leave buffer for 6 instances in 300s
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Basin Hopping Parameters
    T_start = 0.5
    alpha = 0.95
    n_bh_steps = 30
    step_scale_initial = 0.3
    
    # Generate diverse initializations
    inits = []
    
    # 1. Icosahedral shell structure
    inits.append(_generate_icosahedral_shells(size, rng))
    
    # 2. Random shell
    inits.append(_generate_random_shell(size, rng))
    
    # 3. Another random shell with different seed
    rng2 = np.random.default_rng(123)
    inits.append(_generate_random_shell(size, rng2))
    
    # 4. Tighter random shell
    rng3 = np.random.default_rng(456)
    inits.append(_generate_random_shell(size, rng3, r_min=0.8, r_max=1.8))
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb initial position slightly
        pos = init_pos + rng.normal(0, 0.05, init_pos.shape)
        
        # Initial Relaxation
        try:
            current_energy, current_pos = _local_minimize(pos)
        except Exception:
            try:
                current_energy, current_pos = relax(pos, potential, parameter)
            except Exception:
                continue
        
        if current_energy >= float('inf'):
            continue
        
        best_local_energy = current_energy
        best_local_pos = current_pos.copy()
        
        T = T_start
        step_scale = step_scale_initial
        
        for i in range(n_bh_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Kick
            kick = rng.normal(0, step_scale, current_pos.shape)
            new_pos = current_pos + kick
            
            # Relax
            try:
                new_energy, new_pos_relaxed = _local_minimize(new_pos)
            except Exception:
                try:
                    new_energy, new_pos_relaxed = relax(new_pos, potential, parameter)
                except Exception:
                    continue
            
            if new_energy >= float('inf'):
                continue
            
            # Acceptance Criterion (Metropolis)
            delta = new_energy - current_energy
            if delta < 0 or (T > 1e-6 and rng.random() < np.exp(-delta / T)):
                current_energy = new_energy
                current_pos = new_pos_relaxed
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_pos = current_pos.copy()
            
            # Cool down and reduce step size
            T *= alpha
            step_scale *= 0.97
        
        # Update global best
        if best_local_energy < best_energy:
            best_energy = best_local_energy
            best_positions = best_local_pos

    if best_positions is None:
        # Absolute fallback
        rng = np.random.default_rng(size)
        radius = 2.0
        points = rng.uniform(-radius, radius, (size, 3))
        try:
            e, best_positions = relax(points, potential, parameter)
        except Exception:
            best_positions = points

    return best_positions

# EVOLVE-BLOCK-END
