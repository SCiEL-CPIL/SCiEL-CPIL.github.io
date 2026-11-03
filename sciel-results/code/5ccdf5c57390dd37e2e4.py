"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi, cos, sin

# Pre-compute analytic matrices for the objective function
# We want to minimize Delta_x^2 = <x^2> / <1> for f = g * exp(-x^2/2)
# where g = c0 H0 + c4 H4 + c8 H8 + c12 H12
# <1> = sum c_i c_j M_ij
# <x^2> = sum c_i c_j N_ij
# M_ij = int H_i H_j exp(-x^2) dx
# N_ij = int x^2 H_i H_j exp(-x^2) dx

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Orthogonality: int H_n H_m exp(-x^2) dx = sqrt(pi) 2^n n! delta_nm
# So M is diagonal.
M_diag = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# N_ij = int x^2 H_i H_j exp(-x^2) dx
# For this basis, N is diagonal: N_ii = i * M_ii
N_diag = np.array([n * m for n, m in zip(indices, M_diag)])

# c12 is determined by c0, c4, c8 to satisfy g(0)=0
H0_0 = _hermite_val(0, 0)   # 1
H4_0 = _hermite_val(4, 0)   # 12
H8_0 = _hermite_val(8, 0)   # 168
H12_0 = _hermite_val(12, 0) # 20901120

def _compute_bound_analytic(c0, c4, c8):
    """Compute the uncertainty bound for given coefficients."""
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Denominator: sum c_i^2 M_ii
    norm_sq = np.sum(coeffs**2 * M_diag)
    if norm_sq <= 1e-12:
        return 1e10
        
    # Numerator: sum c_i^2 N_ii
    x2_expectation = np.sum(coeffs**2 * N_diag)
    
    # Variance Delta_x^2 = <x^2> - <x>^2. <x>=0 due to symmetry.
    # Product Delta_x Delta_xi = Delta_x^2 because Delta_xi = Delta_x.
    return x2_expectation / norm_sq

def _objective_spherical(theta, phi):
    """
    Objective function in spherical coordinates.
    theta in [0, pi], phi in [0, 2*pi]
    c0 = cos(theta)
    c4 = sin(theta) * cos(phi)
    c8 = sin(theta) * sin(phi)
    """
    c0 = cos(theta)
    c4 = sin(theta) * cos(phi)
    c8 = sin(theta) * sin(phi)
    return _compute_bound_analytic(c0, c4, c8)

def _differential_evolution_2d(objective, bounds, pop_size=30, max_iter=200, 
                                F=0.8, CR=0.5, timeout=100.0):
    """
    Simple Differential Evolution for 2D optimization.
    bounds: list of (low, high) for each dimension.
    Returns best_x, best_f.
    """
    start_time = time.time()
    dim = len(bounds)
    
    # Initialize population
    np.random.seed(42)
    population = np.random.uniform(bounds[0][0], bounds[1][0], (pop_size, dim))
    # Actually need to handle each dimension's bounds
    population = np.array([
        [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
        for _ in range(pop_size)
    ])
    
    # Evaluate initial population
    fitness = np.array([objective(population[i, 0], population[i, 1]) for i in range(pop_size)])
    
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    
    for it in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        # Create trial population
        trial_population = population.copy()
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Differential mutation
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            trial = mutant.copy()
            for j in range(dim):
                if np.random.rand() < CR or j == np.random.randint(dim):
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i, j]
            
            # Clamp to bounds
            for j in range(dim):
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])
            
            # Evaluate trial
            trial_f = objective(trial[0], trial[1])
            
            # Selection
            if trial_f <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_f
                
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
    
    return best_x, best_f

def construct():
    """Find optimal coefficients for H0, H4, H8 using Differential Evolution in spherical coordinates."""
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin
    
    # Use Differential Evolution in 2D spherical coordinates
    # theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0.0, np.pi), (0.0, 2.0 * np.pi)]
    
    try:
        best_theta_phi, best_val = _differential_evolution_2d(
            _objective_spherical,
            bounds,
            pop_size=40,
            max_iter=300,
            F=0.8,
            CR=0.5,
            timeout=120.0
        )
        
        # Convert back to Cartesian
        c0 = cos(best_theta_phi[0])
        c4 = sin(best_theta_phi[0]) * cos(best_theta_phi[1])
        c8 = sin(best_theta_phi[0]) * sin(best_theta_phi[1])
        best_u = np.array([c0, c4, c8])
        
    except Exception as e:
        # Fallback to seed value
        best_u = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
        best_val = _compute_bound_analytic(*best_u)
    
    # Local refinement using Nelder-Mead for extra precision
    from scipy.optimize import minimize
    if time.time() < deadline:
        try:
            res = minimize(
                lambda x: _compute_bound_analytic(x[0], x[1], x[2]),
                best_u,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_u = res.x
        except Exception:
            pass
    
    return best_u
# EVOLVE-BLOCK-END
