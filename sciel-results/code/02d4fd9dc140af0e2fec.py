# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import math


def _build_quadrature(k=10):
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    return x, w, phi, dphi


def _compute_harmonics(x, phi, k=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -k..k at points (x=cos theta, phi).
    """
    n_x = len(x)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_x, n_phi, n_m), dtype=np.float64)
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        log_num = math.lgamma(k + 1 - abs_m)
        log_den = math.lgamma(k + 1 + abs_m)
        N_m = math.sqrt((2 * k + 1) / (4 * np.pi) * math.exp(log_num - log_den))
        
        P = lpmv(abs_m, k, x)
        
        if m == 0:
            Y[:, :, 10] = N_m * P[:, np.newaxis]
        elif m > 0:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return Y.reshape(n_x * n_phi, n_m)


def _compute_ratio(coeffs, Y, w, dphi):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs
    n_x = len(w)
    n_phi = Y.shape[0] // n_x
    weights_full = np.tile(w * dphi, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights_full * f2)
    norm4_sq = np.sum(weights_full * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4 / norm2


def _gaussian_mutation(pop, sigma):
    """Apply Gaussian mutation to population."""
    mutated = pop.copy()
    for i in range(len(pop)):
        mutated[i] += np.random.normal(0, sigma, size=pop.shape[1])
    return mutated


def _uniform_crossover(pop, p_crossover=0.8):
    """Uniform crossover between pairs of individuals."""
    n = len(pop)
    children = np.zeros_like(pop)
    for i in range(0, n - 1, 2):
        if np.random.random() < p_crossover:
            mask = np.random.random(pop.shape[1]) < 0.5
            children[i] = pop[i] * mask + pop[i+1] * (1 - mask)
            children[i+1] = pop[i+1] * mask + pop[i] * (1 - mask)
        else:
            children[i] = pop[i].copy()
            children[i+1] = pop[i+1].copy()
    if n % 2 == 1:
        children[-1] = pop[-1].copy()
    return children


def rudin_problem10() -> np.ndarray:
    k = 10
    n_m = 2 * k + 1
    
    # Build quadrature
    x, w, phi, dphi = _build_quadrature(k)
    
    # Compute harmonics at all quadrature points
    Y = _compute_harmonics(x, phi, k)
    
    # Genetic algorithm parameters
    pop_size = 60
    n_generations = 200
    elite_size = 6
    sigma = 0.5
    p_crossover = 0.8
    
    rng = np.random.default_rng(42)
    
    # Initialize population with diverse starting points
    pop = np.zeros((pop_size, n_m), dtype=np.float64)
    
    # Seed with structured initializations
    # 1. All ones
    pop[0] = np.ones(n_m)
    # 2. Only m=0
    pop[1] = np.zeros(n_m); pop[1, 10] = 1.0
    # 3. Only m=10 cos
    pop[2] = np.zeros(n_m); pop[2, 20] = 1.0
    # 4. Only m=10 sin
    pop[3] = np.zeros(n_m); pop[3, 0] = 1.0
    # 5. Random
    for i in range(4, pop_size):
        pop[i] = rng.standard_normal(n_m)
    
    # Evaluate initial population
    ratios = np.array([_compute_ratio(c, Y, w, dphi) for c in pop])
    
    # Track best
    best_idx = np.argmin(ratios)
    best_coeffs = pop[best_idx].copy()
    best_ratio = ratios[best_idx]
    
    for gen in range(n_generations):
        # Sort by fitness
        order = np.argsort(ratios)
        pop = pop[order]
        ratios = ratios[order]
        
        # Elitism: keep top elite_size
        new_pop = np.zeros_like(pop)
        new_pop[:elite_size] = pop[:elite_size]
        
        # Generate offspring
        for i in range(elite_size, pop_size):
            # Select parents via tournament selection
            p1_idx = rng.choice(pop_size, size=3, replace=False)
            p1 = pop[p1_idx[np.argmin(ratios[p1_idx])]]
            p2_idx = rng.choice(pop_size, size=3, replace=False)
            p2 = pop[p2_idx[np.argmin(ratios[p2_idx])]]
            
            # Crossover
            if rng.random() < p_crossover:
                mask = rng.random(n_m) < 0.5
                child = p1 * mask + p2 * (1 - mask)
            else:
                child = p1.copy()
            
            # Mutation
            child += rng.normal(0, sigma, size=n_m)
            
            new_pop[i] = child
        
        pop = new_pop
        ratios = np.array([_compute_ratio(c, Y, w, dphi) for c in pop])
        
        # Update best
        gen_best_idx = np.argmin(ratios)
        if ratios[gen_best_idx] < best_ratio:
            best_ratio = ratios[gen_best_idx]
            best_coeffs = pop[gen_best_idx].copy()
        
        # Adaptive sigma: decrease over generations
        sigma = 0.5 * (0.99 ** gen)
    
    coeffs = best_coeffs
    
    # Validate
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 0:
            coeffs = coeffs / max_abs
        else:
            return np.ones(n_m, dtype=np.float64)
    
    return coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
