"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math
import random


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    total = 0.0
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e18  # invalid
            terms.append(-math.log(d))
    return math.fsum(terms)


def _energy_grad(pts):
    """Gradient of logarithmic energy w.r.t. points."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j| = -(x_i-x_j)/d^2
            # Actually: log(1/d) = -log(d), d = |x_i - x_j|
            # d(-log d)/dx_i = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
            factor = -1.0 / (d * d)
            grad[i] += factor * diff
            grad[j] -= factor * diff
    return grad


def _project_to_sphere(pts):
    """Project each point onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _constrained_energy(pts_flat):
    """Energy function for optimizer that takes flat array and projects to sphere."""
    pts = pts_flat.reshape(-1, 3)
    pts = _project_to_sphere(pts)
    return _energy(pts)


def construct():
    # Start with Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use BFGS-like optimization with gradient descent on the sphere
    # We'll do iterative gradient descent with projection
    
    current_pts = pts.copy()
    current_energy = _energy(current_pts)
    
    # Gradient descent with adaptive step size
    lr = 0.01
    for iteration in range(5000):
        grad = _energy_grad(current_pts)
        
        # Move in negative gradient direction
        new_pts = current_pts - lr * grad
        
        # Project back to sphere
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy(new_pts)
        
        if new_energy < current_energy:
            current_pts = new_pts
            current_energy = new_energy
            # Slightly increase learning rate
            lr = min(lr * 1.01, 0.1)
        else:
            # Decrease learning rate
            lr *= 0.95
            if lr < 1e-8:
                break
    
    # Try a few random restarts with perturbation to escape local minima
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    for restart in range(20):
        # Perturb the best solution
        perturbation = np.random.randn(N, 3) * 0.1
        trial_pts = _project_to_sphere(best_pts + perturbation)
        trial_energy = _energy(trial_pts)
        
        if trial_energy < best_energy:
            best_pts = trial_pts
            best_energy = trial_energy
        
        # Gradient descent from this perturbed point
        pts_local = trial_pts.copy()
        energy_local = trial_energy
        lr_local = 0.01
        
        for iteration in range(3000):
            grad = _energy_grad(pts_local)
            new_pts = pts_local - lr_local * grad
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy(new_pts)
            
            if new_energy < energy_local:
                pts_local = new_pts
                energy_local = new_energy
                lr_local = min(lr_local * 1.01, 0.1)
            else:
                lr_local *= 0.95
                if lr_local < 1e-8:
                    break
        
        if energy_local < best_energy:
            best_pts = pts_local
            best_energy = energy_local
    
    return _project_to_sphere(best_pts)
# EVOLVE-BLOCK-END
