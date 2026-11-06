"""
Strategy: Global optimization via Differential Evolution (DE) with exact state-vector simulation.
The QAOA landscape for p=2 is highly non-convex with many local optima. Local methods like
Nelder-Mead or BFGS are sensitive to initialization and often converge to suboptimal basins.
Differential Evolution is a robust global optimizer that explores the parameter space broadly.
For n <= 10, the state space dimension is at most 1024, allowing fast exact simulation.
We use DE to scan the space [0, pi] x [0, pi] (with appropriate scaling) to find the global
or near-global optimum, avoiding the pitfalls of local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import sys

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """
    Build the problem Hamiltonian C (MaxCut) and mixing Hamiltonian B (sum X).
    C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    B = sum_{i=0}^{n-1} X_i
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
    
    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    return C, B

def _expected_cut_value(n, edges, angles, C, B):
    """
    Compute the expected cut value for given angles.
    angles: (p, 2) array of (gamma, beta)
    """
    if len(edges) == 0:
        return 0.0
        
    dim = 2**n
    
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    p = len(angles)
    for i in range(p):
        gamma, beta = angles[i]
        
        # Apply exp(-i gamma C)
        # Pre-compute exponentials? No, gamma changes.
        # For small dim, expm is fast.
        try:
            from scipy.linalg import expm
            U_C = expm(-1j * gamma * C)
            psi = U_C @ psi
            
            U_B = expm(-1j * beta * B)
            psi = U_B @ psi
        except Exception:
            return 0.0
            
    # Calculate expectation <psi| C |psi>
    val = np.vdot(psi, C @ psi).real
    return val

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute Hamiltonians once
    try:
        C, B = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Define bounds for DE
    # Gamma is typically in [0, pi], Beta in [0, pi/2] or [0, pi]
    # Let's use [0, pi] for both to be safe, but scale appropriately.
    # Actually, QAOA angles are periodic.
    # Standard ranges: gamma in [0, pi], beta in [0, pi/2]
    # But for p=2, let's search a wider box to ensure global coverage.
    # gamma: [0, pi], beta: [0, pi/2]
    
    bounds = []
    for _ in range(p):
        bounds.append((0.0, np.pi))
        bounds.append((0.0, np.pi/2))
    
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9
        x = np.array(x).reshape(-1, 2)
        val = _expected_cut_value(n, edges, x, C, B)
        return -val
        
    # Run Differential Evolution
    # popsize: 15 (default is 15 * len(bounds) = 15*4=60 for p=2, which is good)
    # maxiter: 100
    # tol: 1e-6
    # seed: 42 for reproducibility
    
    best_angles = np.tile([0.35, 0.25], (p, 1))
    best_val = _expected_cut_value(n, edges, best_angles, C, B)
    
    try:
        # DE is stochastic, so we rely on its global exploration capability.
        # We set 'polish' to False to avoid local refinement which might get stuck,
        # or True to refine the final result. Let's try polish=True for a final boost.
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            polish=True,
            init='latinhypercube'
        )
        
        if res.fun < -best_val:
            best_angles = res.x.reshape(p, 2)
            best_val = -res.fun
            
    except Exception:
        pass
    
    # If time allows, try a second DE run with different seed and tighter bounds?
    # Or just a few random restarts with local optimization?
    # DE should be robust enough.
    
    return best_angles

# EVOLVE-BLOCK-END
