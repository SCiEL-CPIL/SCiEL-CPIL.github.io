"""
Strategy: Exact state-vector simulation with adaptive multi-start Nelder-Mead.
Instead of BFGS with potentially inaccurate numerical gradients, we use Nelder-Mead
which is derivative-free and robust for low-dimensional non-convex problems.
We perform multiple restarts with diverse initializations (including graph-aware
heuristics) to escape local minima. For n<=10, exact state-vector simulation is
fast enough to allow many evaluations.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    # Helper to apply operator on qubit k
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
    
    # Build C
    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    # Build B
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
        # For small matrices, use scipy.linalg.expm
        try:
            from scipy.linalg import expm
            U_C = expm(-1j * gamma * C)
            psi = U_C @ psi
            
            # Apply exp(-i beta B)
            U_B = expm(-1j * beta * B)
            psi = U_B @ psi
        except Exception:
            return 0.0
            
    # Calculate expectation <psi| C |psi>
    val = np.vdot(psi, C @ psi).real
    return val

def _graph_features(n, edges):
    """Extract simple graph features for initialization."""
    degrees = np.zeros(n)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    avg_deg = np.mean(degrees) if n > 0 else 0.0
    max_deg = np.max(degrees) if n > 0 else 0.0
    min_deg = np.min(degrees) if n > 0 else 0.0
    density = len(edges) / (n * (n-1) / 2) if n > 1 else 0.0
    
    return avg_deg, max_deg, min_deg, density

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute Hamiltonians once
    try:
        C, B = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Get graph features
    avg_deg, max_deg, min_deg, density = _graph_features(n, edges)
    
    # Generate diverse initializations
    candidates = []
    
    # Candidate 1: Standard fixed
    candidates.append(np.tile([0.35, 0.25], (p, 1)))
    
    # Candidate 2: Based on density and degree
    if density > 0:
        g1 = 0.5 / max(1.0, max_deg)
        b1 = 0.3 / max(1.0, max_deg)
        candidates.append(np.tile([g1, b1], (p, 1)))
    
    # Candidate 3: Smaller angles for sparse graphs
    if density < 0.4:
        candidates.append(np.tile([0.2, 0.15], (p, 1)))
    
    # Candidate 4: Larger angles for dense graphs
    if density > 0.4:
        candidates.append(np.tile([0.6, 0.4], (p, 1)))
    
    # Add random starts
    np.random.seed(123)
    for _ in range(8):
        gams = np.random.uniform(0.05, 1.2, p)
        bets = np.random.uniform(0.05, 0.6, p)
        candidates.append(np.column_stack((gams, bets)))
    
    best_val = -1.0
    best_angles = candidates[0]
    
    # Evaluate candidates quickly
    for cand in candidates:
        val = _expected_cut_value(n, edges, cand, C, B)
        if val > best_val:
            best_val = val
            best_angles = cand
        if _time_remaining() < 20.0:
            break
            
    # Define objective function for Nelder-Mead
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9
        x = np.array(x).reshape(-1, 2)
        val = _expected_cut_value(n, edges, x, C, B)
        return -val
    
    # Run Nelder-Mead from best candidate
    x0 = best_angles.flatten()
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
        if res.fun < -best_val:
            best_angles = res.x.reshape(p, 2)
            best_val = -res.fun
    except Exception:
        pass
    
    # If time allows, try a few more restarts from different points
    if _time_remaining() > 30.0:
        np.random.seed(456)
        for _ in range(3):
            if _time_remaining() < 20.0:
                break
            gams = np.random.uniform(0.1, 0.8, p)
            bets = np.random.uniform(0.1, 0.4, p)
            x0 = np.column_stack((gams, bets)).flatten()
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
                if res.fun < -best_val:
                    best_angles = res.x.reshape(p, 2)
                    best_val = -res.fun
            except Exception:
                pass
    
    return best_angles

# EVOLVE-BLOCK-END
