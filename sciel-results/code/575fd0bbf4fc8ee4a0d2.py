"""
Strategy: Spectral Graph Theory Initialization + Lightweight Nelder-Mead Refinement.
Instead of relying on random multi-starts and expensive gradient-based L-BFGS-B,
this strategy uses the graph's Laplacian eigenvalues and degree distribution to
generate highly informed initial guesses for the QAOA angles. 
The spectral gap and average degree provide strong priors for the optimal 
gamma (problem strength) and beta (mixing strength). 
We then perform a short, robust Nelder-Mead refinement from these spectral points.
This avoids the local minima traps of random starts and the computational overhead
of exact analytical gradients, while leveraging the strong structure of the 
MaxCut problem on small graphs.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys
from scipy.linalg import expm

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
    Compute the expected cut value.
    """
    if len(edges) == 0:
        return 0.0
        
    dim = 2**n
    p = len(angles)
    
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Forward pass
    for i in range(p):
        gamma, beta = angles[i]
        
        U_C = expm(-1j * gamma * C)
        psi = U_C @ psi
        
        U_B = expm(-1j * beta * B)
        psi = U_B @ psi
        
    # Value: <psi| C |psi>
    val = np.vdot(psi, C @ psi).real
    return val

def _spectral_init(n, edges):
    """
    Generate diverse, graph-aware initializations using spectral properties.
    """
    # Build adjacency matrix
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    # Degrees
    degrees = A.sum(axis=1)
    avg_deg = np.mean(degrees)
    max_deg = np.max(degrees) if n > 0 else 1.0
    
    # Laplacian
    L = np.diag(degrees) - A
    
    # Eigenvalues of Laplacian
    eigvals = np.linalg.eigvalsh(L)
    
    # Spectral gap (second smallest eigenvalue)
    # For disconnected graphs, this is 0. We handle that.
    if len(eigvals) > 1:
        gap = eigvals[1]
    else:
        gap = 1.0
        
    # If graph is disconnected, gap is 0. Use a small epsilon to avoid division by zero
    if gap < 1e-6:
        gap = 0.1
        
    # Theoretical scaling:
    # Gamma is related to the strength of the problem Hamiltonian.
    # For MaxCut, the optimal gamma is often around pi/4 / (max_deg) or similar.
    # Beta is related to the mixing.
    
    # Candidate 1: Based on spectral gap and avg degree
    # Heuristic: gamma ~ pi / (4 * avg_deg) if dense, larger if sparse
    g1 = np.pi / (4.0 * max(1.0, avg_deg))
    b1 = np.pi / (4.0 * n) # Beta scales with system size
    
    # Candidate 2: Based on spectral gap
    # Larger gap -> stronger problem -> potentially larger gamma?
    g2 = np.pi / (2.0 * gap)
    b2 = np.pi / (4.0 * gap)
    
    # Candidate 3: Standard fixed (baseline)
    g3 = 0.35
    b3 = 0.25
    
    # Candidate 4: Sparse graph specific
    g4 = 0.2
    b4 = 0.15
    
    # Candidate 5: Dense graph specific
    g5 = 0.6
    b5 = 0.4
    
    # Construct 2-layer angles
    # Layer 1 and Layer 2 often have similar or slightly different angles.
    # For p=2, we can try (g, b, g, b) or (g1, b1, g2, b2)
    
    inits = []
    
    # Uniform layers
    inits.append(np.array([[g1, b1], [g1, b1]]))
    inits.append(np.array([[g2, b2], [g2, b2]]))
    inits.append(np.array([[g3, b3], [g3, b3]]))
    inits.append(np.array([[g4, b4], [g4, b4]]))
    inits.append(np.array([[g5, b5], [g5, b5]]))
    
    # Mixed layers (Layer 1 more problem, Layer 2 more mixing)
    inits.append(np.array([[g1, b1], [g1*0.5, b1*1.5]]))
    inits.append(np.array([[g2, b2], [g2*0.5, b2*1.5]]))
    
    # Random perturbations around spectral guesses
    np.random.seed(42)
    for _ in range(10):
        g_pert = np.random.normal(g1, 0.1)
        b_pert = np.random.normal(b1, 0.05)
        inits.append(np.array([[g_pert, b_pert], [g_pert, b_pert]]))
        
    return inits

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute Hamiltonians once
    try:
        C, B = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Generate spectral initializations
    inits = _spectral_init(n, edges)
    
    best_val = -1.0
    best_angles = inits[0]
    
    # Objective function for Nelder-Mead
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9
        x = np.array(x).reshape(-1, 2)
        val = _expected_cut_value(n, edges, x, C, B)
        return -val

    # Evaluate all inits first
    for init in inits:
        val = _expected_cut_value(n, edges, init, C, B)
        if val > best_val:
            best_val = val
            best_angles = init
        if _time_remaining() < 20.0:
            break
            
    # Refine the top few candidates using Nelder-Mead
    # Sort inits by value to prioritize refinement
    init_vals = []
    for init in inits:
        val = _expected_cut_value(n, edges, init, C, B)
        init_vals.append((val, init))
    init_vals.sort(key=lambda x: x[0], reverse=True)
    
    # Try refining the top 5 distinct initializations
    refined_count = 0
    for val, init in init_vals:
        if _time_remaining() < 20.0:
            break
        if refined_count >= 5:
            break
            
        x0 = init.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < -best_val:
                best_angles = res.x.reshape(p, 2)
                best_val = -res.fun
        except Exception:
            pass
        refined_count += 1
        
    return best_angles

# EVOLVE-BLOCK-END
