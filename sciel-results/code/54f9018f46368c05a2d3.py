# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
from itertools import combinations
import json
import os
import hashlib
import tempfile

def _exhaustive_optimum(n, edges):
    """Compute exact MaxCut optimum via brute force."""
    if not edges:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for i, j in edges:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _expected_cut_exact(n, edges, gammas, betas, p):
    """
    Compute expected cut exactly using statevector simulation.
    gammas, betas: arrays of length p
    p: number of layers
    """
    # Initial state: |+>^n = (1/sqrt(2))^n * sum over all bitstrings
    # Use statevector approach
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i * gamma * C) where C is the MaxCut Hamiltonian
        # C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
        # Actually for QAOA, C is typically the cost Hamiltonian which for MaxCut is:
        # H_C = sum_{(i,j) in edges} (1 - Z_i Z_j)/2
        # But the phase kickback is exp(-i * gamma * H_C)
        # Let's compute the phase for each basis state
        
        # For each basis state, compute the cut value
        phases = np.ones(dim, dtype=np.complex128)
        for idx in range(dim):
            cut_val = 0
            for i, j in edges:
                bit_i = (idx >> i) & 1
                bit_j = (idx >> j) & 1
                if bit_i != bit_j:
                    cut_val += 1
            # H_C eigenvalue for this state is cut_val (since H_C = sum (1-ZZ)/2, 
            # but actually for MaxCut the cost is the number of cut edges)
            # Wait, let me reconsider. The standard QAOA MaxCut Hamiltonian is:
            # H_C = sum_{(i,j) in E} (I - Z_i Z_j) / 2
            # For a given bitstring, Z_i Z_j = (+1) if same bit, (-1) if different
            # So (I - ZZ)/2 = 0 if same, 1 if different
            # So H_C eigenvalue = number of cut edges
            phases[idx] = np.exp(-1j * gamma * cut_val)
        
        state = state * phases
        
        # Apply exp(-i * beta * sum X_k)
        # This is a product of single-qubit rotations: (cos(beta) I - i sin(beta) X)^n
        # For each qubit, apply rotation
        # X rotation: |0> -> cos(beta)|0> - i sin(beta)|1>
        #              |1> -> -i sin(beta)|0> + cos(beta)|1>
        
        for k in range(n):
            # Build the rotation matrix for qubit k
            # We need to apply R_x(beta) to qubit k
            # The rotation is: [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
            
            c = np.cos(beta)
            s = np.sin(beta)
            rot = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
            
            # Reshape state to apply rotation on qubit k
            # Qubit k is bit position k (0-indexed, LSB is qubit 0)
            # We need to group indices by the value of bit k
            
            new_state = np.zeros_like(state)
            for idx in range(dim):
                bit_k = (idx >> k) & 1
                if bit_k == 0:
                    # |0> -> c|0> - i s|1>
                    idx_0 = idx  # bit k = 0
                    idx_1 = idx | (1 << k)  # bit k = 1
                    new_state[idx_0] += c * state[idx_0]
                    new_state[idx_1] += (-1j * s) * state[idx_0]
                # We'll handle the |1> case when we encounter it
            # Actually, let's do it more carefully
            new_state = np.zeros_like(state)
            for idx in range(dim):
                bit_k = (idx >> k) & 1
                if bit_k == 0:
                    idx_0 = idx
                    idx_1 = idx | (1 << k)
                    # |0> component: contributes to new |0> and new |1>
                    new_state[idx_0] += c * state[idx_0]
                    new_state[idx_1] += (-1j * s) * state[idx_0]
                else:
                    idx_1 = idx
                    idx_0 = idx & ~(1 << k)
                    # |1> component: contributes to new |0> and new |1>
                    new_state[idx_0] += (-1j * s) * state[idx_1]
                    new_state[idx_1] += c * state[idx_1]
            
            state = new_state
    
    # Compute expected cut
    # <Z_i Z_j> for each edge
    expected_cut = 0.0
    for i, j in edges:
        zz_sum = 0.0
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_j = (idx >> j) & 1
            zz_val = 1.0 if bit_i == bit_j else -1.0
            prob = np.abs(state[idx])**2
            zz_sum += zz_val * prob
        # (1 - <ZZ>)/2
        expected_cut += (1 - zz_sum) / 2.0
    
    return expected_cut

def _ratio_objective(params, n, edges, p, optimum):
    """Objective to minimize: -ratio"""
    if len(params) != 4:
        return 1e6
    gammas = params[:2]
    betas = params[2:]
    # Clamp to reasonable ranges
    gammas = np.clip(gammas, 0.0, np.pi)
    betas = np.clip(betas, 0.0, np.pi)
    try:
        exp_cut = _expected_cut_exact(n, edges, gammas, betas, p)
        if optimum == 0:
            return 0.0
        ratio = exp_cut / optimum
        return -ratio
    except Exception:
        return 1e6

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of nodes
        edges: list of (i, j) edges
        p: number of layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Compute exact optimum
    optimum = _exhaustive_optimum(n, edges)
    if optimum == 0:
        return np.array([[0.0, 0.0], [0.0, 0.0]])
    
    # Use Nelder-Mead to optimize the 4 parameters
    from scipy.optimize import minimize
    
    # Initialize with a reasonable guess
    # Based on known good QAOA angles for small graphs
    x0 = np.array([0.35, 0.25, 0.35, 0.25])
    
    result = minimize(
        _ratio_objective,
        x0,
        args=(n, edges, p, optimum),
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
    )
    
    best_params = result.x
    gammas = np.clip(best_params[:2], 0.0, np.pi)
    betas = np.clip(best_params[2:], 0.0, np.pi)
    
    return np.array([[gammas[0], betas[0]], [gammas[1], betas[1]]])
# EVOLVE-BLOCK-END
