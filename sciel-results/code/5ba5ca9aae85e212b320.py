"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def get_N(n):
    return np.sqrt(np.pi) * (2**n) * math.factorial(n)

N0 = get_N(0)
N4 = get_N(4)
N8 = get_N(8)
N12 = get_N(12)

N_mat = np.diag([N0, N4, N8, N12])

M2 = np.zeros((4, 4))
indices = [0, 4, 8, 12]
for i_idx, i in enumerate(indices):
    M2[i_idx, i_idx] = (i + 0.5) * N_mat[i_idx, i_idx]
    
    if i + 2 in indices:
        j_idx = indices.index(i + 2)
        M2[i_idx, j_idx] = 0.25 * N_mat[j_idx, j_idx]
        M2[j_idx, i_idx] = M2[i_idx, j_idx]
        
    if i - 2 in indices:
        j_idx = indices.index(i - 2)
        M2[i_idx, j_idx] = (i * (i - 1) / 2.0) * N_mat[j_idx, j_idx]
        M2[j_idx, i_idx] = M2[i_idx, j_idx]

M11 = np.zeros((4, 4))
for i_idx, i in enumerate(indices):
    M11[i_idx, i_idx] = (i**2 * get_N(i-1) if i > 0 else 0) + 0.25 * get_N(i+1)
    
    if i - 2 in indices:
        j_idx = indices.index(i - 2)
        M11[i_idx, j_idx] = -0.5 * i * get_N(i-1)
        M11[j_idx, i_idx] = M11[i_idx, j_idx]
        
    if i + 2 in indices:
        j_idx = indices.index(i + 2)
        M11[i_idx, j_idx] = -0.5 * (i+2) * get_N(i+1)
        M11[j_idx, i_idx] = M11[i_idx, j_idx]

A = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0/665280.0, -12.0/665280.0, -1680.0/665280.0]
])

B = A.T @ M2 @ A
D = A.T @ M11 @ A
E = A.T @ N_mat @ A

# Precompute Cholesky or eigen decomposition of E for normalization
# We want to minimize (c^T B c)(c^T D c) subject to c^T E c = 1
# This is equivalent to finding the minimum of the product of two Rayleigh quotients.

def objective_constrained(c):
    num1 = c @ B @ c
    num2 = c @ D @ c
    return num1 * num2

# Constraint: c^T E c - 1 = 0
cons = ({'type': 'eq', 'fun': lambda c: c @ E @ c - 1.0, 'jac': lambda c: 2.0 * E @ c})

best_val = 1e10
best_c = np.array([1.0, 0.0, 0.0])

rng = np.random.default_rng(42)

# Generate random starting points on the ellipsoid defined by E
# We can sample from standard normal and normalize by E
for _ in range(200):
    c0 = rng.normal(size=3)
    # Normalize c0 such that c0^T E c0 = 1
    norm_val = np.sqrt(c0 @ E @ c0)
    if norm_val < 1e-12:
        continue
    c0 = c0 / norm_val
    
    # Use SLSQP which handles constraints well
    res = minimize(objective_constrained, c0, method='SLSQP', constraints=cons, options={'ftol': 1e-15, 'maxiter': 1000})
    
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_c = res.x

def construct():
    return best_c

# EVOLVE-BLOCK-END
