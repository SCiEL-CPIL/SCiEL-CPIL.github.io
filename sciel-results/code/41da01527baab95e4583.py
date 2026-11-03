"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.linalg import eigh

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

# Scaling factors S_n = sqrt(2^n n! sqrt(pi))
# such that <H_n/S_n, H_m/S_m> = delta_nm
S = np.array([sqrt((2**n) * factorial(n) * sqrt(pi)) for n in indices])

# H_n(0) values
H0_0 = _hermite_val(0, 0)   # 1
H4_0 = _hermite_val(4, 0)   # 12
H8_0 = _hermite_val(8, 0)   # 168
H12_0 = _hermite_val(12, 0) # 20901120

# h_n = H_n(0) / S_n
h = np.array([H0_0/S[0], H4_0/S[1], H8_0/S[2], H12_0/S[3]])

# We want to minimize <x^2> = sum d_n^2 (n + 0.5) / sum d_n^2
# subject to sum d_n h_n = 0
# d_12 = - (h_0 d_0 + h_4 d_4 + h_8 d_8) / h_12
k0 = -h[0] / h[3]
k4 = -h[1] / h[3]
k8 = -h[2] / h[3]

# Q matrix mapping [d0, d4, d8] to [d0, d4, d8, d12]
Q = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [k0,  k4,  k8]
])

# W matrix: diagonal with entries (n + 0.5)
W = np.diag(np.array([n + 0.5 for n in indices]))

# A = Q^T W Q
# B = Q^T Q
A = Q.T @ W @ Q
B = Q.T @ Q

def construct():
    """Find optimal coefficients for H0, H4, H8 using scaled generalized eigenvalue problem."""
    start_time = time.time()
    deadline = start_time + 150.0
    
    try:
        # Solve A v = lambda B v
        # Smallest eigenvalue corresponds to minimum variance
        eigvals, eigvecs = eigh(A, B)
        idx = np.argmin(eigvals)
        best_v = eigvecs[:, idx]
        
        # Normalize v so that max component is 1 for stability, though not strictly necessary
        max_comp = np.max(np.abs(best_v))
        if max_comp > 1e-15:
            best_v = best_v / max_comp
            # Make first non-zero component positive
            for i in range(3):
                if abs(best_v[i]) > 1e-10:
                    if best_v[i] < 0:
                        best_v = -best_v
                    break
        
        # Recover d vector
        d = Q @ best_v
        
        # Convert back to c coefficients
        c = d / S
        
        # Return c0, c4, c8
        return c[:3]
        
    except Exception:
        # Fallback to a simple heuristic if solver fails
        # Try to minimize by putting weight on lower n
        # But must satisfy constraint.
        # If d12 is large, we need to balance.
        # Simple fallback: c0=1, c4=0, c8=0 => c12 = -1/H12(0)
        # This is a valid solution.
        return np.array([1.0, 0.0, 0.0])

# EVOLVE-BLOCK-END
