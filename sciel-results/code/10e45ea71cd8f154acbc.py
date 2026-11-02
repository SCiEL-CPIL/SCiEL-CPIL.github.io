"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eig

def get_N(n):
    return np.sqrt(np.pi) * (2**n) * math.factorial(n)

# We need math imported
import math

N0 = get_N(0)
N4 = get_N(4)
N8 = get_N(8)
N12 = get_N(12)

N_mat = np.diag([N0, N4, N8, N12])

# M2 is diagonal for indices 0, 4, 8, 12 because no i +/- 2 is in the set
M2 = np.zeros((4, 4))
indices = [0, 4, 8, 12]
for i_idx, i in enumerate(indices):
    M2[i_idx, i_idx] = (i + 0.5) * N_mat[i_idx, i_idx]

# M11 is also diagonal and identical to M2 for these indices
# We can just use M2 for both, or calculate M11 to be safe if logic changes
# But based on derivation, they are identical.
# Let's construct B and E using A

A = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0/665280.0, -12.0/665280.0, -1680.0/665280.0]
])

B = A.T @ M2 @ A
E = A.T @ N_mat @ A

# Solve generalized eigenvalue problem B v = lambda E v
# We want the smallest lambda
vals, vecs = eig(B, E)

# Filter for real eigenvalues (should be all real since matrices are symmetric positive definite/semi-definite)
real_vals = vals[np.isreal(vals)]
if len(real_vals) == 0:
    # Fallback if something goes wrong
    best_c = np.array([1.0, 0.0, 0.0])
else:
    min_val = np.min(real_vals)
    min_idx = np.argmin(real_vals)
    # Get the corresponding eigenvector
    # The eigenvector might be complex, take real part
    best_c = np.real(vecs[:, min_idx])
    # Normalize? The problem is scale invariant for the ratio, 
    # but construct() returns coefficients. 
    # The score depends on the ratio, so scaling doesn't matter for the score value, 
    # but it might matter for numerical stability or if the evaluator checks norms.
    # The task says "score is the upper bound...". 
    # Usually, we return the vector that achieves the minimum.
    # Let's normalize to unit Euclidean norm for consistency, or keep as is.
    # The current program returns `best_c` from optimization which is not normalized.
    # I will return the eigenvector as is, or normalized. 
    # Let's normalize to have norm 1 to be safe.
    best_c = best_c / np.linalg.norm(best_c)

def construct():
    return best_c

# EVOLVE-BLOCK-END
