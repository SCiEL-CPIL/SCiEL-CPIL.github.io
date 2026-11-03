import numpy as np

def find_strassen_decomp():
    """Find a rank-7 decomposition of the all-ones 2x2x2 tensor."""
    # We need U, V, W each 4x7
    # Constraint: for all (i,j,k): sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    # Strategy: use the known Strassen structure but fix the W factors.
    # The 7 Strassen products are:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # The U factors (coefficients of a_ij in each M):
    U = np.array([
        [1,  0,  1,  0,  1, -1,  0],   # a00
        [0,  0,  0,  0,  1,  0,  1],    # a01
        [0,  1,  0,  0,  0,  1,  0],    # a10
        [1,  1,  0,  1,  0,  0, -1],    # a11
    ])
    
    # The V factors (coefficients of b_jk in each M):
    V = np.array([
        [1,  1,  0, -1,  0,  1,  0],    # b00
        [0,  0,  1,  0,  0,  1,  0],    # b01
        [0,  0,  0,  1,  0,  0, 1],     # b10
        [1,  0, -1,  0,  1,  0, 1],     # b11
    ])
    
    # Now solve for W. For each (i,j,k):
    # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    # This is linear in W. Let's set up the linear system.
    
    # Unknowns: W[0..3, 0..6] = 28 unknowns
    # Equations: 8 (one for each (i,j,k))
    # Underdetermined, so we can add constraints to get a unique solution.
    
    # Let's use least squares with additional constraints (e.g., minimize ||W||^2)
    # Or just find any solution.
    
    # Build the linear system: A * w = b
    # where w is the flattened W (28 entries)
    # For each (i,j,k): sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    num_eq = 8
    num_unk = 28
    A = np.zeros((num_eq, num_unk))
    b = np.ones(num_eq)
    
    eq_idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    # W[k*2+i, r] is at position (k*2+i)*7 + r in flattened W
                    pos = (k*2+i)*7 + r
                    A[eq_idx, pos] = U[i*2+j, r] * V[j*2+k, r]
                eq_idx += 1
    
    # Solve least squares
    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    W = w.reshape(4, 7)
    
    # Verify
    max_err = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(7))
                max_err = max(max_err, abs(s - 1))
    
    if max_err < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    else:
        return None
