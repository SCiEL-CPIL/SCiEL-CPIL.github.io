"""Symbolic regression with Chebyshev polynomial basis and adaptive architecture."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover unknown function f(x,y) using a rich polynomial basis.
    
    Uses Chebyshev polynomials in x and y up to degree 5, plus cross terms,
    giving a total of 42 basis functions. The 10 params are used to weight
    selected groups of basis functions, creating a flexible model that
    BFGS can fit efficiently.
    
    Strategy: Use the 10 params to scale different groups of basis functions,
    allowing the model to adaptively emphasize different features of the target.
    """
    start_time = time.time()
    
    # Build Chebyshev basis up to degree 5
    def chebyshev(t, n):
        """Compute Chebyshev polynomial of degree n at points t."""
        if n == 0:
            return np.ones_like(t)
        elif n == 1:
            return t.copy()
        else:
            T0 = np.ones_like(t)
            T1 = t.copy()
            for k in range(2, n + 1):
                T2 = 2 * t * T1 - T0
                T0 = T1
                T1 = T2
            return T1
    
    # Precompute basis functions
    n_points = len(x)
    
    # Pure x and y terms up to degree 5
    basis = []
    for d in range(6):
        basis.append(chebyshev(x, d))
        if d > 0:
            basis.append(chebyshev(y, d))
    
    # Cross terms x^i * y^j for i+j <= 4
    for i in range(5):
        for j in range(5 - i):
            if i + j > 0:  # skip constant (already have it)
                basis.append(chebyshev(x, i) * chebyshev(y, j))
    
    # Add some nonlinear transformations
    basis.append(np.sin(np.pi * x))
    basis.append(np.sin(np.pi * y))
    basis.append(np.cos(np.pi * x))
    basis.append(np.cos(np.pi * y))
    basis.append(x * y)
    basis.append(x**2 * y)
    basis.append(x * y**2)
    basis.append(x**2)
    basis.append(y**2)
    basis.append(x**3)
    basis.append(y**3)
    
    # Stack into matrix: shape (n_points, n_basis)
    B = np.column_stack(basis)
    n_basis = B.shape[1]
    
    # Map 10 params to weights for basis functions
    # Use a structured mapping: each param controls a group of basis functions
    if n_basis <= 10:
        weights = params
    else:
        # Distribute params across basis functions using a fixed pattern
        # Each param scales a subset of basis functions
        weights = np.zeros(n_basis)
        for i in range(10):
            # Param i affects basis functions at indices i, i+10, i+20, ...
            for j in range(i, n_basis, 10):
                weights[j] = params[i]
        
        # Also allow some params to affect multiple groups for flexibility
        # Params 0-4 affect even-indexed basis, params 5-9 affect odd-indexed
        for i in range(5):
            for j in range(i * 2, n_basis, 10):
                weights[j] += 0.5 * params[i]
            for j in range(i * 2 + 1, n_basis, 10):
                weights[j] += 0.5 * params[i + 5]
    
    # Compute prediction
    pred = B @ weights
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Return best result so far (which is just the current prediction)
        return pred
    
    return pred
# EVOLVE-BLOCK-END
