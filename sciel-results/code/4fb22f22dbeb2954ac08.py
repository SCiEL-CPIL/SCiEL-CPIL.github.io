# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a rich basis of candidate functions.
    We use a linear-in-parameters model: f(x,y) = sum_i p_i * phi_i(x,y)
    where phi_i are precomputed basis functions.
    
    Basis functions include:
    - Constants
    - Linear terms
    - Polynomial terms up to degree 3
    - Trigonometric terms
    - Radial basis functions
    - Interaction terms
    """
    n = len(x)
    
    # Precompute basis functions
    # We'll use a mix of polynomial, trigonometric, and RBF basis
    
    # Polynomial basis (up to degree 3)
    # x^i * y^j for i+j <= 3
    poly_terms = []
    for i in range(4):
        for j in range(4 - i):
            poly_terms.append((i, j))
    
    # Build the basis matrix
    basis = np.ones((n, 1))  # constant term
    
    # Add polynomial terms
    for (i, j) in poly_terms:
        if i == 0 and j == 0:
            continue  # already have constant
        term = np.ones(n)
        if i > 0:
            term = term * x ** i
        if j > 0:
            term = term * y ** j
        basis = np.column_stack([basis, term])
    
    # Add trigonometric terms
    trig_freqs = [1, 2, 3]
    for freq in trig_freqs:
        basis = np.column_stack([
            basis,
            np.sin(freq * x),
            np.cos(freq * x),
            np.sin(freq * y),
            np.cos(freq * y),
            np.sin(freq * (x + y)),
            np.cos(freq * (x + y)),
            np.sin(freq * (x - y)),
            np.cos(freq * (x - y)),
        ])
    
    # Add radial basis functions
    # Use a few centers with different widths
    centers = [
        (0, 0),
        (0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
    ]
    widths = [0.5, 1.0, 2.0]
    
    for cx, cy in centers:
        for w in widths:
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            rbf = np.exp(-r2 / (2 * w ** 2))
            basis = np.column_stack([basis, rbf])
    
    # Add interaction terms
    basis = np.column_stack([
        basis,
        x * y,
        x * y * (x + y),
        x * y * (x - y),
        x ** 2 * y,
        x * y ** 2,
    ])
    
    # Now we have a basis matrix with many columns.
    # The params array has length 10, so we need to map 10 params to these basis functions.
    # We'll use a fixed linear combination of the basis functions determined by the 10 params.
    
    # Strategy: use the first 10 basis functions, or a weighted combination.
    # Actually, let's use a different approach: define 10 specific composite basis functions
    # that are linear in the 10 parameters.
    
    # Let's redefine: we'll create 10 basis functions phi_0 through phi_9
    # and compute f = sum_{k=0}^{9} params[k] * phi_k(x,y)
    
    # phi_0: constant
    phi0 = np.ones(n)
    
    # phi_1: x
    phi1 = x.copy()
    
    # phi_2: y
    phi2 = y.copy()
    
    # phi_3: x^2
    phi3 = x ** 2
    
    # phi_4: y^2
    phi4 = y ** 2
    
    # phi_5: x*y
    phi5 = x * y
    
    # phi_6: sin(x)
    phi6 = np.sin(x)
    
    # phi_7: sin(y)
    phi7 = np.sin(y)
    
    # phi_8: cos(x)
    phi8 = np.cos(x)
    
    # phi_9: cos(y)
    phi9 = np.cos(y)
    
    # Build the final prediction as a linear combination
    prediction = (
        params[0] * phi0 +
        params[1] * phi1 +
        params[2] * phi2 +
        params[3] * phi3 +
        params[4] * phi4 +
        params[5] * phi5 +
        params[6] * phi6 +
        params[7] * phi7 +
        params[8] * phi8 +
        params[9] * phi9
    )
    
    return prediction
# EVOLVE-BLOCK-END
