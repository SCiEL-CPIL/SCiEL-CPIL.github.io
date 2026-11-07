"""Symbolic regression with product-of-sums decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: f(x, y) = (sum_i a_i * basis_i(x)) * (sum_j b_j * basis_j(y))
    where basis_k(t) = t^k for k=0..4.
    
    The 10 params are split: params[0:5] for x-basis, params[5:10] for y-basis.
    This gives a multiplicative interaction model that can capture separable
    and interaction-heavy bivariate targets.
    
    The evaluator uses BFGS to fit params starting from all ones.
    We just need to return the prediction given the current params.
    """
    # Split params
    ax = params[0:5]  # coefficients for x basis
    by = params[5:10]  # coefficients for y basis
    
    # Build basis matrices
    # x_basis: (n, 5) with columns [1, x, x^2, x^3, x^4]
    x_basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4
    ])
    
    # y_basis: (n, 5) with columns [1, y, y^2, y^3, y^4]
    y_basis = np.column_stack([
        np.ones_like(y),
        y,
        y**2,
        y**3,
        y**4
    ])
    
    # f(x) = x_basis @ ax  -> shape (n,)
    fx = x_basis @ ax
    
    # g(y) = y_basis @ by  -> shape (n,)
    gy = y_basis @ by
    
    # Product
    pred = fx * gy
    
    return pred
# EVOLVE-BLOCK-END
