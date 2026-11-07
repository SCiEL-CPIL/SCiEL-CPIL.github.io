import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Use a fixed, physics-motivated basis for a nonlinear oscillator (Duffing-like)
    with careful scaling to ensure the optimal parameters are O(1) and close to the initial guess (1.0).
    
    The model is linear in params: y = T @ params.
    The evaluator uses BFGS to fit params. Since the model is linear, BFGS should converge
    if the basis is well-conditioned. The key is to scale the basis terms such that the
    true coefficients are near 1, making the optimization landscape smooth and the initial
    guess (all ones) close to the optimum.
    
    Inputs:
    x: position, range ~ [-0.57, 0.53]
    v: velocity, range ~ [-0.18, 0.28]
    Target y: acceleration or derivative, range ~ [-0.10, 0.11]
    
    Physics:
    A nonlinear oscillator (e.g., Duffing) has acceleration y = -k*x - c*v - a*x^3 - b*v^3 ...
    Since y is small and x is ~0.5, the linear coefficient k is ~0.2.
    To make k ~ 1, we should scale x by ~0.2? No, if we use x/0.5, the coefficient becomes k*0.5 ~ 0.1.
    If we use x/0.1, the coefficient becomes k*0.1 ~ 0.02.
    
    Let's look at the magnitudes.
    y ~ 0.1
    x ~ 0.5
    v ~ 0.2
    
    If we assume y ≈ a*x + b*v:
    a ≈ 0.1/0.5 = 0.2
    b ≈ 0.1/0.2 = 0.5
    
    To make a ≈ 1, we can scale x by 0.2. i.e., use x/0.2 = 5x.
    Then term is 5x, coeff is 0.2. Still not 1.
    
    Actually, the evaluator starts params at 1. If the true params are 0.2, BFGS has to move from 1 to 0.2. That's fine.
    The problem with the previous attempt was likely ill-conditioning or the specific basis choice.
    
    Let's try a basis that is orthogonalized in a simple way or just very standard and well-scaled.
    We want to avoid highly correlated terms like x and x^2 if x is small and centered, but x and x^3 are less correlated.
    
    Let's use:
    1. 1
    2. x
    3. v
    4. x^2
    5. x^3
    6. v^2
    7. v^3
    8. x*v
    9. sin(x)
    10. cos(x)
    
    But we need to be careful with scaling.
    If we use raw x, v, the values are small.
    Let's scale x by 2 (since max is ~0.6, 2*0.6=1.2) and v by 5 (since max is ~0.3, 5*0.3=1.5).
    This keeps the basis terms in O(1) range.
    
    Let X = 2*x
    Let V = 5*v
    
    Basis:
    1
    X
    V
    X^2
    X^3
    V^2
    V^3
    X*V
    sin(X)
    cos(X)
    
    This should be well-conditioned.
    """
    # Scale inputs to be roughly in [-1, 1] or [-2, 2] to avoid numerical issues and keep coeffs O(1)
    X = 2.0 * x
    V = 5.0 * v
    
    # Construct basis matrix
    # We use a mix of polynomial and trigonometric terms to capture nonlinear dynamics
    terms = np.column_stack([
        np.ones_like(x),  # 0: Constant
        X,                # 1: Linear X
        V,                # 2: Linear V
        X**2,             # 3: Quadratic X
        X**3,             # 4: Cubic X
        V**2,             # 5: Quadratic V
        V**3,             # 6: Cubic V
        X * V,            # 7: Cross term
        np.sin(X),        # 8: Sine X
        np.cos(X) - 1.0   # 9: Cosine X (centered to avoid correlation with constant)
    ])
    
    return terms @ params
