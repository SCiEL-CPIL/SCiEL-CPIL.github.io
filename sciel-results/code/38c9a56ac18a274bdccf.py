import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a separable tensor-product basis of Chebyshev polynomials.
    
    We use T_0, T_1, T_2 for each variable, giving 3x3 = 9 basis functions.
    The 10th parameter is used as a global bias term (or we can add T_2(x)*T_2(y)
    with a scaling to make it 10 terms, but 9 is a perfect 3x3 grid).
    Wait, params has length 10.
    
    Let's use a 3x3 grid of Chebyshev products (9 terms) + 1 constant bias?
    No, T_0(x)*T_0(y) = 1*1 = 1, so the constant is already included.
    
    So we have 9 terms:
    T_0(x)T_0(y), T_1(x)T_0(y), T_2(x)T_0(y),
    T_0(x)T_1(y), T_1(x)T_1(y), T_2(x)T_1(y),
    T_0(x)T_2(y), T_1(x)T_2(y), T_2(x)T_2(y)
    
    That's 9 terms. We have 10 params.
    Let's add one more term to make it 10.
    Maybe T_3(x)*T_0(y) or T_0(x)*T_3(y)? Or T_1(x)*T_3(y)?
    Or perhaps use a slightly different set to ensure we use all 10 params meaningfully.
    
    Alternative: Use a 2D polynomial basis of total degree <= 3.
    Degree 0: 1 (1 term)
    Degree 1: x, y (2 terms)
    Degree 2: x^2, xy, y^2 (3 terms)
    Degree 3: x^3, x^2y, xy^2, y^3 (4 terms)
    Total: 1 + 2 + 3 + 4 = 10 terms.
    This matches the 10 parameters perfectly.
    
    However, monomials are ill-conditioned. Chebyshev polynomials are better.
    Let's stick to Chebyshev but select 10 terms from the tensor product space.
    
    Basis functions:
    1. T_0(x)T_0(y) = 1
    2. T_1(x)T_0(y) = x
    3. T_0(x)T_1(y) = y
    4. T_2(x)T_0(y) = 2x^2 - 1
    5. T_1(x)T_1(y) = xy
    6. T_0(x)T_2(y) = 2y^2 - 1
    7. T_3(x)T_0(y) = 4x^3 - 3x
    8. T_2(x)T_1(y) = (2x^2 - 1)y
    9. T_1(x)T_2(y) = x(2y^2 - 1)
    10. T_0(x)T_3(y) = 4y^3 - 3y
    
    This set includes all terms of total degree <= 3, except for the mixed cubic terms
    x^2y and xy^2?
    Let's check:
    T_2(x)T_1(y) = (2x^2 - 1)y = 2x^2y - y. Contains x^2y.
    T_1(x)T_2(y) = x(2y^2 - 1) = 2xy^2 - x. Contains xy^2.
    
    So the span of these 10 Chebyshev tensor products is the same as the span of
    all polynomials of total degree <= 3.
    
    This is a well-conditioned basis for the space of cubic polynomials in 2 variables.
    """
    # Chebyshev polynomials of first kind
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 basis functions
    # Order matters for params, but any fixed order works.
    # We want to ensure the matrix is well-conditioned.
    
    basis = np.column_stack((
        T0_x * T0_y,  # 1
        T1_x * T0_y,  # x
        T0_x * T1_y,  # y
        T2_x * T0_y,  # 2x^2 - 1
        T1_x * T1_y,  # xy
        T0_x * T2_y,  # 2y^2 - 1
        T3_x * T0_y,  # 4x^3 - 3x
        T2_x * T1_y,  # (2x^2 - 1)y
        T1_x * T2_y,  # x(2y^2 - 1)
        T0_x * T3_y   # 4y^3 - 3y
    ))
    
    # Linear combination
    f = basis @ params
    
    return f
