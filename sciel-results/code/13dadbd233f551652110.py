import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Orthonormal Fourier basis on [-1, 1] x [-1, 1]
    # Frequency pi ensures period 2, matching the domain width.
    
    # Pre-compute trig terms
    sin_x = np.sin(np.pi * x)
    cos_x = np.cos(np.pi * x)
    sin_y = np.sin(np.pi * y)
    cos_y = np.cos(np.pi * y)
    
    # 9 terms from tensor product of {1, sin, cos} for x and y
    # Note: These are orthonormal.
    term_1 = np.ones_like(x)
    term_2 = sin_x
    term_3 = cos_x
    term_4 = sin_y
    term_5 = cos_y
    term_6 = sin_x * sin_y
    term_7 = sin_x * cos_y
    term_8 = cos_x * sin_y
    term_9 = cos_x * cos_y
    
    # 10th term: Higher frequency to capture more detail
    # sin(2*pi*x) is orthogonal to the first 9?
    # sin(2*pi*x) is orthogonal to 1, sin(pi*x), cos(pi*x), etc?
    # Integral of sin(2*pi*x) * 1 = 0.
    # Integral of sin(2*pi*x) * sin(pi*x) = 0.
    # Integral of sin(2*pi*x) * cos(pi*x) = 0.
    # Yes, it is orthogonal to all terms that are linear in sin(pi*x)/cos(pi*x) or constant in x.
    # So term_10 = sin(2*pi*x) is orthogonal to the first 9.
    term_10 = np.sin(2 * np.pi * x)
    
    p = params
    pred = (
        p[0] * term_1 +
        p[1] * term_2 +
        p[2] * term_3 +
        p[3] * term_4 +
        p[4] * term_5 +
        p[5] * term_6 +
        p[6] * term_7 +
        p[7] * term_8 +
        p[8] * term_9 +
        p[9] * term_10
    )
    
    return pred
