# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: a full quadratic polynomial basis (10 terms) combined with
    a multiplicative correction term.
    
    The quadratic basis spans all degree-2 polynomials in x, y, z:
    {1, x, y, z, x^2, y^2, z^2, xy, xz, yz}
    
    This is a linear model in params, so BFGS should converge quickly to the
    global minimum for this sub-model.
    
    Additionally, we add a small multiplicative correction:
    correction = p8 * (p9*x + p9*y + p9*z + p9) * (p9*x + p9*y + p9*z + p9)
    Wait, we only have 10 params total. The quadratic basis already uses all 10.
    
    Let's re-evaluate. The current program uses a product of two linear terms:
    (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7)
    This expands to a quadratic form but with a specific structure (rank 1 quadratic part).
    The full quadratic basis is more general.
    
    However, the inspiration 1 & 2 show that a simple linear combination of the
    standard quadratic basis is a strong baseline (score ~7.39).
    
    The current program (score 7.38) is slightly better, likely due to the
    specific structure or the linear correction terms p8, p9.
    
    Let's try a different structural change:
    Use a basis that includes rational terms, as hinted by the "reciprocal/interaction
    basis" finding in the knowledge section (score 9.96 vs 6.38, though those scores
    seem inverted or refer to a different scale/metric, but the hint suggests rational
    structure might be key).
    
    Actually, the knowledge says: "Switching from a mismatched RBF model to a correctly
    sized reciprocal/interaction basis aligns with the known superior performance of
    this feature set (score 9.96 vs 6.38) for the hidden function's likely rational structure."
    Lower is better, so 6.38 is better than 9.96. The "reciprocal/interaction basis"
    scored 6.38.
    
    Let's try a basis with reciprocal terms.
    Basis:
    1, x, y, z, 1/x, 1/y, 1/z, x/y, y/z, x/z
    
    This covers linear, inverse, and ratio structures.
    
    Let's construct this basis.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
        
    # Avoid division by zero (inputs are in [0.5, 2], so safe, but good practice)
    x_safe = np.where(np.abs(x) < 1e-8, 1e-8, x)
    y_safe = np.where(np.abs(y) < 1e-8, 1e-8, y)
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = 1.0 / x_safe
    b5 = 1.0 / y_safe
    b6 = 1.0 / z_safe
    b7 = x / y_safe
    b8 = y / z_safe
    b9 = x / z_safe
    
    # Linear combination
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
