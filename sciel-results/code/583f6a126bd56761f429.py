import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    # t = (2*x - (0.5 + 2)) / (2 - 0.5) = (2*x - 2.5) / 1.5
    tx = (2.0 * x - 2.5) / 1.5
    ty = (2.0 * y - 2.5) / 1.5
    tz = (2.0 * z - 2.5) / 1.5
    
    # Build a richer basis using Chebyshev polynomials up to degree 3
    # and include key interaction terms.
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    # Compute Chebyshev polynomials for each variable
    T0_x = np.ones_like(tx)
    T1_x = tx
    T2_x = 2.0 * tx**2 - 1.0
    T3_x = 4.0 * tx**3 - 3.0 * tx
    
    T0_y = np.ones_like(ty)
    T1_y = ty
    T2_y = 2.0 * ty**2 - 1.0
    T3_y = 4.0 * ty**3 - 3.0 * ty
    
    T0_z = np.ones_like(tz)
    T1_z = tz
    T2_z = 2.0 * tz**2 - 1.0
    T3_z = 4.0 * tz**3 - 3.0 * tz
    
    # Basis selection:
    # We want to cover the most likely functional forms.
    # The previous program used up to degree 2 univariate and degree 1 interactions.
    # Let's expand to include degree 3 univariate terms and more interactions.
    # However, we only have 10 parameters.
    
    # Strategy:
    # Use a mix that captures:
    # 1. Constant
    # 2. Linear terms (x, y, z)
    # 3. Quadratic pure terms (x^2, y^2, z^2)
    # 4. Interaction terms (xy, xz, yz)
    # This is the complete quadratic basis, which scored 7.3930 (similar to current).
    
    # The current program (score 7.3927) uses:
    # 0: 1
    # 1: T1_x
    # 2: T1_y
    # 3: T1_z
    # 4: T2_x
    # 5: T2_y
    # 6: T2_z
    # 7: T1_x * T1_y
    # 8: T1_x * T1_z
    # 9: T1_y * T1_z
    
    # Note: T1_x is linear in x. T2_x is quadratic in x.
    # T1_x * T1_y is bilinear in x,y.
    # This basis is essentially the complete quadratic polynomial basis,
    # just expressed in Chebyshev coordinates.
    # T1_x = (2x-2.5)/1.5 = (2/1.5)x - 2.5/1.5
    # T2_x = 2((2x-2.5)/1.5)^2 - 1 = quadratic in x.
    # So the span is the same as {1, x, y, z, x^2, y^2, z^2, xy, xz, yz}.
    
    # To improve, we need to go beyond quadratic polynomials.
    # Let's try including some cubic terms or higher-order interactions.
    # But BFGS from all ones might struggle if the basis is too complex or ill-conditioned.
    
    # Alternative: Use a rational function basis or power law basis as seen in inspiration 1.
    # Inspiration 1 used: p0 * x^p1 * y^p2 * z^p3 + p4 * x^p5 * y^p6 * z^p7 + p8 * (x+y+z) + p9
    # This allows for non-integer powers, which can capture power-law behaviors.
    # This is a different structural family: non-linear in parameters (power laws).
    # BFGS can handle this, but it's a non-convex optimization problem.
    # The score of inspiration 1 was very bad (32700491930), likely due to poor initialization or local minima.
    
    # Let's try a hybrid: Keep the robust polynomial basis but add a few non-linear terms.
    # Or, try a different polynomial basis that might be better conditioned.
    
    # Actually, let's look at the "stagnation" directive.
    # "Make a STRUCTURAL change: a different design or modelling family..."
    # The current program is a linear model in parameters with a quadratic polynomial basis.
    # Let's try a model that is non-linear in parameters, like a sum of power laws,
    # but with better initialization or a simpler structure to avoid local minima.
    
    # Let's try: f(x,y,z) = p0 * x^p1 * y^p2 * z^p3 + p4
    # This is a single power law term plus a constant.
    # 5 parameters. We have 10.
    # Let's try: f(x,y,z) = p0 * x^p1 * y^p2 * z^p3 + p4 * x^p5 * y^p6 * z^p7 + p8 * (x+y+z) + p9
    # This was inspiration 1. It failed. Why?
    # Maybe the initialization at all ones is bad for power laws.
    # If p1=1, p2=1, p3=1, then x*y*z is in [0.125, 8].
    # If the target is e.g. x+y+z, the power law term will try to fit it poorly.
    
    # Let's try a different structural change: Use a neural-network-like basis with sigmoid activations.
    # But we only have 10 params, so we can't have a full network.
    # Maybe: sum of a few sigmoid terms.
    # f(x,y,z) = p0 * sigmoid(p1*x + p2*y + p3*z + p4) + p5 * sigmoid(p6*x + p7*y + p8*z + p9)
    # This is a 2-layer network with 2 hidden units (or just 2 logistic terms).
    # This is non-linear in parameters.
    # Let's try this.
    
    # Basis:
    # term1 = sigmoid(p[1]*x + p[2]*y + p[3]*z + p[4])
    # term2 = sigmoid(p[6]*x + p[7]*y + p[8]*z + p[9])
    # result = p[0] * term1 + term2  # Or p[0]*term1 + p[5]*term2?
    # We have 10 params.
    # Let's use: result = p[0] * sigmoid(p[1]*x + p[2]*y + p[3]*z + p[4]) + p[5] * sigmoid(p[6]*x + p[7]*y + p[8]*z + p[9])
    
    # This allows for non-linear, saturating behaviors.
    # The inputs are in [0.5, 2].
    # If p[1..3] are small, sigmoid is roughly linear.
    # If p[1..3] are large, it becomes a step function.
    
    # Let's implement this.
    
    # Compute the linear combinations
    lin1 = p[1] * x + p[2] * y + p[3] * z + p[4]
    lin2 = p[6] * x + p[7] * y + p[8] * z + p[9]
    
    # Apply sigmoid, clipping to avoid overflow in exp
    # sigmoid(t) = 1 / (1 + exp(-t))
    # For large positive t, exp(-t) -> 0, sigmoid -> 1
    # For large negative t, exp(-t) -> inf, sigmoid -> 0
    # We can use np.clip on the input to sigmoid to be safe, or just use a stable sigmoid implementation.
    
    def stable_sigmoid(t):
        # t is an array
        # For t > 0: sigmoid(t) = 1 / (1 + exp(-t))
        # For t <= 0: sigmoid(t) = exp(t) / (1 + exp(t))
        out = np.empty_like(t)
        pos = t > 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
        exp_t = np.exp(t[neg])
        out[neg] = exp_t / (1.0 + exp_t)
        return out
    
    sig1 = stable_sigmoid(lin1)
    sig2 = stable_sigmoid(lin2)
    
    result = p[0] * sig1 + sig2
    
    # Note: p[5] is unused in this formulation. Let's use it.
    # result = p[0] * sig1 + p[5] * sig2
    # But then we have 10 params: p0, p1-p4, p5, p6-p9.
    # Let's adjust:
    # term1 = sigmoid(p[1]*x + p[2]*y + p[3]*z + p[4])
    # term2 = sigmoid(p[6]*x + p[7]*y + p[8]*z + p[9])
    # result = p[0] * term1 + p[5] * term2
    
    result = p[0] * sig1 + p[5] * sig2
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
