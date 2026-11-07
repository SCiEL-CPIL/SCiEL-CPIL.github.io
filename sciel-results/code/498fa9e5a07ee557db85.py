import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params is expected to be of length 10
    # Strategy: Rational function structure with a richer basis.
    # Instead of a simple linear combination (current) or a simple linear/linear rational function (inspiration 1),
    # we use a structure that can represent more complex rational behaviors while maintaining numerical stability.
    
    # The idea: 
    # Numerator: A linear combination of a rich basis of terms.
    # Denominator: 1 + a small linear combination to allow for gentle curvature changes.
    # 
    # Why this structure?
    # 1. The current program (linear combination) scored ~10.
    # 2. Inspiration 1 (simple rational) scored very poorly (~13546), likely due to instability or poor initial fit.
    # 3. Inspiration 3 (centered rational) also scored poorly.
    # 4. The key insight from the "findings" is that ratio terms (x/y, etc.) or richer bases help.
    # 5. A rational function f = N/D where N is a linear combo of 1, x, y, z, xy, xz, yz, 1/x, 1/y, 1/z
    #    and D is 1 + small terms can approximate many functions.
    #    However, BFGS starting from all ones might struggle if the denominator is too complex.
    #    
    # Let's try a "soft" rational approach:
    # pred = (sum of basis * params) / (1 + epsilon * sum of small basis * params)
    # But we only have 10 params.
    
    # Let's allocate params:
    # p0-p6: Coefficients for the numerator basis: 1, x, y, z, xy, xz, yz
    # p7-p9: Coefficients for the denominator adjustment: x, y, z (multiplied by a small factor to keep den near 1 initially)
    
    p = params.flatten()
    if len(p) < 10:
        # Fallback
        return np.ones_like(x) * p[0] if len(p) > 0 else np.zeros_like(x)

    # Numerator Basis:
    # 1, x, y, z, xy, xz, yz
    # This captures polynomial behavior up to degree 2.
    
    # Denominator:
    # 1 + p7*x + p8*y + p9*z
    # If p7,p8,p9 are small, den ~ 1, and we get a polynomial.
    # If they grow, we get rational behavior.
    # To avoid instability, we can scale the denominator terms.
    # However, the params are optimized. If the denominator becomes 0, we crash.
    # The evaluator uses BFGS. If the loss becomes inf, BFGS might fail.
    
    # A safer structure:
    # Use a basis that includes reciprocals in the NUMERATOR, and keep the DENOMINATOR simple or 1.
    # Current program: 1, x, y, z, 1/x, 1/y, 1/z, xy, xz, yz. Score ~10.
    # 
    # Let's try a different structural change:
    # What if the function is of the form (a*x + b*y + c*z + d) / (e*x + f*y + g*z + h)?
    # That's 8 params. We have 10.
    # Let's try:
    # Numerator: p0 + p1*x + p2*y + p3*z + p4*xy + p5*xz + p6*yz
    # Denominator: 1 + p7*x + p8*y + p9*z
    # This is Inspiration 1's structure. It failed. Why?
    # Likely because starting from all ones, the denominator is 1+x+y+z, which is large (3.75 to 7).
    # The numerator is 1+x+y+z+xy+xz+yz.
    # The ratio might not match the target function's scale or shape well initially, leading to poor gradients.
    
    # Alternative:
    # What if we use a "tanh" or "sigmoid" like nonlinearity? No, we must be closed form with linear params.
    
    # Let's look at the "findings" again:
    # "Replacing 1/z and 1/y with x/y and y/z ratio terms captures more general rational structure"
    # This suggests the basis should include ratios.
    
    # Let's try a basis with ratios in a linear combination (not a rational function division).
    # Basis: 1, x, y, z, x/y, y/z, z/x, xy, xz, yz
    # This is a linear model but with a richer feature set.
    # This avoids the instability of the denominator in a rational function.
    # The current program used 1/x, 1/y, 1/z.
    # Ratios like x/y are bounded in [0.25, 4] for x,y in [0.5, 2].
    # 1/x is in [0.5, 2].
    # 
    # Let's try this basis:
    # 1, x, y, z, x/y, y/z, z/x, xy, xz, yz
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Compute terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = z
    term4 = x / y
    term5 = y / z
    term6 = z / x
    term7 = x * y
    term8 = x * z
    term9 = y * z
    
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
            p4 * term4 + p5 * term5 + p6 * term6 + 
            p7 * term7 + p8 * term8 + p9 * term9)
            
    return pred
