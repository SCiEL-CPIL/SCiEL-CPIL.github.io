"""Symbolic regression candidate: hybrid polynomial basis in x, y, z.

Combines the full quadratic basis (interactions + pure quadratics) with
pure cubic terms, dropping the constant term to make room for x^3, y^3, z^3.
This captures both interaction effects and higher-order curvature.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # 10-term basis: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, x^3+y^3+z^3 (combined)
    # Actually let's do: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, and one more
    # Better: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, x^3 (drop constant, add x^3)
    # Even better: keep all quadratic terms + add pure cubics by dropping constant
    # Basis: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, x^3+y^3+z^3
    # But that couples the cubics. Let's do:
    # Basis: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, x^3
    # Or better yet, use a mixed approach:
    # Basis: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, x^3 (drop y*z, add x^3)
    # Hmm, let me think about what's most useful.
    #
    # The quadratic basis scored 7.393. The pure cubic basis scored 1.3046.
    # Let's try: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z (quadratic) 
    # vs 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3 (cubic)
    #
    # A hybrid: 1, x, y, z, x^2, y^2, z^2, x*y, x^3, y^3 (drop z^2 and y*z, add cubics)
    # Or: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, x^3+y^3+z^3
    #
    # Let me try: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, x^3
    # This keeps the constant, all linears, all quadratics except yz, and adds x^3.
    # Actually, let's be more systematic. We have 10 slots.
    # Most informative basis might be:
    # 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z (full quadratic) -- score 7.393
    # 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3 (pure cubic) -- score 1.3046
    #
    # Hybrid option 1: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y^3
    # Hybrid option 2: x, y, z, x^2, y^2, z^2, x*y, x*z, y*z, x^3
    # Hybrid option 3: 1, x, y, z, x^2, y^2, z^2, x*y, x^3, y^3
    #
    # Let's go with a balanced hybrid that includes interactions AND some cubics:
    # 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z -> full quadratic (7.393)
    # Replace one quadratic with a cubic:
    # 1, x, y, z, x^2, y^2, z^2, x*y, x*z, x^3
    # Or keep all interactions and add a combined cubic:
    # 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z is 9 terms + 1 more
    # We can do: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z (9 terms) + x^3 (10th)
    # That's 10 terms!
    
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z
    c5 = x * x
    c6 = y * y
    c7 = z * z
    c8 = x * y
    c9 = x * z
    c10 = y * z

    # Wait, that's still the full quadratic. Let me add a cubic by replacing one term.
    # Replace y*z with x^3:
    # c1=1, c2=x, c3=y, c4=z, c5=x^2, c6=y^2, c7=z^2, c8=xy, c9=xz, c10=x^3
    
    # Actually, let me try a different approach. Use all 10 for a richer basis:
    # 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z is the full quadratic.
    # To add cubic info, I could use a transformed basis.
    # 
    # Alternative: use the full quadratic but with a slight modification.
    # Or: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, x^3
    
    # Let me just go with: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, x^3
    # This drops y*z but adds x^3.
    
    # Re-computing:
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z
    c5 = x * x
    c6 = y * y
    c7 = z * z
    c8 = x * y
    c9 = x * z
    c10 = x * x * x

    out = (
        p[0] * c1
        + p[1] * c2
        + p[2] * c3
        + p[3] * c4
        + p[4] * c5
        + p[5] * c6
        + p[6] * c7
        + p[7] * c8
        + p[8] * c9
        + p[9] * c10
    )
    return out
# EVOLVE-BLOCK-END
