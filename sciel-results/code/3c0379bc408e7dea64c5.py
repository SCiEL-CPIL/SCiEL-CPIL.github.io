import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to roughly [-1, 1]
    # Domain is [0.5, 2.0], center 1.25, range 1.5, half-range 0.75
    xc = (x - 1.25) / 0.75
    yc = (y - 1.25) / 0.75
    zc = (z - 1.25) / 0.75

    # Build a basis that combines:
    # 1. Constant term
    # 2. Linear terms
    # 3. Key interaction terms (from inspiration 2's structure but normalized)
    # 4. Key non-linear terms (from current program's cubic/quadratic structure)
    #
    # We have exactly 10 parameters. Let's select a diverse set that covers
    # likely function structures: linear, quadratic, cubic, and interactions.
    #
    # b0: constant
    # b1: xc
    # b2: yc
    # b3: zc
    # b4: xc * yc
    # b5: xc * zc
    # b6: yc * zc
    # b7: xc * yc * zc
    # b8: xc * xc
    # b9: yc * yc
    #
    # Wait, the current program scored 1.0823 with:
    # b0=1, b1=xc, b2=yc, b3=zc, b4=xc^3, b5=yc^3, b6=zc^3, b7=xc*yc*zc, b8=xc^2*yc^2, b9=yc^2*zc^2
    #
    # Inspiration 2 scored 3.1997 with:
    # f0=1, f1=x, f2=y, f3=z, f4=xy, f5=xz, f6=yz, f7=xyz, f8=1/x, f9=1/y
    #
    # Inspiration 1 scored 7.3933 with:
    # b0=1, b1=xc, b2=yc, b3=zc, b4=xc^2, b5=yc^2, b6=zc^2, b7=xc*yc, b8=xc*zc, b9=yc*zc
    #
    # The current program (score 1.08) is the best so far. It uses cubics and high-order interactions.
    # However, it lacks pure quadratic terms (xc^2, etc.) and pure pair interactions (xc*yc, etc.).
    # Let's try to combine the strengths.
    #
    # A good candidate basis might be:
    # 0: 1
    # 1: xc
    # 2: yc
    # 3: zc
    # 4: xc*yc
    # 5: xc*zc
    # 6: yc*zc
    # 7: xc*yc*zc
    # 8: xc^2
    # 9: yc^2
    #
    # But wait, the current program has cubics. Maybe the target function is polynomial of degree 3?
    # If so, we need up to degree 3.
    # Full polynomial degree 3 in 3 variables has 20 terms (1 + 3 + 6 + 10). We only have 10 params.
    # So we must choose the most likely terms.
    #
    # Let's look at the current program's basis again:
    # 1, xc, yc, zc, xc^3, yc^3, zc^3, xc*yc*zc, xc^2*yc^2, yc^2*zc^2
    # This is quite sparse.
    #
    # Let's try a basis that includes:
    # 1, xc, yc, zc, xc^2, yc^2, zc^2, xc*yc, xc*zc, yc*zc
    # This is a full quadratic model (10 terms). This is a very standard and robust choice.
    # Inspiration 1 used exactly this (with normalized inputs) and scored 7.39.
    # Why did it score worse than the current program (1.08)?
    # Maybe the target function has higher order terms?
    #
    # Let's try to mix in some cubic terms while keeping the quadratic structure.
    # Basis:
    # 0: 1
    # 1: xc
    # 2: yc
    # 3: zc
    # 4: xc^2
    # 5: yc^2
    # 6: zc^2
    # 7: xc*yc
    # 8: xc*zc
    # 9: yc*zc
    # This is the quadratic basis.
    #
    # What if we replace some quadratic terms with cubic terms?
    # The current program replaced:
    # xc^2 -> xc^3
    # yc^2 -> yc^3
    # zc^2 -> zc^3
    # xc*yc -> xc*yc*zc
    # xc*zc -> xc^2*yc^2
    # yc*zc -> yc^2*zc^2
    #
    # This seems arbitrary. Let's try a different mix.
    # Maybe:
    # 0: 1
    # 1: xc
    # 2: yc
    # 3: zc
    # 4: xc^2
    # 5: yc^2
    # 6: zc^2
    # 7: xc*yc*zc
    # 8: xc^3
    # 9: yc^3
    #
    # Or perhaps we should consider that BFGS starts from all ones.
    # If the true function is simple, a simpler basis might converge better.
    #
    # Let's try the full quadratic basis again, but maybe with a different scaling?
    # No, normalization is key.
    #
    # Let's look at the scores again.
    # Current (1.08): 1, x, y, z, x^3, y^3, z^3, xyz, x^2y^2, y^2z^2
    # Insp 1 (7.39): 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    # Insp 2 (3.20): 1, x, y, z, xy, xz, yz, xyz, 1/x, 1/y
    #
    # The current program is significantly better. This suggests the function is not a simple quadratic.
    # It might be a cubic or have strong interaction terms.
    #
    # Let's try to include more interaction terms and cubics.
    # Basis:
    # 0: 1
    # 1: xc
    # 2: yc
    # 3: zc
    # 4: xc^2
    # 5: yc^2
    # 6: zc^2
    # 7: xc*yc
    # 8: xc*zc
    # 9: yc*zc
    # This is the quadratic basis. It scored 7.39.
    #
    # What if we use:
    # 0: 1
    # 1: xc
    # 2: yc
    # 3: zc
    # 4: xc*yc
    # 5: xc*zc
    # 6: yc*zc
    # 7: xc*yc*zc
    # 8: xc^2
    # 9: yc^2
    #
    # Let's try to combine the "best" terms from the current program and the quadratic basis.
    # The current program has: 1, x, y, z, x^3, y^3, z^3, xyz, x^2y^2, y^2z^2
    #
    # Maybe the function is separable? f(x,y,z) = g(x) + h(y) + k(z)?
    # If so, we only need terms in x, y, z individually.
    # Basis: 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3
    # This allows up to cubic in each variable independently.
    #
    # Let's try this basis.
    # 0: 1
    # 1: xc
    # 2: yc
    # 3: zc
    # 4: xc^2
    # 5: yc^2
    # 6: zc^2
    # 7: xc^3
    # 8: yc^3
    # 9: zc^3

    b0 = np.ones_like(x)
    b1 = xc
    b2 = yc
    b3 = zc
    b4 = xc * xc
    b5 = yc * yc
    b6 = zc * zc
    b7 = xc * xc * xc
    b8 = yc * yc * yc
    b9 = zc * zc * zc

    return (
        params[0] * b0
        + params[1] * b1
        + params[2] * b2
        + params[3] * b3
        + params[4] * b4
        + params[5] * b5
        + params[6] * b6
        + params[7] * b7
        + params[8] * b8
        + params[9] * b9
    )
