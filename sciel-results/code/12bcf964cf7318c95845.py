import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Parameters:
        # p[0:4]: Weights for 4 Corner RBFs
        # p[4]: log-width for x
        # p[5]: log-width for y
        # p[6]: Bias
        # p[7]: Linear X
        # p[8]: Linear Y
        # p[9]: Quadratic X^2 + Y^2 (or separate? We have 1 param left)
        # Let's use p[9] for X^2 and maybe we need Y^2?
        # We only have 1 param left if we use 4 RBFs + 2 widths + 3 poly (bias, lin x, lin y).
        # So p[9] can be X^2. What about Y^2?
        # Maybe use p[9] for X^2 - Y^2? Or just X^2?
        # If the function is symmetric, X^2 and Y^2 might be correlated.
        # Let's try:
        # p[0:4]: Weights
        # p[4]: log_sx
        # p[5]: log_sy
        # p[6]: Bias
        # p[7]: Lin X
        # p[8]: Lin Y
        # p[9]: X^2 (and assume Y^2 is handled by RBFs or is similar?)
        
        # Alternative:
        # p[0:3]: Weights for 3 centers? No.
        
        # Let's stick to 5 RBFs (Model A) but change the polynomial part?
        # Model A: 5 RBFs, 2 widths, Bias, Lin X, Lin Y.
        # This is p95.
        
        # What if we use 4 RBFs (Corners) and 1 Center RBF? That's 5.
        # What if we use 4 RBFs (Corners) and 2 Center RBFs? No.
        
        # Let's try:
        # 5 RBFs (Corners + Center)
        # 1 Width (Isotropic)
        # 1 Bias
        # 1 Lin X
        # 1 Lin Y
        # 1 Quad X
        # 1 Quad Y
        # 1 Quad XY
        # Total: 5+1+1+1+1+1+1+1 = 12. Too many.
        
        # Back to 10 params.
        # 5 RBFs
        # 1 Width
        # 1 Bias
        # 1 Lin X
        # 1 Lin Y
        # 1 Quad X
        # 1 Quad Y
        # Total: 10.
        # This drops the XY term and uses isotropic width.
        
        w = params[0:5]
        log_s = params[5]
        b0 = params[6]
        b1 = params[7]
        b2 = params[8]
        b3 = params[9] # Quad X?
        
        # We need Quad Y too?
        # We don't have a param for Quad Y if we use 5 RBFs + 1 Width + 1 Bias + 1 Lin X + 1 Lin Y + 1 Quad X.
        # That's 10.
        
        # Is Quad X enough?
        # If the function is $x^2 + y^2$, we can't fit $y^2$ well without a param.
        # But the RBFs can approximate $y^2$.
        
        # Let's try this Model C:
        # 5 RBFs (Corners + Center)
        # 1 Width
        # 1 Bias
        # 1 Lin X
        # 1 Lin Y
        # 1 Quad X
        # 1 Quad Y
        # Wait, 5+1+1+1+1+1+1 = 11.
        
        # Okay, 4 RBFs (Corners)
        # 1 Width
        # 1 Bias
        # 1 Lin X
        # 1 Lin Y
        # 1 Quad X
        # 1 Quad Y
        # 1 Quad XY
        # Total: 4+1+1+1+1+1+1+1 = 10.
        
        # This seems like a strong candidate.
        # 4 Corners capture the boundaries.
        # Polynomials capture the global trend.
        # Isotropic width is simpler.
        
        w = params[0:4]
        log_s = params[4]
        b0 = params[5]
        b1 = params[6]
        b2 = params[7]
        b3 = params[8] # X^2
        b4 = params[9] # Y^2
        
        # Where is XY? I skipped it.
        # If I use 4 RBFs, 1 Width, 1 Bias, 1 Lin X, 1 Lin Y, 1 Quad X, 1 Quad Y.
        # That's 4+1+1+1+1+1+1 = 10.
        # Yes.
        
        s = np.exp(log_s)
        
        k1 = np.exp(-((x + 1.0)**2 + (y + 1.0)**2) / (2.0 * s**2))
        k2 = np.exp(-((x - 1.0)**2 + (y + 1.0)**2) / (2.0 * s**2))
        k3 = np.exp(-((x + 1.0)**2 + (y - 1.0)**2) / (2.0 * s**2))
        k4 = np.exp(-((x - 1.0)**2 + (y - 1.0)**2) / (2.0 * s**2))
        
        p0 = np.ones_like(x)
        p1 = x
        p2 = y
        p3 = x * x
        p4 = y * y
        
        return (w[0] * k1 +
                w[1] * k2 +
                w[2] * k3 +
                w[3] * k4 +
                b0 * p0 +
                b1 * p1 +
                b2 * p2 +
                b3 * p3 +
                b4 * p4)
