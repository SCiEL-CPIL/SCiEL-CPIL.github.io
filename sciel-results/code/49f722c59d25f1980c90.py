import numpy as np
import time

def construct():
        start_time = time.time()
        deadline = start_time + 110.0 # 10s margin
        
        # 1. Generate initial Golay-Rudin-Shapiro sequence
        p, q = [1], [1]
        while len(p) < 24:
            p, q = p + q, p + [-c for c in q]
        coeffs = np.array(p[:24], dtype=float)
        
        # 2. Helper to evaluate sup norm
        # Use dense sampling + refinement
        def sup_norm(c):
            # c is array of 24 floats (+1/-1)
            # Sample 2048 points
            thetas = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
            # Evaluate P(e^{i theta})
            # P(z) = sum c_k z^k
            # Using matrix multiplication or direct sum
            # Direct sum with broadcasting
            z = np.exp(1j * thetas)
            # c shape (24,), z shape (2048,)
            # We want sum_k c_k z^k for each theta
            # powers = z[:, None] ** np.arange(24) -> (2048, 24)
            # vals = powers @ c -> (2048,)
            powers = np.power(z[:, None], np.arange(24))
            vals = powers @ c
            abs_vals = np.abs(vals)
            
            # Find peaks
            # Simple: take max
            max_val = np.max(abs_vals)
            
            # Refine top few peaks
            # Get indices of top 5
            idxs = np.argsort(abs_vals)[-5:]
            refined_max = max_val
            for idx in idxs:
                theta0 = thetas[idx]
                # Golden section or ternary search in [theta0 - dtheta, theta0 + dtheta]
                dtheta = (2 * np.pi) / 2048
                a = theta0 - dtheta
                b = theta0 + dtheta
                # Ternary search for max of |P(e^{i theta})|
                # |P(e^{i theta})| is smooth
                for _ in range(20): # 20 iterations for high precision
                    m1 = a + (b - a) / 3
                    m2 = b - (b - a) / 3
                    f1 = np.abs(np.sum(c * np.exp(1j * m1 * np.arange(24))))
                    f2 = np.abs(np.sum(c * np.exp(1j * m2 * np.arange(24))))
                    if f1 < f2:
                        a = m1
                    else:
                        b = m2
                mid = (a + b) / 2
                f_mid = np.abs(np.sum(c * np.exp(1j * mid * np.arange(24))))
                if f_mid > refined_max:
                    refined_max = f_mid
            return refined_max

        current_norm = sup_norm(coeffs)
        
        # 3. Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(24):
                # Flip bit i
                new_coeffs = coeffs.copy()
                new_coeffs[i] *= -1
                new_norm = sup_norm(new_coeffs)
                if new_norm < current_norm - 1e-9:
                    coeffs = new_coeffs
                    current_norm = new_norm
                    improved = True
                    break # Restart scan from 0 to ensure convergence to local min
            
            # Check time
            if time.time() > deadline:
                break

        # Return as list of ints
        return [int(c) for c in coeffs]
