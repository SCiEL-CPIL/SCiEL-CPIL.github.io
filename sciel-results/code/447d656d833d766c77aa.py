def _solve_lambda_fast(delta, A, k, dt):
    C = np.log(A * dt)
    lam = A * dt * 0.5 # Initial guess
    for _ in range(10):
        term = lam * (delta + C - np.log(lam))
        if 1 + term <= 0:
            # Adjust lam to be smaller
            lam *= 0.5
            continue
        G = lam - np.log(1 + term)
        # Derivative
        # G'(lam) = 1 - (delta + C - ln(lam) - 1) / (1 + term)
        dG = 1.0 - (delta + C - np.log(lam) - 1.0) / (1.0 + term)
        if dG == 0: break
        new_lam = lam - G / dG
        if new_lam <= 0 or new_lam > A * dt:
            new_lam = np.clip(new_lam, 1e-12, A * dt)
        if abs(new_lam - lam) < 1e-8:
            lam = new_lam
            break
        lam = new_lam
    return np.clip(lam, 1e-12, A * dt)
