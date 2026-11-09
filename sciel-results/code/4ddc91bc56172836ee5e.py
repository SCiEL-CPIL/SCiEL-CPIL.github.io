"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    t0, t1 = t_span
    dt = t1 - t0
    n = len(y0)
    y = np.array(y0, dtype=float).copy()
    
    # First call to satisfy the "f must be called at least once" requirement
    # We'll fold this into the integration loop to avoid wasting a call,
    # but we need to ensure at least one call happens.
    # Actually, RK4 calls f 4 times per step, so the first step covers it.
    # But if steps=0, we need a call. Here steps will be > 0.
    
    # The problem is linear and well-behaved.
    # A = 21^2 * tridiag(1, -2, 1).
    # Eigenvalues of tridiag(1, -2, 1) for N=20 are -4*sin^2(k*pi/(2*(N+1))) for k=1..N.
    # Largest magnitude eigenvalue of A is approx 21^2 * 4 * sin^2(20*pi/42) ~ 441 * 4 * 1 ~ 1764.
    # Stability limit for RK4 is h * lambda < 2.78 (approx).
    # So h < 2.78 / 1764 ~ 1.57e-3.
    # Total time T = 0.1.
    # Min steps ~ 0.1 / 1.57e-3 ~ 64.
    # RK4 error is O(h^5). With h ~ 1e-3, error is ~ 1e-15.
    # Let's use more steps to be safe and reduce error further.
    # 1000 steps gives h = 1e-4. Error ~ 1e-20 (limited by machine precision).
    # 2000 steps gives h = 5e-5. Error ~ 1e-25.
    # Let's aim for high accuracy with reasonable call count.
    # Current program uses 1000 steps -> 4000 calls.
    # Score = -log10(rel) - 0.5*log10(calls).
    # If rel ~ 1e-14 (floor), score = 14 - 0.5*log10(calls).
    # To maximize score, we want calls as small as possible while keeping rel low.
    # If we can get rel ~ 1e-10 with 100 calls, score = 10 - 0.5*2 = 9.
    # If we get rel ~ 1e-14 with 4000 calls, score = 14 - 0.5*3.6 = 12.2.
    # So high accuracy is worth the calls.
    
    # Let's try an adaptive approach or just a fixed high-step count.
    # Given the stiffness is moderate (lambda_max ~ 1764, T=0.1), RK4 is stable for h < 1e-3.
    # Let's use 200 steps (h=5e-4). Stable? 1764*5e-4 = 0.88 < 2.78. Yes.
    # Error for 200 steps: h^5 ~ (5e-4)^5 ~ 3e-18. Very small.
    # Calls = 200 * 4 = 800.
    # Score ~ 14 - 0.5*log10(800) ~ 14 - 1.4 = 12.6.
    
    # Can we do fewer steps?
    # 100 steps (h=1e-3). 1764*1e-3 = 1.76 < 2.78. Stable.
    # Error ~ (1e-3)^5 = 1e-15.
    # Calls = 400.
    # Score ~ 14 - 0.5*2.6 = 12.7.
    
    # 50 steps (h=2e-3). 1764*2e-3 = 3.5 > 2.78. Unstable.
    # So 100 steps is near the stability limit.
    # Let's use 100 steps to minimize calls while maintaining stability and high accuracy.
    
    steps = 100
    h = dt / steps
    t = t0
    
    for _ in range(steps):
        # RK4 step
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        # Deadline check
        if time.time() > deadline:
            break
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y)):
            # Fallback to smaller steps? Or just return what we have?
            # Given the problem is well-behaved, this shouldn't happen.
            # But if it does, break.
            break

    return y
# EVOLVE-BLOCK-END
