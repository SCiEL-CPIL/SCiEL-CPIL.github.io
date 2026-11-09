"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk4(f, t0, t1, y0, steps, deadline=None):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            return y, True
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return y, False
    return y, False


def _backward_euler(f, t0, t1, y0, steps, deadline=None):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    eye = np.eye(n)
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            return y
        z = y.copy()
        for _newton in range(4):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            # Solve (I - hJ) step = -(z - y - h*fz)
            # Matrix is sparse? No, we use dense.
            # If n=100, 100x100 solve is fast.
            try:
                step = np.linalg.solve(eye - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                step = np.zeros(n)
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Benchmark to determine max steps
    # Use a small number of steps to estimate cost
    # We want to be safe, so we benchmark 10 steps
    h_bench = (t1 - t0) / 100
    t_bench = t0
    y_bench = y0.copy()
    
    start_time = time.time()
    # Run 10 RK4 steps
    for _ in range(10):
        k1 = f(t_bench, y_bench)
        k2 = f(t_bench + h_bench / 2, y_bench + h_bench / 2 * k1)
        k3 = f(t_bench + h_bench / 2, y_bench + h_bench / 2 * k2)
        k4 = f(t_bench + h_bench, y_bench + h_bench * k3)
        y_bench = y_bench + h_bench / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_bench += h_bench
        if not np.all(np.isfinite(y_bench)):
            # If it blows up immediately, we can't benchmark time accurately, 
            # but we should still try to solve. 
            # Fall back to seed behavior or just use a moderate N.
            pass
    end_time = time.time()
    time_per_step = (end_time - start_time) / 10
    
    # Calculate max steps based on time (leave 50s margin)
    max_time = 250.0
    if time_per_step > 0:
        max_steps_by_time = int(max_time / time_per_step)
    else:
        max_steps_by_time = 500000
    
    # Max steps based on call limit (2,000,000 calls / 4 calls per step)
    max_steps_by_calls = 500000
    
    N = min(max_steps_by_time, max_steps_by_calls)
    # Ensure at least 2000 steps (seed level)
    N = max(N, 2000)
    
    # Set a deadline for the main integration
    deadline = time.time() + 280.0
    
    y, timed_out = _rk4(f, t0, t1, y0, N, deadline)
    
    # Check validity
    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
        # Fall back to implicit Euler
        # Implicit Euler is more expensive. 
        # Cost per step is ~5 (1 for fz, 100 for J) * 4 iterations = ~400 calls? 
        # Wait, J calculation is n calls. n=100. 
        # So 1 + 100 = 101 calls per Newton iteration. 4 iterations = 404 calls per step.
        # Max steps for implicit: 2,000,000 / 404 ~ 4950 steps.
        # Time cost: 4950 steps * (100 solves + 101 f calls).
        # This is very expensive. 
        # If RK4 fails, we are in trouble. 
        # But RK4 is very robust for non-stiff. 
        # Let's just use a smaller N for implicit, e.g., 1000 steps.
        N_impl = 1000
        y = _backward_euler(f, t0, t1, y0, N_impl, deadline)
        
    return y
# EVOLVE-BLOCK-END
