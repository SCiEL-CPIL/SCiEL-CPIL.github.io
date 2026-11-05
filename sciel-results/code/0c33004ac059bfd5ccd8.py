import numpy as np
import time
from scipy.optimize import minimize, basinhopping

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])

    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis):
        """
        Vectorized computation of minimum outer hexagon side length.
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]  # (6, 2)
            dots = verts @ normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(np.abs(dots)))

        return 2.0 * max_proj / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess based on known optimal configuration
    # All inner hexagons at 30 degrees (flat-top), outer at 30 degrees
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Define the local minimizer for Basin Hopping
    def local_minimizer(x):
        res = minimize(objective, x, method='Powell',
                       options={'maxiter': 100, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
        return res

    # Define the step generator (random kicks)
    def step_taker(x, stepsize):
        # Add random noise to each parameter
        # Step size is the standard deviation of the Gaussian noise
        x_new = x + stepsize * np.random.randn(len(x))
        return x_new

    # Basin Hopping optimization
    if time.time() < deadline - 10:
        try:
            # Use Basin Hopping with Powell as the local minimizer
            # T is the temperature, stepsize is the size of the random kicks
            bh = basinhopping(objective, x0, minimizer=local_minimizer,
                              stepsize=0.5, T=1000.0,
                              niter=50, accept_test=None,
                              step_taker=step_taker)
            
            if bh.fun < best_score:
                best_score = bh.fun
                best_params = bh.x.copy()
        except Exception as e:
            pass

    # Final polish with high-precision Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-10  # Tiny margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
