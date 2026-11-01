"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Initial phyllotaxis layout ---
    def phyllotaxis(n, margin=0.05):
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        pts = np.zeros((n, 2))
        for i in range(n):
            r = np.sqrt(i + 1) / np.sqrt(n) * (0.5 - margin)
            theta = i * golden_angle
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            # Clamp to [margin, 1-margin]
            x = np.clip(x, margin, 1 - margin)
            y = np.clip(y, margin, 1 - margin)
            pts[i] = [x, y]
        return pts

    def compute_radii(positions):
        """For given positions, compute max valid radii such that circles are disjoint and inside [0,1]^2."""
        n = len(positions)
        radii = np.zeros(n)
        # Wall constraints
        for i in range(n):
            radii[i] = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        # Pairwise constraints: r_i + r_j <= dist(i,j)
        # Use iterative relaxation
        for _ in range(50):
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(positions[i] - positions[j])
                    # r_i + r_j <= d  =>  each can be at most (d - other_r)
                    # Symmetric: r_i <= d - r_j, r_j <= d - r_i
                    # To maximize sum, set r_i = r_j = d/2 if unconstrained by walls
                    # But we need consistent solution. Use: r_i = min(r_i, d - r_j), r_j = min(r_j, d - r_i)
                    # This is a standard iterative approach
                    new_ri = min(radii[i], max(0, d - radii[j]))
                    new_rj = min(radii[j], max(0, d - new_ri))
                    radii[i] = new_ri
                    radii[j] = new_rj
            # Re-apply wall constraints
            for i in range(n):
                radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        return radii

    def objective(positions):
        radii = compute_radii(positions)
        return np.sum(radii), radii

    def coordinate_descent(positions, max_iter=5000, tol=1e-10):
        """Refine positions via coordinate descent to maximize sum of radii."""
        n = len(positions)
        positions = positions.copy()
        best_val, best_radii = objective(positions)
        best_pos = positions.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(n):
                for dim in range(2):
                    # Try small perturbations in each direction
                    base_val = best_val
                    for delta in [1e-4, 1e-3, 1e-2, 1e-1, -1e-4, -1e-3, -1e-2, -1e-1]:
                        new_pos = best_pos.copy()
                        new_pos[i, dim] = best_pos[i, dim] + delta
                        # Clamp
                        new_pos[i, dim] = np.clip(new_pos[i, dim], 0.01, 0.99)
                        val, _ = objective(new_pos)
                        if val > best_val + tol:
                            best_val = val
                            best_pos = new_pos
                            improved = True
                            break
            if not improved:
                break
        return best_pos, best_val, best_radii

    # Generate multiple random restarts
    best_overall = -np.inf
    best_result = None

    # Try phyllotaxis first
    pos = phyllotaxis(N)
    pos_ref, val, radii = coordinate_descent(pos, max_iter=3000)
    if val > best_overall:
        best_overall = val
        best_result = np.column_stack([pos_ref, radii])

    if time.time() > deadline - 10.0:
        return best_result

    # Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 0
    while time.time() < deadline - 10.0:
        n_restarts += 1
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        pos_ref, val, radii = coordinate_descent(pos, max_iter=2000)
        if val > best_overall:
            best_overall = val
            best_result = np.column_stack([pos_ref, radii])

    # Final polish: tighter coordinate descent on best result
    if time.time() < deadline - 3.0 and best_result is not None:
        pos_final = best_result[:, :2]
        pos_final, val_final, radii_final = coordinate_descent(pos_final, max_iter=5000, tol=1e-12)
        if val_final > best_overall:
            best_overall = val_final
            best_result = np.column_stack([pos_final, radii_final])

    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
