"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    rng = np.random.default_rng(42)
    
    # Initial positions: jittered grid to spread circles
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side + (rng.random() - 0.5) * 0.1 / side
            y = (j + 0.5) / side + (rng.random() - 0.5) * 0.1 / side
            positions.append([x, y])
    positions = np.array(positions[:N])
    radii = np.full(N, 0.01)
    
    def relax(positions, radii, iters=200):
        """Relax positions to resolve overlaps and boundary violations."""
        for _ in range(iters):
            moved = False
            # Handle boundary violations
            for i in range(N):
                r = radii[i]
                x, y = positions[i]
                if x < r:
                    positions[i, 0] = r
                    moved = True
                if x > 1 - r:
                    positions[i, 0] = 1 - r
                    moved = True
                if y < r:
                    positions[i, 1] = r
                    moved = True
                if y > 1 - r:
                    positions[i, 1] = 1 - r
                    moved = True
            
            # Handle overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist - 1e-10:
                        if dist < 1e-12:
                            # Nearly coincident, push apart randomly
                            angle = rng.random() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1e-8
                        overlap = min_dist - dist
                        # Push each apart by half the overlap
                        push = overlap / 2
                        nx = dx / dist
                        ny = dy / dist
                        positions[i, 0] -= push * nx
                        positions[i, 1] -= push * ny
                        positions[j, 0] += push * nx
                        positions[j, 1] += push * ny
                        moved = True
            if not moved:
                break
        # Final boundary clamp
        for i in range(N):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, 1 - r)
            positions[i, 1] = np.clip(positions[i, 1], r, 1 - r)
        return positions
    
    def is_feasible(positions, radii, tol=1e-8):
        for i in range(N):
            r = radii[i]
            x, y = positions[i]
            if x < r - tol or x > 1 - r + tol or y < r - tol or y > 1 - r + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    # Grow radii iteratively
    max_iter = 5000
    growth_step = 0.001
    for it in range(max_iter):
        # Try to grow the smallest radii
        min_idx = np.argmin(radii)
        # Try to grow all radii slightly
        old_radii = radii.copy()
        radii += growth_step
        # Relax positions
        positions = relax(positions, radii, iters=100)
        # Check feasibility
        if is_feasible(positions, radii, tol=1e-7):
            continue
        else:
            # Revert radii
            radii = old_radii
            # Try growing just the smallest one
            old_r = radii[min_idx]
            radii[min_idx] += growth_step
            positions = relax(positions, radii, iters=100)
            if is_feasible(positions, radii, tol=1e-7):
                continue
            else:
                radii[min_idx] = old_r
                # If we can't grow, try a smaller step
                growth_step *= 0.9
                if growth_step < 1e-5:
                    break
    
    # Final relaxation
    positions = relax(positions, radii, iters=500)
    
    # Ensure feasibility
    for _ in range(100):
        if is_feasible(positions, radii, tol=1e-9):
            break
        # Shrink radii slightly
        radii *= 0.999
        positions = relax(positions, radii, iters=200)
    
    result = np.column_stack([positions, radii])
    return result
# EVOLVE-BLOCK-END

