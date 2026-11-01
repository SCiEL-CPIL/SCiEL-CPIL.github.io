"""Improved packing of 50 equal circles in a unit square via force-directed relaxation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import random

def _relax(centres, r, max_iter=200, damping=0.5):
    """
    Relax the positions of the circles to remove overlaps for a given radius r.
    Returns the final centres and a boolean indicating if no overlaps remain.
    """
    centres = centres.copy()
    
    for _ in range(max_iter):
        moved = False
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = centres[j][0] - centres[i][0]
                dy = centres[j][1] - centres[i][1]
                dist_sq = dx*dx + dy*dy
                min_dist = 2 * r
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        # Random nudge if coincident
                        centres[i][0] += random.uniform(-0.01, 0.01)
                        centres[i][1] += random.uniform(-0.01, 0.01)
                        centres[j][0] += random.uniform(-0.01, 0.01)
                        centres[j][1] += random.uniform(-0.01, 0.01)
                        moved = True
                    else:
                        # Unit vector from i to j
                        ux = dx / dist
                        uy = dy / dist
                        # Overlap amount
                        overlap = min_dist - dist
                        # Push each by half the overlap, with damping
                        shift = (overlap / 2) * damping
                        centres[i][0] -= ux * shift
                        centres[i][1] -= uy * shift
                        centres[j][0] += ux * shift
                        centres[j][1] += uy * shift
                        moved = True
        
        # Boundary constraints
        for i in range(N):
            # x boundary
            if centres[i][0] < -0.5 + r:
                centres[i][0] = -0.5 + r
                moved = True
            elif centres[i][0] > 0.5 - r:
                centres[i][0] = 0.5 - r
                moved = True
            
            # y boundary
            if centres[i][1] < -0.5 + r:
                centres[i][1] = -0.5 + r
                moved = True
            elif centres[i][1] > 0.5 - r:
                centres[i][1] = 0.5 - r
                moved = True
        
        if not moved:
            break

    # Check if configuration is valid (no overlaps)
    valid = True
    for i in range(N):
        if centres[i][0] < -0.5 + r - 1e-9 or centres[i][0] > 0.5 - r + 1e-9:
            valid = False
            break
        if centres[i][1] < -0.5 + r - 1e-9 or centres[i][1] > 0.5 - r + 1e-9:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                dx = centres[j][0] - centres[i][0]
                dy = centres[j][1] - centres[i][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < (2 * r - 1e-9) * (2 * r - 1e-9):
                    valid = False
                    break
            if not valid:
                break
                
    return centres, valid

def _construct_grid():
    """Create a grid-based initial configuration."""
    # 8 columns, 7 rows = 56 points, take first 50
    ticks_x = np.linspace(-0.5 + 0.05, 0.5 - 0.05, 8)
    ticks_y = np.linspace(-0.5 + 0.05, 0.5 - 0.05, 7)
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N]

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    best_r = 0.0
    best_centres = np.zeros((N, 2))
    
    # Use grid initialization
    initial_centres = _construct_grid()
    
    num_restarts = 5
    for restart in range(num_restarts):
        np.random.seed(restart * 12345)
        random.seed(restart * 12345)
        
        # Perturb the grid slightly to break symmetry
        centres = initial_centres + np.random.uniform(-0.01, 0.01, (N, 2))
        centres = np.clip(centres, -0.5, 0.5)
        
        # Start with a radius smaller than the grid's minimum distance
        # Grid spacing is approx 1/7 ~ 0.14, so radius ~ 0.07
        # Start at 0.05 and go up to 0.08
        r = 0.05
        max_r = 0.08
        step = 0.0005
        
        current_centres = centres.copy()
        last_valid_r = 0.0
        
        while r <= max_r:
            # Try to relax with current radius
            relaxed_centres, valid = _relax(current_centres, r, max_iter=100)
            
            if valid:
                last_valid_r = r
                current_centres = relaxed_centres
                r += step
            else:
                # If we fail, try a smaller step or break
                # If we are close to max, we might be stuck.
                # Let's try one more time with more iterations?
                relaxed_centres, valid = _relax(current_centres, r, max_iter=500)
                if valid:
                    last_valid_r = r
                    current_centres = relaxed_centres
                    r += step
                else:
                    break
        
        if last_valid_r > best_r:
            best_r = last_valid_r
            best_centres = current_centres
            
    return best_centres
# EVOLVE-BLOCK-END
